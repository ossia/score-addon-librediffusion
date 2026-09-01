#pragma once

// Model-agnostic frame production for the StreamDiffusion node.
//
//   * AsyncJob / AsyncFrame: everything one frame needs, and one finished frame (keyframe + optional
//     RIFE sweep). The same payload serves every model family.
//   * AsyncFrameProducer<Job, Frame>: a worker thread fed by a newest-wins triple buffer. It calls a
//     caller-supplied produce(Job&, Frame&) that does the heavy, BLOCKING GPU work. No CUDA event
//     crosses a thread: the library's inference calls sync their own stream before returning, and the
//     triple buffer is the host-side hand-off.
//   * PacedFrameConsumer: the render-side FIFO. In async mode it is drained by a fractional credit at
//     the MEASURED production rate, so repeats and skips spread evenly instead of stuttering; in sync
//     mode it simply yields one frame per tick.
//
// TensorRT contexts are single-thread: once a producer runs, the worker is the ONLY caller of that
// pipeline. Any render-thread mutation of the pipeline stops (drains + joins) the producer first.

// triple_buffer is the lock-free producer->consumer hand-off. Always use the
// vendored copy: it is API-compatible with ossia::triple_buffer but self-
// contained, so this compiles in every mode -- standalone (no libossia) and
// against the score SDK, whose bundled ossia/detail/triple_buffer.hpp may be an
// older revision that misses <utility>. It lives in its own namespace, as
// score's headers may include the ossia one too.
#include "compat/triple_buffer.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace lo
{

// A finished frame: the diffused keyframe, plus the optional 2^exp RIFE sub-frame sweep
// prev -> cur (display order, last == cur).
struct AsyncFrame
{
  std::vector<unsigned char> rgba;
  std::vector<unsigned char> sweep;
  int sweep_n{0};
  int w{0}, h{0};
  uint64_t gen{0};  // configuration generation; the consumer drops frames from a stale one
};

// Everything a frame needs, captured on the render thread.
struct AsyncJob
{
  std::vector<unsigned char> ref_rgba;      // input frame at model resolution; empty = txt2img
  std::vector<unsigned char> control_rgba;  // ControlNet map (model res) or IP-Adapter style image
  int control_w{0}, control_h{0};
  std::vector<float> ehs;                   // img2img-turbo external text embedding; empty = prompt
  uint64_t ref_hash{0};
  uint64_t control_hash{0};
  // Live pipeline parameters, applied by the producer when they differ from the pushed ones.
  float controlnet_scale{0.f};
  float ipadapter_scale{0.f};
  float guidance{1.f};
  float delta{1.f};
  float lora_scale{1.f};
  int seed{0};
  int w{0}, h{0};
  int exp{0};       // RIFE interpolation exponent
  uint64_t gen{0};
  uint64_t key{0};  // hash of everything above that changes the output; drives submission
};

// -------------------------------------------------------------------------------------------------
// The transport: a worker thread that turns Jobs into Frames via a blocking produce callback. In
// continuous mode the same job is handed to the callback again and again.
// -------------------------------------------------------------------------------------------------
template <typename Job, typename Frame>
class AsyncFrameProducer
{
public:
  // produce(job, out) -> true if `out` should be published. Runs on the worker thread.
  using produce_fn = std::function<bool(Job&, Frame&)>;

  explicit AsyncFrameProducer(produce_fn fn)
      : m_produce{std::move(fn)}
      , m_frame_tb{Frame{}}
      , m_job_tb{Job{}}
  {
  }

  ~AsyncFrameProducer() { stop(); }

  AsyncFrameProducer(const AsyncFrameProducer&) = delete;
  AsyncFrameProducer& operator=(const AsyncFrameProducer&) = delete;

  // continuous: when no newer job has arrived, re-run the last one (pipelines whose output evolves
  // from call to call: stream-batched or temporal). Otherwise the worker idles until the next submit.
  void start(bool continuous)
  {
    if(m_thread.joinable())
      return;
    m_continuous = continuous;
    m_thread = std::jthread([this](std::stop_token st) { loop(st); });
  }

  // Drain + join. Safe to call when not running. The job buffer is emptied so a restart begins clean.
  void stop()
  {
    if(!m_thread.joinable())
      return;
    m_thread.request_stop();
    {
      std::lock_guard<std::mutex> lk(m_wake_mtx);
      m_job_ready.store(true, std::memory_order_release);
    }
    m_job_cv.notify_all();
    m_thread.join();
    {
      Job drop;
      while(m_job_tb.consume(drop)) { }
    }
    m_job_ready.store(false, std::memory_order_release);
    m_busy.store(false, std::memory_order_release);
  }

  bool running() const { return m_thread.joinable(); }
  bool continuous() const { return m_continuous; }
  bool busy() const { return m_busy.load(std::memory_order_acquire); }

  // render -> producer (newest-wins). Wakes the worker.
  void submit(Job job)
  {
    m_job_tb.produce(std::move(job));
    // Published under the mutex the worker waits on, or a store landing between the worker's
    // predicate check and its wait is a lost wakeup.
    {
      std::lock_guard<std::mutex> lk(m_wake_mtx);
      m_job_ready.store(true, std::memory_order_release);
    }
    m_job_cv.notify_one();
  }

  // producer -> render (newest-wins). True when a fresh frame was moved into `out`.
  bool consume(Frame& out) { return m_frame_tb.consume(out); }

private:
  void loop(std::stop_token stop)
  {
    Job job;
    bool have_job = false;
    for(;;)
    {
      {
        Job nj;
        if(m_job_tb.consume(nj))
        {
          job = std::move(nj);
          have_job = true;
          m_job_ready.store(false, std::memory_order_release);
        }
      }
      if(!have_job)
      {
        std::unique_lock<std::mutex> lk(m_wake_mtx);
        m_job_cv.wait(lk, [&] {
          return stop.stop_requested() || m_job_ready.load(std::memory_order_acquire);
        });
        if(stop.stop_requested())
          return;
        m_job_ready.store(false, std::memory_order_release);
        continue;
      }
      if(stop.stop_requested())
        return;

      m_busy.store(true, std::memory_order_release);
      Frame out;
      // An exception escaping a thread function is std::terminate: treat a throw as a dropped frame.
      bool ok = false;
      try
      {
        ok = m_produce(job, out);
      }
      catch(const std::exception& e)
      {
        std::fprintf(stderr, "AsyncFrameProducer: produce threw (%s); frame dropped\n", e.what());
      }
      catch(...)
      {
        std::fprintf(stderr, "AsyncFrameProducer: produce threw; frame dropped\n");
      }
      if(ok)
        m_frame_tb.produce(std::move(out));
      m_busy.store(false, std::memory_order_release);

      if(!m_continuous)
        have_job = false;
    }
  }

  produce_fn m_produce;
  bool m_continuous{false};
  std::jthread m_thread;
  librediffusion::compat::triple_buffer<Frame> m_frame_tb;  // producer -> render (frames out)
  librediffusion::compat::triple_buffer<Job> m_job_tb;      // render -> producer (jobs in)
  std::mutex m_wake_mtx;                    // companion for the cv only (guards no data)
  std::condition_variable m_job_cv;         // wake the idle worker when a job is submitted
  std::atomic<bool> m_job_ready{false};     // lost-wakeup-safe predicate
  std::atomic<bool> m_busy{false};          // a frame is being produced right now
};

// -------------------------------------------------------------------------------------------------
// Render-side FIFO of sub-frames. on_keyframe() ingests a produced frame (measuring the production
// rate and bounding the buffered latency); present() drains it at that rate with a fractional credit
// (async), present_next() yields exactly one frame per call (sync).
// -------------------------------------------------------------------------------------------------
class PacedFrameConsumer
{
public:
  // `budget_sweeps` bounds buffered latency (Smooth=3, Fresh=2, LowLatency=1). `tnow` is
  // steady-clock seconds. Frames whose gen != cur_gen are dropped.
  void on_keyframe(const AsyncFrame& fresh, double tnow, int budget_sweeps, uint64_t cur_gen)
  {
    if(fresh.gen != cur_gen)
      return;
    const size_t nbytes = (size_t)fresh.w * fresh.h * 4;
    if(nbytes == 0)
      return;

    const int n = std::max(1, fresh.sweep_n);
    if(m_last_kf_t > 0.0)
    {
      const double gap = tnow - m_last_kf_t;
      if(gap > 1e-3 && gap < 5.0)
      {
        const double inst_rate = (double)n / gap;
        m_prod_rate = (m_prod_rate <= 0.0) ? inst_rate : 0.8 * m_prod_rate + 0.2 * inst_rate;
      }
    }
    m_last_kf_t = tnow;

    if(fresh.sweep_n > 1 && fresh.sweep.size() >= (size_t)n * nbytes)
    {
      for(int i = 0; i < n; ++i)
        m_frames.emplace_back(
            fresh.sweep.begin() + (size_t)i * nbytes, fresh.sweep.begin() + (size_t)(i + 1) * nbytes);
    }
    else
    {
      m_frames.emplace_back(fresh.rgba);
    }

    // Drop the stalest frames when over budget -> bounded latency.
    const size_t max_frames = (size_t)std::max(1, budget_sweeps) * n;
    while(m_frames.size() > max_frames)
      m_frames.pop_front();
  }

  // Advance the credit-based drain by the per-tick wall dt and yield the frame to show. False until
  // the first frame exists. `out_ptr` stays valid until the next present*() call.
  bool present(double dt, const unsigned char*& out_ptr, size_t& out_bytes)
  {
    // Wall-clock deltas can be nonsense (rewound clock, paused transport, debugger stop).
    if(!(dt > 0.0) || !std::isfinite(dt))
      dt = 0.0;
    else if(dt > 5.0)
      dt = 5.0;

    m_drain_credit += (m_prod_rate > 0.0) ? m_prod_rate * dt : 1.0;
    m_drain_credit = std::clamp(m_drain_credit, 0.0, 1e6);

    m_advanced = false;
    int to_pop = (int)m_drain_credit;
    if(to_pop > 0)
    {
      m_drain_credit -= (double)to_pop;
      while(to_pop > 0 && !m_frames.empty())
      {
        pop_front();
        --to_pop;
      }
      // Content-starved: drop the unmet credit rather than skip-burst when frames arrive.
      if(to_pop > 0)
        m_drain_credit = 0.0;
    }
    return emit(out_ptr, out_bytes);
  }

  // Sync pacing: one queued frame per call, holding the last one when the FIFO is empty.
  bool present_next(const unsigned char*& out_ptr, size_t& out_bytes)
  {
    m_advanced = false;
    if(!m_frames.empty())
      pop_front();
    return emit(out_ptr, out_bytes);
  }

  // Whether the last present*() yielded a new frame rather than holding the previous one.
  bool advanced() const { return m_advanced; }
  bool empty() const { return m_frames.empty(); }
  double prod_rate() const { return m_prod_rate; }
  size_t fifo_size() const { return m_frames.size(); }
  bool have_emit() const { return m_have_emit; }

  // Forget everything, including the measured rate.
  void reset()
  {
    m_frames.clear();
    m_last_emit.clear();
    m_have_emit = false;
    m_advanced = false;
    m_prod_rate = 0.0;
    m_drain_credit = 0.0;
    m_last_kf_t = 0.0;
  }

private:
  void pop_front()
  {
    m_last_emit = std::move(m_frames.front());
    m_frames.pop_front();
    m_have_emit = true;
    m_advanced = true;
  }

  bool emit(const unsigned char*& out_ptr, size_t& out_bytes) const
  {
    if(!m_have_emit)
      return false;
    out_ptr = m_last_emit.data();
    out_bytes = m_last_emit.size();
    return true;
  }

  std::deque<std::vector<unsigned char>> m_frames;
  std::vector<unsigned char> m_last_emit;
  bool m_have_emit{false};
  bool m_advanced{false};
  double m_prod_rate{0.0};     // EMA of measured sub-frames per second; 0 = unknown
  double m_drain_credit{0.0};  // fractional sub-frames owed this tick
  double m_last_kf_t{0.0};
};

}  // namespace lo
