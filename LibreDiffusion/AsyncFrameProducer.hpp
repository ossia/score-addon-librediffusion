#pragma once

// ---------------------------------------------------------------------------------------------------
// Generic, model-agnostic async frame production for the StreamDiffusion node (option (A): the
// producer thread BLOCKS on its own CUDA stream inside the produce callback; the render thread never
// touches the GPU and only consumes finished frames through a lock-free triple_buffer).
//
// This is the reusable extraction of what the FLUX.2-klein path (kleinProducerLoop / runKleinAsync)
// pioneered, so SD / SD-turbo / SDXS / SDXL — any pipeline whose diffusion is slower than the display
// rate (e.g. SDXL @1024 ~10fps) — can decouple diffusion from presentation with the SAME machinery:
//
//   * AsyncFrameProducer<Job,Frame> : the TRANSPORT — a dedicated worker thread + two newest-wins
//     triple_buffers (job in / frame out) + a cv wakeup. It calls a caller-supplied `produce(Job&,
//     Frame&)` that does the heavy, BLOCKING GPU work on the worker thread. No CUDA event ever crosses
//     a thread boundary: CUDA's job reduces to blocking the one worker thread until its own stream
//     drains (the library's txt2img/img2img/flux2_stream_frame_cached already do this internally), and
//     the triple_buffer is the thread-safe host-side hand-off.
//
//   * PacedFrameConsumer : the render-side STEADY-CLOCK pacing — a sub-frame FIFO drained by a
//     fractional credit accumulator at the MEASURED production rate, so repeats (content<display) and
//     skips (content>display) spread EVENLY instead of bunching into stutters. Self-calibrating from
//     the measured keyframe rate; no hardcoded fps.
//
// IMPORTANT (TRT contexts are single-thread): once a producer is running, the worker thread must be the
// ONLY caller of that pipeline's inference API. Any main-thread config change must stop()/join the
// producer first (drain), exactly as the klein path does around set_prompt / createConfiguration.
// ---------------------------------------------------------------------------------------------------

// triple_buffer is the lock-free producer->consumer hand-off. Always use the
// vendored copy: it is API-compatible with ossia::triple_buffer but self-
// contained, so this compiles in every mode -- standalone (no libossia) and,
// crucially, against the score SDK, whose bundled ossia/detail/triple_buffer.hpp
// is an older revision that misses <utility> and fails to build. No score
// header included by this object pulls the ossia copy, so there is no clash.
#include "compat/triple_buffer.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <deque>
#include <functional>
#include <mutex>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace lo
{

// A finished frame published producer -> render. `rgba` is the diffused keyframe (cur). `sweep` is the
// optional 2^exp RIFE sub-frame sequence prev->cur (display order, last == cur); empty when there is no
// interpolation. Model-agnostic — identical payload for klein and SD/SDXL.
struct AsyncFrame
{
  std::vector<unsigned char> rgba;   // the keyframe (cur); also == sweep tail when a sweep is present
  std::vector<unsigned char> sweep;  // 2^exp sub-frames prev->cur concatenated (empty if no interp)
  int sweep_n{0};                    // number of sub-frames in `sweep` (0 -> present `rgba`)
  int w{0}, h{0};
  uint64_t gen{0};                   // generation id; the consumer drops frames from a stale config
};

// A job handed render -> producer (newest-wins). Carries everything the produce callback needs that
// varies per tick: the reference/input frame (img2img) or black (txt2img), and the interpolation exp.
struct AsyncJob
{
  std::vector<unsigned char> ref_rgba;  // reference frame (img2img) or black (txt2img)
  bool ref_changed{false};              // whether the input genuinely changed (re-run VAE encode etc.)
  int w{0}, h{0};
  int exp{0};                           // RIFE interpolation exp (the producer renders the sweep)
  uint64_t gen{0};
  bool valid{false};
};

// -------------------------------------------------------------------------------------------------
// The transport: a worker thread that turns Jobs into Frames via a blocking produce callback.
// Job and Frame must be movable. The produce callback receives the job by MUTABLE ref so it can clear
// one-shot flags across self-rearm re-runs (e.g. AsyncJob::ref_changed after the first VAE encode).
// -------------------------------------------------------------------------------------------------
template <typename Job, typename Frame>
class AsyncFrameProducer
{
public:
  // produce(job, out) -> true if `out` is valid and should be published. Runs on the worker thread.
  using produce_fn = std::function<bool(Job& /*job*/, Frame& /*out*/)>;

  // rerun_when_idle: when no newer job has arrived, re-run the last job flat-out (keeps the producer
  // saturated so an interpolating consumer always has fresh keyframes — the klein model). Set false
  // for deterministic static content (e.g. txt2img with a fixed seed) to avoid pegging the GPU
  // regenerating an identical frame; the producer then idles until the next submit().
  explicit AsyncFrameProducer(produce_fn fn, bool rerun_when_idle = true)
      : m_produce{std::move(fn)}
      , m_rerun_idle{rerun_when_idle}
      , m_frame_tb{Frame{}}
      , m_job_tb{Job{}}
  {
  }

  ~AsyncFrameProducer() { stop(); }

  AsyncFrameProducer(const AsyncFrameProducer&) = delete;
  AsyncFrameProducer& operator=(const AsyncFrameProducer&) = delete;

  void start()
  {
    if(m_thread.joinable())
      return;
    m_thread = std::jthread([this](std::stop_token st) { loop(st); });
  }

  // Drain + join. Safe to call when not running. After stop() the job buffer is emptied so a restart
  // begins clean.
  void stop()
  {
    if(!m_thread.joinable())
      return;
    m_thread.request_stop();
    // Wake the (possibly sleeping) worker so it observes the stop request.
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
  bool busy() const { return m_busy.load(std::memory_order_acquire); }

  // render -> producer (newest-wins). Wakes the worker.
  void submit(Job job)
  {
    m_job_tb.produce(std::move(job));
    // The predicate MUST be published under m_wake_mtx (as stop() already does): the worker
    // evaluates it inside m_job_cv.wait() while holding that mutex, and a store+notify landing
    // between that evaluation and the worker actually blocking is a lost wakeup -- the job then
    // sits in the triple buffer, unnoticed, until the next submit.
    {
      std::lock_guard<std::mutex> lk(m_wake_mtx);
      m_job_ready.store(true, std::memory_order_release);
    }
    m_job_cv.notify_one();
  }

  // producer -> render (newest-wins). Returns true if a fresh frame was moved into `out`.
  bool consume(Frame& out) { return m_frame_tb.consume(out); }

private:
  void loop(std::stop_token stop)
  {
    Job job;
    bool have_job = false;  // the last job we ran; reused to self-rearm when rerun_when_idle.
    for(;;)
    {
      // Pull the freshest job if one was published (lock-free, newest-wins).
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
        // Nothing to run -> block until a job is signalled (lost-wakeup-safe predicate).
        std::unique_lock<std::mutex> lk(m_wake_mtx);
        m_job_cv.wait(lk, [&] {
          return stop.stop_requested() || m_job_ready.load(std::memory_order_acquire);
        });
        if(stop.stop_requested())
          return;
        m_job_ready.store(false, std::memory_order_release);
        continue;  // loop back to consume the job we were just signalled about
      }
      if(stop.stop_requested())
        return;

      m_busy.store(true, std::memory_order_release);
      Frame out;
      // The produce body allocates w*h*4 (and up to 2^exp times that for a RIFE sweep) and calls
      // into TensorRT, so bad_alloc / length_error / a library exception are all reachable. An
      // exception escaping a thread function is std::terminate -- i.e. the whole host process --
      // with no way for the node to intervene. Treat a throw as a failed frame: publish nothing
      // and keep the worker alive (the node degrades, the render thread holds its last frame).
      bool ok = false;
      try
      {
        ok = m_produce(job, out);
      }
      catch(const std::exception& e)
      {
        std::fprintf(stderr, "AsyncFrameProducer: produce threw (%s); frame dropped\n", e.what());
        ok = false;
      }
      catch(...)
      {
        std::fprintf(stderr, "AsyncFrameProducer: produce threw; frame dropped\n");
        ok = false;
      }
      if(ok)
        m_frame_tb.produce(std::move(out));
      m_busy.store(false, std::memory_order_release);

      // Self-rearm policy: keep `job` and re-run flat-out (saturated), or idle until the next submit.
      if(!m_rerun_idle)
        have_job = false;
    }
  }

  produce_fn m_produce;
  bool m_rerun_idle{true};
  std::jthread m_thread;
  ossia::triple_buffer<Frame> m_frame_tb;  // producer -> render (frames out)
  ossia::triple_buffer<Job> m_job_tb;      // render -> producer (jobs in)
  std::mutex m_wake_mtx;                    // companion for the cv only (guards no data)
  std::condition_variable m_job_cv;         // wake the idle worker when a job is submitted
  std::atomic<bool> m_job_ready{false};     // lost-wakeup-safe predicate
  std::atomic<bool> m_busy{false};          // a frame is being produced right now
};

// -------------------------------------------------------------------------------------------------
// Render-side steady-clock pacing. Holds a sub-frame FIFO and drains it by a fractional credit at the
// MEASURED production rate, so the display stays smooth regardless of the GPU's actual fps. Extracted
// verbatim from runKleinAsync's drain so klein and SD/SDXL present identically.
//
// Per render tick:
//   1. on_keyframe(fresh, tnow, budget_sweeps) for each freshly-consumed producer frame (appends its
//      sub-frames, measures the rate, trims to the latency budget);
//   2. present(dt, ptr, bytes) advances the credit and yields the frame to upload this tick.
// -------------------------------------------------------------------------------------------------
class PacedFrameConsumer
{
public:
  // Ingest a freshly-produced keyframe+sweep. `budget_sweeps` bounds buffered latency (Smooth=3,
  // Fresh=2, LowLatency=1). `tnow` is steady-clock seconds. Drops frames whose gen != cur_gen.
  void on_keyframe(const AsyncFrame& fresh, double tnow, int budget_sweeps, uint64_t cur_gen)
  {
    if(fresh.gen != cur_gen)
      return;
    const int w = fresh.w, h = fresh.h;
    const size_t nbytes = (size_t)w * h * 4;
    if(nbytes == 0)
      return;

    const int n = std::max(1, fresh.sweep_n);
    if(m_last_kf_t > 0.0)
    {
      const double gap = tnow - m_last_kf_t;
      if(gap > 1e-3 && gap < 5.0)  // reject physically-impossible gaps
      {
        m_kf_interval = (m_kf_interval <= 0.0) ? gap : 0.8 * m_kf_interval + 0.2 * gap;
        const double inst_rate = (double)n / gap;  // this sweep's sub-frames over the interval
        m_prod_rate = (m_prod_rate <= 0.0) ? inst_rate : 0.8 * m_prod_rate + 0.2 * inst_rate;
      }
    }
    m_last_kf_t = tnow;

    const size_t max_frames = (size_t)std::max(1, budget_sweeps) * n;

    // Append this sweep's sub-frames in display order (sweep ends at the new keyframe).
    if(fresh.sweep_n > 1 && fresh.sweep.size() >= (size_t)n * nbytes)
    {
      for(int i = 0; i < n; ++i)
        m_frames.emplace_back(
            fresh.sweep.begin() + (size_t)i * nbytes,
            fresh.sweep.begin() + (size_t)(i + 1) * nbytes);
    }
    else
    {
      m_frames.emplace_back(fresh.rgba);  // no interpolation -> the single keyframe
    }

    // Trim from the FRONT (drop the stalest queued frames) when over budget -> bounds latency.
    while(m_frames.size() > max_frames)
      m_frames.pop_front();
  }

  // Advance the credit-based even-spread drain by the per-tick wall dt and yield the frame to show.
  // Returns false until the first frame is available (producer warming up). On true, `out_ptr`/
  // `out_bytes` point at the frame to upload (owned by this consumer; valid until the next call).
  bool present(double dt, const unsigned char*& out_ptr, size_t& out_bytes)
  {
    // Validate dt exactly like on_keyframe validates its gap: wall-clock deltas can be nonsense
    // (a rewound or non-monotonic clock, a paused transport, a debugger stop). A negative dt used
    // to push the accumulator negative, so nothing was emitted until seconds of positive dt had
    // paid the debt back; a NaN dt wedged the pacer PERMANENTLY (NaN + anything is NaN) and
    // (int)NaN is undefined behaviour on top.
    if(!(dt > 0.0) || !std::isfinite(dt))
      dt = 0.0;
    else if(dt > 5.0)
      dt = 5.0;   // same plausibility bound as on_keyframe

    if(m_prod_rate > 0.0)
      m_drain_credit += m_prod_rate * dt;
    else
      m_drain_credit += 1.0;  // before the rate is known, fall back to 1-per-tick

    // Bound the accumulator before the cast: it is a product of two measured quantities, and
    // (int) of an out-of-range double is undefined behaviour.
    m_drain_credit = std::clamp(m_drain_credit, 0.0, 1e6);

    int to_pop = (int)m_drain_credit;
    if(to_pop > 0)
    {
      m_drain_credit -= (double)to_pop;
      while(to_pop > 0 && !m_frames.empty())
      {
        m_last_emit = std::move(m_frames.front());
        m_frames.pop_front();
        m_have_emit = true;
        --to_pop;
      }
      // FIFO drained before satisfying the credit -> content-starved; drop the unmet credit (don't
      // bank it, or we'd skip-burst when frames arrive) and hold the last frame.
      if(to_pop > 0)
        m_drain_credit = 0.0;
    }
    // else: credit < 1 this tick -> emit nothing new, hold the last frame (an evenly-spaced repeat).

    if(!m_have_emit)
      return false;
    out_ptr = m_last_emit.data();
    out_bytes = m_last_emit.size();
    return true;
  }

  // Flush pending sub-frames (e.g. prompt/exp change) but keep the measured rate (cadence unchanged).
  void flush_frames()
  {
    m_frames.clear();
    m_drain_credit = 0.0;
  }

  // Full reset (config change): forget everything including the measured rate.
  void reset()
  {
    m_frames.clear();
    m_last_emit.clear();
    m_have_emit = false;
    m_prod_rate = 0.0;
    m_drain_credit = 0.0;
    m_kf_interval = 0.0;
    m_last_kf_t = 0.0;
  }

  double prod_rate() const { return m_prod_rate; }
  size_t fifo_size() const { return m_frames.size(); }
  bool have_emit() const { return m_have_emit; }

private:
  std::deque<std::vector<unsigned char>> m_frames;  // pending sub-frames, display order
  std::vector<unsigned char> m_last_emit;           // last frame shown (held when the FIFO drains)
  bool m_have_emit{false};
  double m_prod_rate{0.0};     // EMA of MEASURED sub-frames produced per second; 0 = not yet measured
  double m_drain_credit{0.0};  // fractional sub-frames owed this tick (carries the remainder)
  double m_kf_interval{0.0};   // EMA of the MEASURED keyframe interval [s]
  double m_last_kf_t{0.0};     // wall time the last sweep was adopted
};

}  // namespace lo
