#pragma once
#include "librediffusion_loader.hpp"

#include <ossia/detail/lockfree_queue.hpp>
#include <ossia/detail/triple_buffer.hpp>
#include <ossia/detail/variant.hpp>

#include <QImage>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <deque>
#include <functional>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>
namespace lo
{
struct CachedEngine;
}

namespace lo
{

class SDConfig
{
public:
  SDConfig();
  ~SDConfig();
  SDConfig(const SDConfig&) = delete;
  SDConfig& operator=(const SDConfig&) = delete;
  SDConfig(SDConfig&& other) noexcept;
  SDConfig& operator=(SDConfig&& other) noexcept;

  explicit operator bool() const noexcept { return m_handle != nullptr; }
  librediffusion_config_handle get() const noexcept { return m_handle; }
  librediffusion_config_handle release() noexcept;

private:
  librediffusion_config_handle m_handle{nullptr};
};

class SDPipeline
{
public:
  SDPipeline() = default;
  explicit SDPipeline(librediffusion_config_handle config);
  ~SDPipeline();
  SDPipeline(const SDPipeline&) = delete;
  SDPipeline& operator=(const SDPipeline&) = delete;
  SDPipeline(SDPipeline&& other) noexcept;
  SDPipeline& operator=(SDPipeline&& other) noexcept;

  explicit operator bool() const noexcept { return m_handle != nullptr; }
  librediffusion_pipeline_handle get() const noexcept { return m_handle; }
  void reset();

private:
  librediffusion_pipeline_handle m_handle{nullptr};
};

class SDClip
{
public:
  SDClip() = default;
  explicit SDClip(const char* engine_path);
  ~SDClip();
  SDClip(const SDClip&) = delete;
  SDClip& operator=(const SDClip&) = delete;
  SDClip(SDClip&& other) noexcept;
  SDClip& operator=(SDClip&& other) noexcept;

  explicit operator bool() const noexcept { return m_handle != nullptr; }
  librediffusion_clip_handle get() const noexcept { return m_handle; }

private:
  librediffusion_clip_handle m_handle{nullptr};
};

/**
 * @brief RAII wrapper around a FLUX.2-klein streaming pipeline handle.
 */
class SDFluxStream
{
public:
  SDFluxStream() = default;
  SDFluxStream(
      const char* transformer, const char* qwen, const char* vae_decoder,
      const char* vae_encoder, const char* tokenizer_json, int Th, int Tw,
      unsigned long long seed);
  ~SDFluxStream();
  SDFluxStream(const SDFluxStream&) = delete;
  SDFluxStream& operator=(const SDFluxStream&) = delete;
  SDFluxStream(SDFluxStream&& other) noexcept;
  SDFluxStream& operator=(SDFluxStream&& other) noexcept;

  explicit operator bool() const noexcept { return m_handle != nullptr; }
  librediffusion_flux2_stream_handle get() const noexcept { return m_handle; }
  void reset();

private:
  librediffusion_flux2_stream_handle m_handle{nullptr};
};

/**
 * @brief RAII wrapper around a RIFE frame interpolator handle.
 */
class SDRife
{
public:
  SDRife() = default;
  explicit SDRife(const char* engine_path);
  ~SDRife();
  SDRife(const SDRife&) = delete;
  SDRife& operator=(const SDRife&) = delete;
  SDRife(SDRife&& other) noexcept;
  SDRife& operator=(SDRife&& other) noexcept;

  explicit operator bool() const noexcept { return m_handle != nullptr; }
  librediffusion_rife_handle get() const noexcept { return m_handle; }
  void reset();

private:
  librediffusion_rife_handle m_handle{nullptr};
};

struct SDXLEmbeddings
{
  librediffusion_half_t* embeddings{nullptr};
  librediffusion_half_t* pooled_embeds{nullptr};
  librediffusion_half_t* time_ids{nullptr};

  SDXLEmbeddings() = default;
  ~SDXLEmbeddings();
  SDXLEmbeddings(const SDXLEmbeddings&) = delete;
  SDXLEmbeddings& operator=(const SDXLEmbeddings&) = delete;
  SDXLEmbeddings(SDXLEmbeddings&& other) noexcept;
  SDXLEmbeddings& operator=(SDXLEmbeddings&& other) noexcept;

  void reset();
  explicit operator bool() const noexcept { return embeddings != nullptr; }
};

struct SDConfigState
{
  librediffusion_model_type_t model_type{MODEL_SD_15};
  librediffusion_pipeline_mode_t pipeline_mode{MODE_SINGLE_FRAME};
  int width{512};
  int height{512};
  int latent_width{64};
  int latent_height{64};
  int batch_size{1};
  int denoising_steps{1};
  float guidance_scale{1.2f};
  float delta{1.0f};
  bool do_add_noise{true};
  bool use_denoising_batch{false};
  int cfg_type{2};
  int text_seq_len{77};
  int text_hidden_dim{768};
  int clip_pad_token{49407};
  int pooled_embedding_dim{1280};
  std::vector<int> timestep_indices;
  std::string unet_engine_path;
  std::string vae_encoder_path;
  std::string vae_decoder_path;

  // ControlNet: index returned by config_add_controlnet (>=0 when enabled, else -1).
  int controlnet_index{-1};
  float controlnet_scale{0.6f};
  // IP-Adapter: enabled when the workflow is *_IPADAPTER (engine auto-detected as IP-variant).
  bool ipadapter_enabled{false};
  int ipadapter_num_tokens{4};
  float ipadapter_scale{0.7f};
  // IP-Adapter on-device image encoder: fingerprint of the last style image fed via
  // set_ipadapter_image, so we only re-encode when the "Control / Style" texture changes.
  uint64_t ipadapter_style_hash{0};
  bool ipadapter_image_set{false};
};

/**
 * @brief StreamDiffusion processor using dynamic C API
 */
struct StreamDiffusion
{
public:
  halp_meta(name, "StreamDiffusion");
  halp_meta(c_name, "streamdiffusion");
  halp_meta(category, "AI/Generative");
  halp_meta(author, "StreamDiffusion authors, Jean-Michaël Celerier");
  halp_meta(description, "Funky little images.");
  halp_meta(uuid, "a202d577-f92e-4d47-b863-62be5c02084e");
  halp_meta(manual_url, "https://ossia.io/score-docs/processes/streamdiffusion.html");

  enum Workflow : int8_t
  {
    SD_TXT2IMG,
    SD_IMG2IMG,
    SD_TXT2IMG_CONTROLNET,
    SD_IMG2IMG_CONTROLNET,
    SD_TXT2IMG_IPADAPTER,
    SD_IMG2IMG_IPADAPTER,
    SDTURBO_TXT2IMG,
    SDTURBO_IMG2IMG,
    SDXL_TXT2IMG,
    SDXL_IMG2IMG,
    SDXL_TXT2IMG_CONTROLNET,
    SDXL_IMG2IMG_CONTROLNET,
    V2V_TXT2IMG,
    V2V_IMG2IMG,
    FLUX2_KLEIN_TXT2IMG,
    FLUX2_KLEIN_IMG2IMG,
  };

  enum KleinQuality : int8_t
  {
    Quality, // transformer_bf16.plan  (higher fidelity)
    Speed    // transformer_fp8_calib.plan (faster)
  };

  // Async present model: how the render thread plays the producer's RIFE sweeps. Trades latency vs
  // motion continuity. (Only used when Async is on.)
  enum KleinPacing : int8_t
  {
    Smooth,  // sequential FIFO, no skips: play every sweep in order -> continuous motion, +latency
    Fresh,   // newest-wins, 1-keyframe latency, blend the boundary -> low latency, slight morph at seams
    LowLatency // newest-wins, no buffer: present the latest sweep ASAP, hold last frame if late
  };

  enum Cfg : int8_t
  {
    None,
    Self,
    Full,
    Initialize
  };


  struct inputs_t
  {
    halp::texture_input<"In"> image;
    // ControlNet control map (canny/depth/pose/...) OR IP-Adapter style image.
    // Preprocessing is EXTERNAL: feed an already-preprocessed control map here for
    // ControlNet. Only used by the *_CONTROLNET / *_IPADAPTER workflows.
    halp::texture_input<"Control / Style"> control;
    halp::val_port<"Trigger", std::optional<halp::impulse>> trigger;
    struct : halp::enum_t<Workflow, "Workflow">
    {
      enum widget
      {
        combobox
      };
    } workflow;

    struct : halp::lineedit<"Prompt +", "mushroom kingdom, charcoal, velvia">
    {
    } prompt;
    struct : halp::lineedit<"Prompt -", "anime">
    {
    } negative_prompt;
    struct : halp::lineedit<"Engines", "">
    {
      enum widget
      {
        folder
      };
    } model;
    struct : halp::spinbox_i32<"Seed", halp::free_range_max<>>
    {
    } seed;
    struct : halp::knob_f32<"Guidance", halp::range{0.5, 10.0, 1.0}>
    {
    } guidance;
    struct : halp::lineedit<"Timesteps", "15, 25">
    {
    } t1;
    struct : halp::xy_spinboxes_t<int, "Resolution", halp::range{64, 2048, 512}>
    {
    } size;
    struct : halp::enum_t<Cfg, "Guidance type">
    {
      halp_meta(description, "How negative prompts are computed")
      enum widget
      {
        combobox
      };
    } cfg;

    struct : halp::toggle<"Add noise", halp::toggle_setup{.init = true}>
    {
    } add_noise;
    struct : halp::toggle<"Denoising batch">
    {
    } denoise_batch;
    halp::toggle<"Manual mode"> manual;

    struct : halp::knob_f32<"Delta", halp::range{0.0, 2.0, 1.0}>
    {
    } delta;

    struct : halp::knob_f32<"Feed prev. input", halp::range{0.0, 1.0, 0.0}>
    {
    } feed_prev_in;
    struct : halp::knob_f32<"Feed prev. output", halp::range{0.0, 1.0, 0.0}>
    {
    } feed_prev_out;

    // ControlNet conditioning scale (only used by the *_CONTROLNET workflow).
    struct : halp::knob_f32<"ControlNet scale", halp::range{0.0, 2.0, 0.6}>
    {
      halp_meta(description, "ControlNet conditioning strength (control-aware unet.engine + controlnet.engine required)")
    } controlnet_scale;

    // IP-Adapter scale (only used by the *_IPADAPTER workflows).
    struct : halp::knob_f32<"IP-Adapter scale", halp::range{0.0, 2.0, 0.7}>
    {
      halp_meta(description, "IP-Adapter style strength (IP-variant unet.engine required)")
    } ipadapter_scale;

    // FLUX.2-klein only: which transformer engine to load (fidelity vs speed)
    struct : halp::enum_t<KleinQuality, "Klein quality">
    {
      halp_meta(description, "FLUX.2-klein: bf16 (Quality) vs fp8 (Speed) transformer")
      enum widget
      {
        combobox
      };
    } klein_quality;

    // RIFE output frame interpolation factor: 2^exp displayed frames per real frame.
    // 0 = off (opt-in). Only used on the FLUX.2-klein paths.
    struct : halp::spinbox_i32<"Interpolation exp", halp::range{0, 3, 0}>
    {
      halp_meta(description, "RIFE optical-flow interpolation: 0=off, 1=2x, 2=4x, 3=8x (klein output)")
    } rife_exp;

    // FLUX.2-klein only: run diffusion on a worker thread (async) so the render thread
    // emits a smooth interpolated frame every tick instead of stalling ~200ms on diffusion.
    struct : halp::toggle<"Async">
    {
      halp_meta(description, "FLUX.2-klein: diffuse on a worker thread; render thread interpolates "
                             "continuously between the two latest real frames (fluid). Off = "
                             "synchronous (diffuse on the render thread).")
    } klein_async;

    // FLUX.2-klein async only: how the render thread plays the producer's sweeps (latency vs continuity).
    struct : halp::enum_t<KleinPacing, "Async pacing">
    {
      halp_meta(description, "Async present model: Smooth (sequential FIFO, continuous motion, +latency); "
                             "Fresh (newest-wins, 1-keyframe latency, blended seams); "
                             "LowLatency (newest ASAP, may micro-hold).")
      enum widget { combobox };
    } klein_pacing;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;
  } outputs;

  // ---- Phase C async klein producer/consumer types --------------------------------------------
  // A finished SWEEP published by the producer thread into the triple_buffer: the diffused keyframe
  // (cur) PLUS the 2^exp RIFE sub-frames interpolating prev->cur (display order, last == cur).
  // RIFE now runs ON THE PRODUCER THREAD (next to diffusion), so the render thread does NO GPU compute
  // (no RIFE, no diffusion) — it only picks one sub-frame by phase and uploads it. This removes the
  // render-vs-diffusion GPU contention that was starving production.
  struct KleinRealFrame
  {
    std::vector<unsigned char> rgba;  // the keyframe (cur). Also == sweep tail when sweep present.
    std::vector<unsigned char> sweep; // 2^exp sub-frames prev->cur concatenated (empty if no prev/no interp)
    int sweep_n{0};                   // number of sub-frames in `sweep` (0 -> present `rgba`)
    int w{0}, h{0};
    uint64_t gen{0};   // generation id; render drops frames from a stale config
  };

  // The diffusion job handed render->producer (latest-wins). Carries everything the producer loop
  // needs. The producer thread is the ONLY caller of the flux2 C-API (TRT contexts are single-thread).
  struct KleinJob
  {
    std::vector<unsigned char> ref_rgba;  // reference frame (img2img) or black (txt2img)
    bool ref_changed{false};              // whether to re-run set_reference (VAE encode)
    int w{0}, h{0};
    int exp{0};                           // RIFE interpolation exp (producer renders the sweep)
    uint64_t gen{0};
    bool valid{false};
  };

  StreamDiffusion() noexcept;
  ~StreamDiffusion();

  void operator()();

  static bool is_available() noexcept;

private:
  void blendTextures();
  const sd::liblibrediffusion& m_sd;

  CachedEngine* m_cached_engine{nullptr};
  SDConfigState m_config_state;
  std::vector<SDXLEmbeddings> m_embeddings;
  SDXLEmbeddings m_negative_embeddings;

  inputs_t m_prev_inputs{};

  bool createConfiguration(const inputs_t& in_config, const std::vector<int>& timestep_indices);
  bool updatePromptEmbedding(const std::string& prompt, SDXLEmbeddings& embeddings);
  bool updatePromptEmbeddings(const std::string& prompt, std::vector<SDXLEmbeddings>& embeddings);
  bool updateScheduler(const std::string& timestep_str);

  // FLUX.2-klein streaming path (self-contained, bypasses the SD pipeline machinery)
  void runKlein(const inputs_t& in_config);
  bool createKleinStream(const inputs_t& in_config);
  void emitWithRife(
      const unsigned char* frame_rgba, int w, int h);

  SDFluxStream m_klein_stream;
  SDRife m_rife;
  std::string m_klein_model_path;
  int m_klein_quality{-1};
  int m_klein_w{0};
  int m_klein_h{0};
  std::string m_klein_prompt;
  std::vector<unsigned char> m_klein_prev_out; // last real klein frame, for RIFE
  std::vector<unsigned char> m_rife_scratch;   // (2^exp) frames buffer
  bool m_klein_have_prev{false};
  // Task 1: cached reference — re-encode (VAE) only when the input frame changes.
  uint64_t m_klein_ref_hash{0};
  bool m_klein_ref_set{false};
  // Task 2: RIFE display decoupling — emit one queued frame per tick; diffuse only when
  // the queue is empty (every ~2^exp ticks), so diffusion runs only when necessary.
  std::deque<std::vector<unsigned char>> m_klein_queue;
  int m_klein_last_exp{-1};

  // ---- Phase C: steady-clock paced async (dedicated producer thread + triple_buffer) ----------
  // The render thread (operator()) emits ONE frame per tick on a STEADY wall-clock phase, fully
  // decoupled from when diffusion keyframes arrive. A dedicated node-owned producer thread runs the
  // (~150ms) diffusion on the klein stream's own low-prio CUDA stream and hands finished keyframes
  // back through a lock-free triple_buffer (no Qt-main-thread hop). The render thread keeps the two
  // latest keyframes, precomputes the RIFE sweep ONCE per adopted keyframe, and per-tick just memcpy's
  // the phase-indexed cached sub-frame -> constant tiny per-tick cost -> tight pacing.
  void runKleinAsync(const inputs_t& in_config);
  void startKleinProducer();
  void stopKleinProducer();
  void kleinProducerLoop(std::stop_token stop);

  std::jthread m_klein_producer;
  // Both directions are lock-free triple_buffers (newest-wins). The cv/mutex below are ONLY a wakeup
  // for the sleeping producer (a lock-free queue can't block a thread; you'd otherwise busy-spin a
  // core) — they guard NO data. The render thread never holds the mutex on its hot path.
  ossia::triple_buffer<KleinRealFrame> m_klein_real_tb;  // producer -> render (keyframes out)
  ossia::triple_buffer<KleinJob> m_klein_job_tb;         // render -> producer (job in)
  std::mutex m_klein_wake_mtx;            // companion for the cv only (no data)
  std::condition_variable m_klein_job_cv; // wake the idle producer when a job is produced
  std::atomic<bool> m_klein_job_ready{false};      // set by render after produce; lost-wakeup-safe predicate
  std::atomic<bool> m_klein_producer_busy{false};  // a keyframe is being diffused right now

  // ---- PRODUCER-thread-only state (the producer owns RIFE so the render thread does NO GPU compute) -
  SDRife m_klein_producer_rife;            // RIFE handle used ONLY by the producer thread
  std::vector<unsigned char> m_klein_prev_key;  // previous keyframe (producer-side, to interpolate from)
  bool m_klein_have_prev_key{false};
  int m_klein_exp{0};                      // render-side: last exp submitted to the producer (change detect)
  std::string m_klein_rife_path;           // rife engine path (for lazy producer-side create)

  // ---- RENDER-thread-only state: SUB-FRAME FIFO drained at the MEASURED PRODUCTION RATE -------------
  // Async is slightly less smooth than sync because content arrives at the diffusion rate (e.g. 7fps x
  // 2^exp sub-frames) which rarely equals the display rate. Popping exactly 1/tick then HOLDING on empty
  // bunches the repeats into visible stutters. Instead we drain via a fractional CREDIT accumulator:
  // each tick credit += measured_prod_rate * dt; we pop floor(credit) frames (keep the fraction). When
  // content < display -> some ticks pop 0 (repeat), spread EVENLY; when content > display (bigger GPU) ->
  // some ticks pop 2+ (skip), spread EVENLY. Self-calibrating from the measured rate — no hardcoded fps.
  std::deque<std::vector<unsigned char>> m_klein_frames;  // pending sub-frames, display order
  std::vector<unsigned char> m_klein_last_emit;           // last frame shown (held when the FIFO drains)
  bool m_klein_have_emit{false};
  double m_klein_prod_rate{0.0};      // EMA of MEASURED sub-frames produced per second; 0 = not yet measured
  double m_klein_drain_credit{0.0};   // fractional sub-frames owed this tick (carries the remainder)
  double m_klein_kf_interval{0.0};    // EMA of the MEASURED keyframe interval [s]; 0 = not yet measured
  double m_klein_last_kf_t{0.0};      // wall time the last sweep was adopted (for interval measurement)
  double m_klein_last_tick_t{0.0};    // wall time of previous tick (for render dt)
  uint64_t m_klein_gen{0};            // bumped on config change -> invalidates in-flight jobs
  double m_klein_dbg_last_pub_t{0.0};   // FPS dump: wall time of previous producer publish
  double m_klein_dbg_last_emit_t{0.0};  // FPS dump: wall time of previous receiver present
  uint64_t m_klein_async_ref_hash{0};
  bool m_klein_async_ref_set{false};

  struct Image
  {
    QByteArray storage;
    QImage image;
  };

  Image m_prev_input;
  QImage m_ext_input;
  QImage m_cur_input;
  Image m_prev_output;
};

}
