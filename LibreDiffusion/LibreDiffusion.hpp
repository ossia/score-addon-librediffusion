#pragma once
#include "AsyncFrameProducer.hpp"
#include "librediffusion_loader.hpp"

// triple_buffer pulled in transitively via AsyncFrameProducer.hpp above, which
// already selects the ossia header or the vendored standalone copy.

#include "Image.hpp"

#include <halp/buffer.hpp>
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

/**
 * @brief RAII wrapper around an img2img-turbo (GaParmar/img2img-turbo) skip-VAE pipeline handle.
 */
class SDImg2ImgTurbo
{
public:
  SDImg2ImgTurbo() = default;
  SDImg2ImgTurbo(const char* unet, const char* vae_encoder, const char* vae_decoder);
  ~SDImg2ImgTurbo();
  SDImg2ImgTurbo(const SDImg2ImgTurbo&) = delete;
  SDImg2ImgTurbo& operator=(const SDImg2ImgTurbo&) = delete;
  SDImg2ImgTurbo(SDImg2ImgTurbo&& other) noexcept;
  SDImg2ImgTurbo& operator=(SDImg2ImgTurbo&& other) noexcept;

  explicit operator bool() const noexcept { return m_handle != nullptr; }
  librediffusion_img2img_turbo_handle get() const noexcept { return m_handle; }
  void reset();

private:
  librediffusion_img2img_turbo_handle m_handle{nullptr};
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
  // Runtime LoRA: the loaded UNet engine declares lora_scale[N] (engines exported with PATH:runtime).
  // Last applied uniform scale (so we only push on change).
  float lora_scale{1.0f};
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
    FLUX2_KLEIN_INPAINT,
    // github.com/GaParmar/img2img-turbo (pix2pix-turbo / CycleGAN-turbo skip-VAE). One-step image
    // translation: input frame + a CLIP text embedding (via the "Embedding" port) -> output.
    // NOT a generic SD-turbo img2img workflow. Self-contained C-API (librediffusion_img2img_turbo_*).
    IMG2IMG_TURBO,
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
    // img2img-turbo (IMG2IMG_TURBO) only: the CLIP text embedding [1,77,1024] = 78848 floats, fed from
    // upstream (a prompt-encoder object, or a baked constant for the CycleGAN day2night/etc. models).
    // Keeps CLIP-text out of this node. Unused by every other workflow. (A plain float list, not a
    // texture/cpu_buffer: GFX nodes only accept parameter/texture inputs, so it rides on a value port.)
    halp::val_port<"Embedding", std::vector<float>> ehs;
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

    // Runtime LoRA strength: live-adjustable when the engine was exported with --lora PATH:runtime
    // (UNet declares a lora_scale input). 1 = LoRA fully on (== a baked engine), 0 = off. Applied
    // uniformly to all runtime-LoRA slots; ignored by engines without a lora_scale input.
    struct : halp::knob_f32<"LoRA scale", halp::range{0.0, 2.0, 1.0}>
    {
      halp_meta(description, "Runtime LoRA strength (engine exported with --lora PATH:runtime)")
    } lora_scale;

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

    // Run diffusion on a worker thread (async) so the render thread emits a smooth, steady-clock-paced
    // frame every tick instead of stalling on diffusion. FLUX.2-klein and plain SD/SD-turbo/SDXS/SDXL
    // txt2img/img2img (the slow ones — e.g. SDXL @1024 ~10fps — benefit most); ControlNet/IP-Adapter
    // stay synchronous for now (per-tick conditioning).
    struct : halp::toggle<"Async">
    {
      halp_meta(description, "Diffuse on a worker thread; the render thread presents steady-clock-paced "
                             "frames and (with RIFE) interpolates between the latest real frames. Off = "
                             "synchronous (diffuse on the render thread). Applies to klein and plain "
                             "SD/SDXL txt2img/img2img; CN/IP stay synchronous.")
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

  // Async producer/consumer payloads (AsyncFrame keyframe+sweep, AsyncJob ref+exp) are the shared
  // types in AsyncFrameProducer.hpp — used identically by the klein and SD/SDXL async paths.

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

  // img2img-turbo path (self-contained skip-VAE C-API; static 512x512, one-step, host RGBA bytes)
  void runImg2ImgTurbo(const inputs_t& in_config);
  SDImg2ImgTurbo m_i2it;
  std::string m_i2it_model_path;
  std::vector<unsigned char> m_i2it_in;   // input frame, RGBA8 512x512 (persistent scratch)
  std::vector<unsigned char> m_i2it_out;  // output frame, RGBA8 512x512 (persistent scratch)
  SDClip m_i2it_clip;                      // CLIP encoder (prompt -> embedding), like SD/SDXL
  SDXLEmbeddings m_i2it_embeddings;        // device fp16 [1,77,1024] derived from the prompt
  std::string m_i2it_prompt;               // last prompt, to recompute only on change

  SDFluxStream m_klein_stream;
  SDRife m_rife;
  std::string m_klein_model_path;
  int m_klein_quality{-1};
  int m_klein_w{0};
  int m_klein_h{0};
  std::string m_klein_prompt;
  std::string m_klein_sched;                   // last-applied Timesteps string (klein sigma schedule)
  uint64_t m_klein_mask_hash{0};               // hash of the last-applied inpaint mask (0 = none)
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

  // ---- Async (Phase C): steady-clock paced producer/consumer, SHARED between klein and SD/SDXL. -----
  // A node-owned producer thread runs the heavy diffusion (+RIFE) on the model's own CUDA stream and
  // publishes finished keyframe+sweep frames through a lock-free triple_buffer; the render thread
  // (runKleinAsync / runSDAsync -> presentAsync) presents ONE credit-paced sub-frame per tick, fully
  // decoupled. Both paths use the same AsyncFrameProducer + PacedFrameConsumer (see AsyncFrameProducer.hpp).

  // Shared render-side driver: submit a job when the reference/exp changes, consume the freshest
  // keyframe into `consumer`, and present one paced frame to outputs.image. `ref_constant_hash` is true
  // for txt2img (black ref hashes constant -> submitted once).
  void presentAsync(
      AsyncFrameProducer<AsyncJob, AsyncFrame>& producer, PacedFrameConsumer& consumer,
      const unsigned char* ref, bool have_input, int exp, uint64_t gen, int pacing, int w, int h,
      bool ref_constant_hash, uint64_t& ref_hash, bool& ref_set, int& last_exp, double& last_tick_t);

  // --- FLUX.2-klein async (the producer body calls flux2_stream_set_reference + frame_cached + RIFE) ---
  void runKleinAsync(const inputs_t& in_config);
  void ensureKleinProducer();
  void stopKleinProducer();

  std::unique_ptr<AsyncFrameProducer<AsyncJob, AsyncFrame>> m_klein_producer;
  PacedFrameConsumer m_klein_consumer;
  SDRife m_klein_producer_rife;                 // RIFE handle owned by the klein producer thread
  std::vector<unsigned char> m_klein_prev_key;  // previous keyframe (producer-side) to interpolate from
  bool m_klein_have_prev_key{false};
  std::string m_klein_rife_path;                // rife engine path (lazy producer-side create)
  uint64_t m_klein_async_ref_hash{0};
  bool m_klein_async_ref_set{false};
  int m_klein_async_exp{-1};                    // last exp submitted (change detect)
  double m_klein_last_tick_t{0.0};              // wall time of previous async tick (render dt)
  uint64_t m_klein_gen{0};                      // bumped on config change -> invalidates in-flight jobs

  // --- Generic async for SD/SD-turbo/SDXS/SDXL (producer body calls txt2img/img2img + RIFE). Scoped to
  // plain txt2img/img2img (no per-tick CN/IP conditioning). ---
  void runSDAsync(const inputs_t& in_config, unsigned char* input_tex_bytes, int w, int h);
  void ensureSDProducer();
  void stopSDProducer();
  static bool sdAsyncEligible(int8_t workflow) noexcept;

  std::unique_ptr<AsyncFrameProducer<AsyncJob, AsyncFrame>> m_sd_producer;
  PacedFrameConsumer m_sd_consumer;
  SDRife m_sd_producer_rife;                    // RIFE handle owned by the SD producer thread
  std::vector<unsigned char> m_sd_prev_key;     // previous keyframe (producer-side) to interpolate from
  bool m_sd_have_prev_key{false};
  bool m_sd_async_img2img{false};               // stable while the producer runs (workflow change rebuilds)
  std::string m_sd_rife_path;                   // rife engine path (lazy producer-side create)
  uint64_t m_sd_gen{0};                         // bumped on config change -> invalidates in-flight jobs
  uint64_t m_sd_async_ref_hash{0};
  bool m_sd_async_ref_set{false};
  int m_sd_async_exp{-1};                       // last exp submitted (change detect)
  double m_sd_last_tick_t{0.0};                 // wall time of previous async tick (render dt)

  lo::rgba_image m_prev_input;
  lo::rgba_image m_ext_input;
  lo::rgba_image m_cur_input;
  lo::rgba_image m_prev_output;
};

}
