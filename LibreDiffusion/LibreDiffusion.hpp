#pragma once
#include "AsyncFrameProducer.hpp"
#include "Handles.hpp"
#include "Image.hpp"

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lo
{
struct CachedEngine;

// Settings of an SD-family pipeline (SD1.5 / SD-turbo / SDXL / V2V), as pushed through the config API.
struct SDConfigState
{
  librediffusion_model_type_t model_type{MODEL_SD_15};
  librediffusion_pipeline_mode_t pipeline_mode{MODE_SINGLE_FRAME};
  int width{512};
  int height{512};
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

  int controlnet_index{-1};  // >= 0 when the workflow drives a ControlNet
  bool ipadapter_enabled{false};

  // Live parameters as last pushed to the pipeline (by the producer thread, from the job).
  float controlnet_scale{0.6f};
  float ipadapter_scale{0.7f};
  float lora_scale{-1.f};  // -1: not pushed yet (the engine may have no runtime-LoRA slots)
  int seed{0};
  bool seeded{false};
};

/**
 * @brief Real-time diffusion (SD1.5 / SD-turbo / SDXL / StreamV2V / FLUX.2-klein / img2img-turbo)
 * through the librediffusion C API, loaded at runtime.
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
    // github.com/GaParmar/img2img-turbo (pix2pix-turbo / CycleGAN-turbo skip-VAE): one-step image
    // translation from the input frame and a CLIP text embedding. Not a generic SD-turbo img2img.
    IMG2IMG_TURBO,
  };

  enum KleinQuality : int8_t
  {
    Quality, // transformer_bf16.plan
    Speed    // transformer_fp8_calib.plan
  };

  // How the render thread plays the producer's frames when Async is on: latency vs continuity.
  enum Pacing : int8_t
  {
    Smooth,    // deep FIFO, every frame in order: continuous motion, more latency
    Fresh,     // one keyframe of buffering
    LowLatency // present the newest frame as soon as possible
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
    // The `{}`: halp::texture_input is an aggregate whose rgba_texture has no default member
    // initialisers, so without it the texture fields are indeterminate until the host writes them.
    halp::texture_input<"In"> image{};
    // ControlNet control map (already preprocessed: canny/depth/pose/...), IP-Adapter style image,
    // or the FLUX.2-klein inpaint mask (white = regenerate).
    halp::texture_input<"Control / Style"> control{};
    // img2img-turbo only: an external CLIP text embedding [1,77,1024] = 78848 floats. When empty the
    // embedding is derived from the Prompt through the bundle's clip.engine.
    halp::val_port<"Embedding", std::vector<float>> ehs;
    // Manual mode: render one frame per impulse.
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
      halp_meta(c_name, "prompt_positive")
    } prompt;
    struct : halp::lineedit<"Prompt -", "anime">
    {
      halp_meta(c_name, "prompt_negative")
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
    struct : halp::toggle<"Manual mode">
    {
      halp_meta(description, "Render a new frame only when Trigger fires; otherwise hold the last one")
    } manual;

    struct : halp::knob_f32<"Delta", halp::range{0.0, 2.0, 1.0}>
    {
    } delta;

    struct : halp::knob_f32<"Feed prev. input", halp::range{0.0, 1.0, 0.0}>
    {
    } feed_prev_in;
    struct : halp::knob_f32<"Feed prev. output", halp::range{0.0, 1.0, 0.0}>
    {
    } feed_prev_out;

    struct : halp::knob_f32<"ControlNet scale", halp::range{0.0, 2.0, 0.6}>
    {
      halp_meta(description, "ControlNet conditioning strength (control-aware unet.engine + controlnet.engine required)")
    } controlnet_scale;

    struct : halp::knob_f32<"IP-Adapter scale", halp::range{0.0, 2.0, 0.7}>
    {
      halp_meta(description, "IP-Adapter style strength (IP-variant unet.engine required)")
    } ipadapter_scale;

    // Live when the engine was exported with --lora PATH:runtime; ignored otherwise.
    struct : halp::knob_f32<"LoRA scale", halp::range{0.0, 2.0, 1.0}>
    {
      halp_meta(description, "Runtime LoRA strength (engine exported with --lora PATH:runtime)")
    } lora_scale;

    struct : halp::enum_t<KleinQuality, "Klein quality">
    {
      halp_meta(description, "FLUX.2-klein: bf16 (Quality) vs fp8 (Speed) transformer")
      enum widget
      {
        combobox
      };
    } klein_quality;

    // RIFE optical-flow interpolation between rendered frames: 2^exp displayed frames per rendered one.
    // Needs rife_ifnet_fp16.plan in the engine folder or its parent.
    struct : halp::spinbox_i32<"Interpolation exp", halp::range{0, 3, 0}>
    {
      halp_meta(description, "RIFE optical-flow interpolation: 0=off, 1=2x, 2=4x, 3=8x")
    } rife_exp;

    struct : halp::toggle<"Async">
    {
      halp_meta(description, "Diffuse on a worker thread; the render thread presents steady-clock-paced "
                             "frames (and RIFE sweeps) instead of stalling on the GPU. Off = render on the "
                             "render thread.")
    } async_mode;

    struct : halp::enum_t<Pacing, "Async pacing">
    {
      halp_meta(description, "Smooth (every frame in order, +latency); Fresh (one keyframe of buffering); "
                             "LowLatency (newest frame as soon as possible)")
      enum widget { combobox };
    } pacing;

    struct : halp::spinbox_i32<"GPU", halp::range{0, 15, 0}>
    {
      halp_meta(description, "CUDA device ordinal (CUDA's fastest-first order, not necessarily nvidia-smi's). "
                             "Changing it reloads the engines on that device; engines already loaded "
                             "elsewhere stay resident until the host restarts.")
    } gpu;

    // ---- Engine builder: runs the embedded exporter (uv + train-lora.py) out of process. ----
    struct : halp::lineedit<"Python cache", "">
    {
      halp_meta(description, "Root for the Python toolchain (uv cache, interpreter, venv, exporter). "
                             "Keep it SHORT on Windows (default C:\\lrd): long venv paths break some wheels.")
      enum widget
      {
        folder
      };
    } python_cache;
    struct : halp::lineedit<"Build folder", "">
    {
      halp_meta(description, "Where the built engines go (train-lora.py --output). Empty = the Engines folder.")
      enum widget
      {
        folder
      };
    } build_folder;
    struct : halp::lineedit<"Build options", "--type sd15 --model stabilityai/sd-turbo --min-resolution 512 --max-resolution 512">
    {
      halp_meta(description, "train-lora.py arguments (everything but --output)")
    } build_options;
    halp::val_port<"Build", std::optional<halp::impulse>> build;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;
    halp::val_port<"Build status", std::string> build_status;
  } outputs;

  StreamDiffusion() noexcept;
  ~StreamDiffusion();

  void operator()();

  static bool is_available() noexcept;

private:
  enum class Family : int8_t
  {
    SD,     // SD1.5 / SD-turbo / SDXL / StreamV2V through the pipeline + CLIP + EngineCache
    Klein,  // FLUX.2-klein streaming pipeline
    Turbo   // img2img-turbo skip-VAE pipeline
  };
  static Family familyOf(Workflow) noexcept;
  static bool isImg2Img(Workflow) noexcept;
  static bool isControlNet(Workflow) noexcept;
  static bool isIPAdapter(Workflow) noexcept;

  // Replace non-finite knob values with their defaults and clamp the ones whose arithmetic is not
  // defined out of range.
  void sanitizeControls();

  // The inputs that decide whether setup can succeed at all. A configuration that failed is not
  // retried until one of them changes; continuous knobs are deliberately absent.
  struct SetupKey
  {
    std::string model;
    std::string prompt;
    std::string negative_prompt;
    std::string timesteps;
    int width{0};
    int height{0};
    int gpu{0};
    int8_t workflow{-1};
    int8_t cfg{-1};
    int8_t klein_quality{-1};
    bool add_noise{false};
    bool denoise_batch{false};
    bool valid{false};

    friend bool operator==(const SetupKey&, const SetupKey&) noexcept = default;
  };
  static SetupKey setupKey(const inputs_t& in);
  bool setupBlocked(const inputs_t& in) const;
  void noteSetupFailure(const inputs_t& in) { m_failed_setup = setupKey(in); }
  void noteSetupSuccess() noexcept { m_failed_setup.valid = false; }
  SetupKey m_failed_setup{};

  // Validate + clamp the Resolution port (positive, <= k_max_resolution, multiple of `step`).
  // False when the request cannot be honoured at all.
  bool resolveResolution(const inputs_t& in, int step, int& w, int& h);
  int m_reported_size_w{0};
  int m_reported_size_h{0};

  // ---- Configuration: render thread. Every mutation of a pipeline stops the producer first. ----
  bool configure(const inputs_t& in);
  bool configureSD(const inputs_t& in);
  bool createSDPipeline(const inputs_t& in, std::vector<int> timestep_indices);
  bool updatePromptEmbedding(const std::string& prompt, SDXLEmbeddings& embeddings);
  bool updatePromptEmbeddings(const std::string& prompt, std::vector<SDXLEmbeddings>& embeddings);
  bool updateScheduler(const std::string& timestep_str);
  bool configureKlein(const inputs_t& in);
  bool createKleinStream(const inputs_t& in);
  bool configureTurbo(const inputs_t& in);
  // Drop the current family's pipelines, the RIFE handle and every frame in flight.
  void releaseFamily();

  // ---- Rendering: the producer thread when Async is on, the render thread otherwise. ----
  bool produceFrame(AsyncJob& job, AsyncFrame& out);
  bool renderFrame(const AsyncJob& job, unsigned char* out_rgba);
  bool renderSD(const AsyncJob& job, unsigned char* out_rgba);
  bool renderKlein(const AsyncJob& job, unsigned char* out_rgba);
  bool renderTurbo(const AsyncJob& job, unsigned char* out_rgba);
  void interpolate(const AsyncJob& job, AsyncFrame& out);

  // ---- Per tick ----
  void builderTick(const inputs_t& in, bool build_requested);
  bool buildJob(const inputs_t& in, AsyncJob& job);
  void blendFeedback(rgba_image& cur, const inputs_t& in);
  void renderTick(const inputs_t& in, bool triggered);
  void ensureProducer(bool continuous);
  void stopProducer();

  const sd::liblibrediffusion& m_sd;
  inputs_t m_prev_inputs{};
  Family m_family{Family::SD};
  int m_device{0};
  std::string m_model_dir;   // the Engines folder the current pipelines were loaded from
  int m_w{0}, m_h{0};        // model output size
  bool m_continuous{false};  // the pipeline's output evolves between identical calls
  uint64_t m_gen{0};         // bumped on every configuration change

  // SD family
  CachedEngine* m_cached_engine{nullptr};
  SDConfigState m_config_state;
  std::vector<SDXLEmbeddings> m_embeddings;
  SDXLEmbeddings m_negative_embeddings;

  // FLUX.2-klein family
  SDFluxStream m_klein_stream;
  int m_klein_quality{-1};
  unsigned long long m_klein_seed{0};
  std::string m_klein_prompt;
  std::string m_klein_sched;
  uint64_t m_klein_mask_hash{0};

  // img2img-turbo family
  SDImg2ImgTurbo m_i2it;
  SDClip m_i2it_clip;
  SDXLEmbeddings m_i2it_embeddings;
  std::string m_i2it_prompt;

  // Producer-thread state (the render thread touches it only while the producer is stopped).
  SDRife m_rife;
  bool m_rife_tried{false};
  std::vector<unsigned char> m_prev_key;  // previous keyframe, interpolated from
  uint64_t m_applied_ref_hash{0};         // klein: reference currently VAE-encoded
  uint64_t m_applied_control_hash{0};     // SD: control map / style image currently uploaded
  bool m_reported_no_style{false};

  // Render-thread state
  std::unique_ptr<AsyncFrameProducer<AsyncJob, AsyncFrame>> m_producer;
  PacedFrameConsumer m_consumer;
  uint64_t m_submitted_key{0};
  bool m_reported_no_control{false};
  double m_last_tick_t{0.0};
  rgba_image m_cur_input;
  rgba_image m_prev_input;
  rgba_image m_prev_output;
  std::string m_build_status;
};

}
