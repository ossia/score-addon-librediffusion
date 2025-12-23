#pragma once
#include "librediffusion_loader.hpp"

#include <ossia/detail/lockfree_queue.hpp>
#include <ossia/detail/variant.hpp>

#include <QImage>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <atomic>
#include <memory>
#include <string>
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
    SD_TXT2IMG_IPADAPTER,
    SD_IMG2IMG_IPADAPTER,
    SDTURBO_TXT2IMG,
    SDTURBO_IMG2IMG,
    SDXL_TXT2IMG,
    SDXL_IMG2IMG,
    V2V_TXT2IMG,
    V2V_IMG2IMG,
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

    struct : halp::toggle<"Add noise">
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
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;
  } outputs;

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
