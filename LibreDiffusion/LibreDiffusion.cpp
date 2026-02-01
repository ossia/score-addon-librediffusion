/**
 * LibreDiffusion - Ported to StreamDiffusion C API via dynamic loading
 */

#include "LibreDiffusion.hpp"

#include "EngineCache.hpp"
#include "schedulers/lcm_dreamshaper_v7.hpp"
#include "schedulers/sd-turbo.hpp"
#include "schedulers/sdxl-turbo.hpp"
#include <ossia/detail/fmt.hpp>
#include <ctre.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3.hpp>
#include <State/ValueParser.hpp>
#include <QDebug>


#include <algorithm>
#include <ranges>

// Prompt interpolation
namespace lo
{

struct WeightedPromptElement
{
  std::string text;
  double value;
};

/**
 * Small parser for the following prompt language:
 *
 * (some text, 0.1), (other text, 0.5), (blablabla, 1)
 */
std::optional<std::vector<WeightedPromptElement>>
parse_input_string(std::string_view str);

}

BOOST_FUSION_ADAPT_STRUCT(
    lo::WeightedPromptElement,
    (std::string, text)(double, value))

namespace lo
{

namespace x3 = boost::spirit::x3;

struct TextContentTag;
struct NumberTag;
struct WeightedPromptElementTag;
struct DataListTag;

const x3::rule<TextContentTag, std::string> text_content = "text_content";
const x3::rule<NumberTag, double> number = "number";
const x3::rule<WeightedPromptElementTag, WeightedPromptElement> data_item
    = "data_item";
const x3::rule<DataListTag, std::vector<WeightedPromptElement>> data_list
    = "data_list";

auto const text_content_def = x3::lexeme[*(x3::char_ - ':')];
auto const number_def = x3::double_;
auto const data_item_def = '(' >> text_content >> ':' >> number >> ')';
auto const data_list_def = data_item % ',';

BOOST_SPIRIT_DEFINE(text_content, number, data_item, data_list);

std::optional<std::vector<WeightedPromptElement>>
parse_input_string(std::string_view str)
{
  std::vector<WeightedPromptElement> result_data;
  auto iterator = str.begin();
  auto const end_iterator = str.end();

  const auto success = x3::phrase_parse(
      iterator, end_iterator, data_list, x3::ascii::space, result_data);

  if (success && iterator == end_iterator)
    return result_data;

  return std::nullopt;
}

}

namespace lo
{
static constexpr auto all_numbers = ctre::search_all<"\\d+">;

static std::vector<int> get_steps(std::string s)
{
  if(s.empty())
    return {};

  int start = 0;
  for(; start < s.size(); start++) {
    if(s[start] == ' ' || s[start] == '\t' || s[start] == '\n')
      continue;
    break;
  }
  int end = s.size() - 1;
  for(; end > start; end--) {
    if(s[end] == ' ' || s[end] == '\t' || s[end] == '\n')
      continue;
    break;
  }

  s = s.substr(start, end - start + 1);
  if(s.empty())
    return {};
  if(s.contains(',') && !s.starts_with('[')) {
    s.insert(s.begin(), '[');
    s.insert(s.end() - 1, ']');
  }

  if(auto v = State::parseValue(s))
  {
    if(auto value = v->target<int>())
      return {std::clamp(*value, 0, 49)};
    else if(auto value = v->target<float>())
      return {std::clamp((int)*value, 0, 49)};
    else if(auto value = v->target<std::vector<ossia::value>>()) {
      std::vector<int> res;
      for(auto& val : *value) {
        res.push_back(std::clamp(ossia::convert<int>(val), 0, 49));
      }
      return res;
    }
  }

  return std::ranges::to<std::vector<int>>(
      all_numbers(s)
      | std::views::transform([](auto&& match) { return std::clamp(match.to_number(10), 0, 49); }));
}

SDConfig::SDConfig()
{
  const auto& sd = sd::liblibrediffusion::instance();
  if (sd.available)
  {
    sd.config_create(&m_handle);
  }
}

SDConfig::~SDConfig()
{
  if (m_handle)
  {
    const auto& sd = sd::liblibrediffusion::instance();
    if (sd.available)
    {
      sd.config_destroy(m_handle);
    }
  }
}

SDConfig::SDConfig(SDConfig&& other) noexcept
    : m_handle{other.m_handle}
{
  other.m_handle = nullptr;
}

SDConfig& SDConfig::operator=(SDConfig&& other) noexcept
{
  if (this != &other)
  {
    if (m_handle)
    {
      const auto& sd = sd::liblibrediffusion::instance();
      if (sd.available)
        sd.config_destroy(m_handle);
    }
    m_handle = other.m_handle;
    other.m_handle = nullptr;
  }
  return *this;
}

librediffusion_config_handle SDConfig::release() noexcept
{
  auto h = m_handle;
  m_handle = nullptr;
  return h;
}

SDPipeline::SDPipeline(librediffusion_config_handle config)
{
  const auto& sd = sd::liblibrediffusion::instance();
  if (sd.available && config)
  {
    sd.pipeline_create(config, &m_handle);
    if (m_handle)
    {
      sd.pipeline_init_all(m_handle);
    }
  }
}

SDPipeline::~SDPipeline()
{
  reset();
}

SDPipeline::SDPipeline(SDPipeline&& other) noexcept
    : m_handle{other.m_handle}
{
  other.m_handle = nullptr;
}

SDPipeline& SDPipeline::operator=(SDPipeline&& other) noexcept
{
  if (this != &other)
  {
    reset();
    m_handle = other.m_handle;
    other.m_handle = nullptr;
  }
  return *this;
}

void SDPipeline::reset()
{
  if (m_handle)
  {
    const auto& sd = sd::liblibrediffusion::instance();
    if (sd.available)
    {
      sd.pipeline_destroy(m_handle);
    }
    m_handle = nullptr;
  }
}

SDClip::SDClip(const char* engine_path)
{
  const auto& sd = sd::liblibrediffusion::instance();
  if (sd.available && engine_path)
  {
    sd.clip_create(engine_path, &m_handle);
  }
}

SDClip::~SDClip()
{
  if (m_handle)
  {
    const auto& sd = sd::liblibrediffusion::instance();
    if (sd.available)
    {
      sd.clip_destroy(m_handle);
    }
  }
}

SDClip::SDClip(SDClip&& other) noexcept
    : m_handle{other.m_handle}
{
  other.m_handle = nullptr;
}

SDClip& SDClip::operator=(SDClip&& other) noexcept
{
  if (this != &other)
  {
    if (m_handle)
    {
      const auto& sd = sd::liblibrediffusion::instance();
      if (sd.available)
        sd.clip_destroy(m_handle);
    }
    m_handle = other.m_handle;
    other.m_handle = nullptr;
  }
  return *this;
}

SDXLEmbeddings::~SDXLEmbeddings()
{
  reset();
}

SDXLEmbeddings::SDXLEmbeddings(SDXLEmbeddings&& other) noexcept
    : embeddings{other.embeddings}
    , pooled_embeds{other.pooled_embeds}
    , time_ids{other.time_ids}
{
  other.embeddings = nullptr;
  other.pooled_embeds = nullptr;
  other.time_ids = nullptr;
}

SDXLEmbeddings& SDXLEmbeddings::operator=(SDXLEmbeddings&& other) noexcept
{
  if (this != &other)
  {
    reset();
    embeddings = other.embeddings;
    pooled_embeds = other.pooled_embeds;
    time_ids = other.time_ids;
    other.embeddings = nullptr;
    other.pooled_embeds = nullptr;
    other.time_ids = nullptr;
  }
  return *this;
}

void SDXLEmbeddings::reset()
{
  const auto& sd = sd::liblibrediffusion::instance();
  if (sd.available)
  {
    if (embeddings)
    {
      sd.cuda_free(embeddings);
      embeddings = nullptr;
    }
    if (pooled_embeds)
    {
      sd.cuda_free(pooled_embeds);
      pooled_embeds = nullptr;
    }
    if (time_ids)
    {
      sd.cuda_free(time_ids);
      time_ids = nullptr;
    }
  }
}

StreamDiffusion::StreamDiffusion() noexcept
    : m_sd{sd::liblibrediffusion::instance()}
{
  m_prev_inputs.workflow.value = {};
  m_prev_inputs.add_noise.value = {};
  m_prev_inputs.prompt.value = {};
  m_prev_inputs.negative_prompt.value = {};
  m_prev_inputs.model.value = {};
  m_prev_inputs.seed.value = {};
  m_prev_inputs.guidance.value = {};
  m_prev_inputs.t1.value = {};
  m_prev_inputs.size.value = {};
  m_prev_inputs.cfg.value = {};
  m_prev_inputs.add_noise.value = {};
  m_prev_inputs.denoise_batch.value = {};
}

StreamDiffusion::~StreamDiffusion()
{
  if (m_cached_engine)
  {
    EngineCache::instance().release(m_cached_engine);
    m_cached_engine = nullptr;
  }
}

bool StreamDiffusion::is_available() noexcept
{
  return sd::liblibrediffusion::instance().available;
}

void StreamDiffusion::blendTextures()
{
  const auto model_sz = m_cur_input.size();

  const int byte_count = model_sz.width() * model_sz.height() * 4;
  if(inputs.feed_prev_in > 0 && inputs.feed_prev_out > 0
     && m_prev_input.image.size() == model_sz && m_prev_output.image.size() == model_sz)
  {
    const uint8_t* prev_input = (const uint8_t*)m_prev_input.storage.data();
    const uint8_t* prev_output = (const uint8_t*)m_prev_output.storage.data();
    uint8_t* cur_input = m_cur_input.bits();
    float alpha = inputs.feed_prev_in;
    float beta = inputs.feed_prev_out;
    const float sum = alpha + beta;
    if(sum > 1.f)
    {
      alpha /= sum;
      beta /= sum;
    }
    const int a = std::clamp(int(inputs.feed_prev_in * 256.f), 0, 256);
    const int b = std::clamp(int(inputs.feed_prev_out * 256.f), 0, 256 - a);
    const int c = 256 - a - b;

    for(int i = 0; i < byte_count; i += 4)
    {
      cur_input[i + 0]
          = (c * cur_input[i + 0] + a * prev_input[i + 0] + b * prev_output[i + 0] + 128)
            >> 8;
      cur_input[i + 1]
          = (c * cur_input[i + 1] + a * prev_input[i + 1] + b * prev_output[i + 1] + 128)
            >> 8;
      cur_input[i + 2]
          = (c * cur_input[i + 2] + a * prev_input[i + 2] + b * prev_output[i + 2] + 128)
            >> 8;
    }
  }
  else if(inputs.feed_prev_in > 0)
  {
    // Blend previous input
    if(m_prev_input.image.size() == model_sz)
    {
      const uint8_t* prev_input = (const uint8_t*)m_prev_input.storage.data();
      uint8_t* cur_input = m_cur_input.bits();
      const int a = std::clamp(int(inputs.feed_prev_in * 256.f), 0, 256);
      const int c = 256 - a;

      for(int i = 0; i < byte_count; i += 4)
      {
        cur_input[i + 0] = (c * cur_input[i + 0] + a * prev_input[i + 0] + 128) >> 8;
        cur_input[i + 1] = (c * cur_input[i + 1] + a * prev_input[i + 1] + 128) >> 8;
        cur_input[i + 2] = (c * cur_input[i + 2] + a * prev_input[i + 2] + 128) >> 8;
      }
    }
  }
  else if(inputs.feed_prev_out > 0)
  {
    // Blend previous input
    if(m_prev_output.image.size() == model_sz)
    {
      const uint8_t* prev_output = (const uint8_t*)m_prev_output.storage.data();
      uint8_t* cur_input = m_cur_input.bits();
      const int a = std::clamp(int(inputs.feed_prev_out * 256.f), 0, 256);
      const int c = 256 - a;

      for(int i = 0; i < byte_count; i += 4)
      {
        cur_input[i + 0] = (c * cur_input[i + 0] + a * prev_output[i + 0] + 128) >> 8;
        cur_input[i + 1] = (c * cur_input[i + 1] + a * prev_output[i + 1] + 128) >> 8;
        cur_input[i + 2] = (c * cur_input[i + 2] + a * prev_output[i + 2] + 128) >> 8;
      }
    }
  }
}

bool StreamDiffusion::createConfiguration(const inputs_t& in_config, const std::vector<int>& timestep_indices)
{
  if (!m_sd.available)
    return false;

  auto width = in_config.size.value.x;
  auto height = in_config.size.value.y;

  if (timestep_indices.empty())
    return false;

  // Determine model type and mode from workflow
  librediffusion_model_type_t model_type = MODEL_SD_15;
  librediffusion_pipeline_mode_t pipeline_mode = MODE_SINGLE_FRAME;

  switch (in_config.workflow)
  {
    case Workflow::SD_TXT2IMG:
    case Workflow::SD_IMG2IMG:
    case Workflow::SD_TXT2IMG_CONTROLNET:
    case Workflow::SD_TXT2IMG_IPADAPTER:
    case Workflow::SD_IMG2IMG_IPADAPTER:
      model_type = MODEL_SD_15;
      pipeline_mode = MODE_SINGLE_FRAME;
      break;
    case Workflow::SDTURBO_TXT2IMG:
    case Workflow::SDTURBO_IMG2IMG:
      model_type = MODEL_SD_TURBO;
      pipeline_mode = MODE_SINGLE_FRAME;
      break;
    case Workflow::SDXL_TXT2IMG:
    case Workflow::SDXL_IMG2IMG:
      model_type = MODEL_SDXL_TURBO;
      pipeline_mode = MODE_SINGLE_FRAME;
      break;
    case Workflow::V2V_TXT2IMG:
    case Workflow::V2V_IMG2IMG:
      model_type = MODEL_SD_15;
      pipeline_mode = MODE_TEMPORAL_V2V;
      break;
  }

  // Check if we already have a cached engine
  bool need_new_engine = false;
  if (!m_cached_engine)
  {
    need_new_engine = true;
  }
  else if (m_cached_engine->model_path != in_config.model.value
           || m_cached_engine->pipeline_mode != pipeline_mode)
  {
    // Need different engine
    EngineCache::instance().release(m_cached_engine);
    m_cached_engine = nullptr;
    need_new_engine = true;
  }
  // else keep existing engine

  if (need_new_engine)
  {
    // Try to acquire from cache (key = model path + pipeline mode)
    m_cached_engine = EngineCache::instance().acquire(in_config.model.value, pipeline_mode);

    if (!m_cached_engine)
    {
      auto new_engine = std::make_unique<CachedEngine>();
      new_engine->model_path = in_config.model.value;
      new_engine->pipeline_mode = pipeline_mode;

      // Create CLIP encoders
      std::string clip1_path = in_config.model.value + "/clip.engine";
      new_engine->clip1 = new SDClip{clip1_path.c_str()};
      if (!*new_engine->clip1)
        return false;

      if(model_type == MODEL_SDXL_TURBO)
      {
        std::string clip2_path = in_config.model.value + "/clip2.engine";
        new_engine->clip2 = new SDClip{clip2_path.c_str()};
        if (!*new_engine->clip2)
          return false;
      }

      // Store in cache
      m_cached_engine = EngineCache::instance().store(std::move(new_engine));
      // FIXME evict engines if unused
    }
  }
  else
  {
    qDebug() << "StreamDiffusion: keeping existing engine, will reinit buffers";
  }

  // Store configuration state (this is per-instance, not cached)
  m_config_state.model_type = model_type;
  m_config_state.pipeline_mode = pipeline_mode;
  m_config_state.width = width;
  m_config_state.height = height;
  m_config_state.latent_width = width / 8;
  m_config_state.latent_height = height / 8;
  m_config_state.batch_size = 1;
  m_config_state.timestep_indices = std::move(timestep_indices);
  m_config_state.denoising_steps = m_config_state.timestep_indices.size();

  m_config_state.unet_engine_path = in_config.model.value + "/unet.engine";
  m_config_state.vae_encoder_path = in_config.model.value + "/vae_encoder.engine";
  m_config_state.vae_decoder_path = in_config.model.value + "/vae_decoder.engine";

  // Model-specific settings
  if(model_type == MODEL_SD_TURBO)
  {
    m_config_state.use_denoising_batch = false;
    m_config_state.do_add_noise = in_config.add_noise;
    m_config_state.denoising_steps = 1;
    m_config_state.cfg_type = 0;
    m_config_state.delta = in_config.delta;
    m_config_state.guidance_scale = 0.0f;
    m_config_state.text_seq_len = 77;
    m_config_state.text_hidden_dim = 1024;
    m_config_state.clip_pad_token = 0;
  }
  else if(model_type == MODEL_SDXL_TURBO)
  {
    m_config_state.use_denoising_batch = false;
    m_config_state.do_add_noise = in_config.add_noise;
    m_config_state.denoising_steps = 1;
    m_config_state.cfg_type = 0;
    m_config_state.delta = in_config.delta;
    m_config_state.guidance_scale = 0.0f;
    m_config_state.text_seq_len = 77;
    m_config_state.text_hidden_dim = 2048;
    m_config_state.pooled_embedding_dim = 1280;
    m_config_state.clip_pad_token = 0;
  }
  else // MODEL_SD_15
  {
    m_config_state.use_denoising_batch = in_config.denoise_batch;
    m_config_state.do_add_noise = in_config.add_noise;
    switch (in_config.cfg)
    {
      case None:
        m_config_state.cfg_type = SD_CFG_NONE;
        break;
      case Full:
        m_config_state.cfg_type = SD_CFG_FULL;
        break;
      case Self:
        m_config_state.cfg_type = SD_CFG_SELF;
        break;
      case Initialize:
        m_config_state.cfg_type = SD_CFG_INITIALIZE;
        break;
    }
    m_config_state.delta = in_config.delta;
    m_config_state.guidance_scale = in_config.guidance;
    m_config_state.text_seq_len = 77;
    m_config_state.text_hidden_dim = 768;
    m_config_state.clip_pad_token = 49407;
  }

  // Create config handle for pipeline
  SDConfig config;
  if (!config)
    return false;

  // Apply settings via C API
  m_sd.config_set_device(config.get(), 0);
  m_sd.config_set_model_type(config.get(), model_type);
  m_sd.config_set_pipeline_mode(config.get(), pipeline_mode);
  m_sd.config_set_dimensions(
      config.get(), width, height, m_config_state.latent_width, m_config_state.latent_height);
  m_sd.config_set_batch_size(config.get(), m_config_state.batch_size);
  m_sd.config_set_denoising_steps(config.get(), m_config_state.denoising_steps);
  m_sd.config_set_guidance_scale(config.get(), m_config_state.guidance_scale);
  m_sd.config_set_delta(config.get(), m_config_state.delta);
  m_sd.config_set_add_noise(config.get(), m_config_state.do_add_noise ? 1 : 0);
  m_sd.config_set_denoising_batch(config.get(), m_config_state.use_denoising_batch ? 1 : 0);
  m_sd.config_set_cfg_type(
      config.get(), static_cast<librediffusion_cfg_type_t>(m_config_state.cfg_type));
  m_sd.config_set_text_config(
      config.get(), m_config_state.text_seq_len, m_config_state.text_hidden_dim,
      m_config_state.clip_pad_token);

  if(model_type == MODEL_SDXL_TURBO)
  {
    m_sd.config_set_sdxl_config(config.get(), m_config_state.pooled_embedding_dim, 6);
  }

  m_sd.config_set_unet_engine(config.get(), m_config_state.unet_engine_path.c_str());
  m_sd.config_set_vae_encoder(config.get(), m_config_state.vae_encoder_path.c_str());
  m_sd.config_set_vae_decoder(config.get(), m_config_state.vae_decoder_path.c_str());

  m_sd.config_set_timestep_indices(
      config.get(), m_config_state.timestep_indices.data(),
      m_config_state.timestep_indices.size());

  // Temporal coherence settings for V2V mode
  if(pipeline_mode == MODE_TEMPORAL_V2V)
  {
    m_sd.config_set_temporal_params(
        config.get(),
        1,                            // use_cached_attn
        in_config.add_noise ? 1 : 0,  // use_feature_injection
        0.8f,                         // injection_strength
        0.78f,                        // similarity_threshold
        1,                            // cache_interval
        1);                           // cache_maxframes
  }

  // Create or reinit pipeline
  if (m_cached_engine->pipeline && *m_cached_engine->pipeline)
  {
    // Rreinit buffers with new config
    m_sd.pipeline_reinit_buffers(m_cached_engine->pipeline->get(), config.get());
  }
  else
  {
    // Reload whole pipeline
    delete m_cached_engine->pipeline;
    m_cached_engine->pipeline = new SDPipeline{config.get()};

    if (!*m_cached_engine->pipeline)
      return false;
  }

  m_prev_inputs = inputs;
  return true;
}

bool StreamDiffusion::updatePromptEmbedding(const std::string& prompt, SDXLEmbeddings& embeddings)
{
  if (!m_sd.available || !m_cached_engine || !m_cached_engine->pipeline || !m_cached_engine->clip1)
    return false;

  if(m_config_state.model_type == MODEL_SDXL_TURBO)
  {
    if (!m_cached_engine->clip2)
      return false;

    // Compute SDXL embeddings
    librediffusion_error_t err = m_sd.clip_compute_embeddings_sdxl(
        m_cached_engine->clip1->get(), m_cached_engine->clip2->get(), prompt.c_str(),
        m_config_state.batch_size, m_config_state.height, m_config_state.width,
        nullptr, // default stream
        &embeddings.embeddings, &embeddings.pooled_embeds, &embeddings.time_ids);

    if(err != LIBREDIFFUSION_SUCCESS)
      return false;

    // Prepare SDXL conditioning
    m_sd.prepare_sdxl_conditioning(
        m_cached_engine->pipeline->get(), embeddings.pooled_embeds, embeddings.time_ids);
  }
  else
  {
    // Compute standard CLIP embeddings
    librediffusion_error_t err = m_sd.clip_compute_embeddings(
        m_cached_engine->clip1->get(), prompt.c_str(), m_config_state.clip_pad_token,
        nullptr, // default stream
        &embeddings.embeddings);

    if(err != LIBREDIFFUSION_SUCCESS)
      return false;
  }
  return true;
}

bool StreamDiffusion::updatePromptEmbeddings(const std::string& prompt, std::vector<SDXLEmbeddings>& embeddings)
{
  if (!m_sd.available || !m_cached_engine || !m_cached_engine->pipeline || !m_cached_engine->clip1)
    return false;

  // Reset existing embeddings
  embeddings.clear();

  // Split if necessary:
  if (auto weights = parse_input_string(prompt))
  {
    ossia::small_vector<float, 8> bweight;
    ossia::small_vector<librediffusion_half_t*, 8> bembeds;
    for (const auto& [k, v] : *weights)
    {
      SDXLEmbeddings e;
      updatePromptEmbedding(k, e);
      bembeds.push_back(e.embeddings);
      embeddings.push_back(std::move(e));
      bweight.push_back(v);
    }

    m_sd.blend_embeds(m_cached_engine->pipeline->get(), bembeds.data(), bweight.data(), bembeds.size(), m_config_state.text_seq_len, m_config_state.text_hidden_dim);
  }
  else
  {
    SDXLEmbeddings e;
    updatePromptEmbedding(prompt, e);
    embeddings.push_back(std::move(e));

    m_sd.prepare_embeds(m_cached_engine->pipeline->get(), embeddings.front().embeddings,
                        m_config_state.text_seq_len, m_config_state.text_hidden_dim);
  }
  return true;
}

bool StreamDiffusion::updateScheduler(const std::string& timestep_str)
{
  if (!m_sd.available || !m_cached_engine || !m_cached_engine->pipeline)
    return false;

  auto timestep_indices = get_steps(timestep_str);
  if (timestep_indices.empty())
    return false;

  // For turbo models, only use first step
  if(m_config_state.model_type == MODEL_SDXL_TURBO
     || m_config_state.model_type == MODEL_SD_TURBO)
  {
    timestep_indices.resize(1);
  }

  m_config_state.timestep_indices = std::move(timestep_indices);
  m_config_state.denoising_steps = m_config_state.timestep_indices.size();

  // Build scheduler arrays from precomputed tables
  static thread_local std::vector<float> timesteps;
  timesteps.clear();
  static thread_local std::vector<float> alpha_list;
  alpha_list.clear();
  static thread_local std::vector<float> beta_list;
  beta_list.clear();
  static thread_local std::vector<float> c_skip_list;
  c_skip_list.clear();
  static thread_local std::vector<float> c_out_list;
  c_out_list.clear();

  std::span<const int> scheduler_timesteps;
  std::span<const streamdiffusion::TimestepParams> scheduler_params;
  // Get the appropriate scheduler parameters
  {

    using namespace streamdiffusion;
    switch(m_config_state.model_type)
    {
      case MODEL_SD_15:
        scheduler_timesteps = SCHEDULER_SIMIANLUO_LCM_DREAMSHAPER_V7::TIMESTEP_VALUES;
        scheduler_params = SCHEDULER_SIMIANLUO_LCM_DREAMSHAPER_V7::TIMESTEP_PARAMS;
        break;
      case MODEL_SD_TURBO:
        scheduler_timesteps = SCHEDULER_STABILITYAI_SD_TURBO::TIMESTEP_VALUES;
        scheduler_params = SCHEDULER_STABILITYAI_SD_TURBO::TIMESTEP_PARAMS;
        break;
      case MODEL_SDXL_TURBO:
        scheduler_timesteps = SCHEDULER_STABILITYAI_SDXL_TURBO::TIMESTEP_VALUES;
        scheduler_params = SCHEDULER_STABILITYAI_SDXL_TURBO::TIMESTEP_PARAMS;
        break;
    }

    for(int idx : m_config_state.timestep_indices)
    {
      if(idx < 0 || idx >= std::ssize(scheduler_params))
        continue;
      if(idx < 0 || idx >= std::ssize(scheduler_timesteps))
        continue;
      auto params = scheduler_params[idx];
      int t = scheduler_timesteps[idx];

      timesteps.push_back(static_cast<float>(t));
      alpha_list.push_back(params.alpha_prod_t_sqrt);
      beta_list.push_back(params.beta_prod_t_sqrt);
      c_skip_list.push_back(params.c_skip);
      c_out_list.push_back(params.c_out);
    }
  }

  if (timesteps.empty())
    return false;

  m_sd.prepare_scheduler(m_cached_engine->pipeline->get(),
                          timesteps.data(),
                          alpha_list.data(),
                          beta_list.data(),
                          c_skip_list.data(),
                          c_out_list.data(),
                          timesteps.size());

  return true;
}

void StreamDiffusion::operator()()
{
  // Check library availability
  if (!m_sd.available)
    return;

  const auto& in_config = this->inputs;
  if(in_config.model.value.empty())
    return;

  // Check for configuration changes that require pipeline recreation
  const auto prev_t1 = get_steps(m_prev_inputs.t1.value);
  const auto new_t1 = get_steps(in_config.t1.value);
  const auto n_prev_t1 = prev_t1.size();
  const auto n_new_t1 = new_t1.size();
  
  bool need_rebuild = false;
  bool need_update_scheduler = false;
  bool need_update_positive_embeds = false;
  bool need_update_negative_embeds = false;
  bool need_reseed = false;
  bool need_update_guidance = false;
  bool need_update_delta = false;
  if (n_prev_t1 != n_new_t1 || n_new_t1 <= 0)
    need_rebuild = true;
  if (m_prev_inputs.add_noise.value != in_config.add_noise.value)
    need_rebuild = true;
  if (m_prev_inputs.denoise_batch.value != in_config.denoise_batch.value)
    need_rebuild = true;
  if (m_prev_inputs.model.value != in_config.model.value)
    need_rebuild = true;
  if (m_prev_inputs.workflow.value != in_config.workflow.value)
    need_rebuild = true;
  if (m_prev_inputs.size.value.x != in_config.size.value.x)
    need_rebuild = true;
  if (m_prev_inputs.size.value.y != in_config.size.value.y)
    need_rebuild = true;
  if (m_prev_inputs.cfg.value != in_config.cfg.value)
    need_rebuild = true;
  if(std::signbit(m_prev_inputs.guidance.value - 1.) != std::signbit(in_config.guidance.value - 1.))
    need_rebuild = true;

  if (m_prev_inputs.t1.value != in_config.t1.value)
    need_update_scheduler = true;
  if (m_prev_inputs.prompt.value != in_config.prompt.value || m_embeddings.empty())
    need_update_positive_embeds = true;
  if (m_prev_inputs.negative_prompt.value != in_config.negative_prompt.value || !m_negative_embeddings)
    need_update_negative_embeds = true;
  if (m_prev_inputs.seed.value != in_config.seed.value)
    need_reseed = true;
  if (m_prev_inputs.guidance.value != in_config.guidance.value)
    need_update_guidance = true;
  if (m_prev_inputs.delta.value != in_config.delta.value)
    need_update_delta= true;

  if (need_rebuild)
  {
    // Don't delete the pipeline - createConfiguration will reinit it
    // This preserves the expensive TensorRT engines
    m_embeddings.clear();
    m_negative_embeddings.reset();
    need_update_scheduler = true;
    need_update_positive_embeds = true;
    need_update_negative_embeds = true;
    need_reseed = true;
    need_update_guidance = false;
    need_update_delta = false;
  }

  if (in_config.prompt.value.empty())
    return;

  // Create or reinit pipeline if needed
  if (need_rebuild || !m_cached_engine || !m_cached_engine->pipeline)
  {
    if (!createConfiguration(in_config, new_t1))
      return;
  }

  if (!m_cached_engine || !m_cached_engine->pipeline)
    return;

  const int model_tex_w = in_config.size.value.x;
  const int model_tex_h = in_config.size.value.y;

  unsigned char* input_tex_bytes{inputs.image.texture.bytes};
  m_cur_input = QImage{};

  // Create output texture
  switch (this->inputs.workflow)
  {
    case Workflow::SD_TXT2IMG:
    case Workflow::SD_TXT2IMG_CONTROLNET:
    case Workflow::SD_TXT2IMG_IPADAPTER:
    case Workflow::SDTURBO_TXT2IMG:
    case Workflow::SDXL_TXT2IMG:
    case Workflow::V2V_TXT2IMG:
      this->outputs.image.create(model_tex_w, model_tex_h);
      break;
    case Workflow::SD_IMG2IMG:
    case Workflow::SD_IMG2IMG_IPADAPTER:
    case Workflow::SDTURBO_IMG2IMG:
    case Workflow::SDXL_IMG2IMG:
    case Workflow::V2V_IMG2IMG:
    {
      if(inputs.image.texture.width <= 0)
        return;
      if(inputs.image.texture.height <= 0)
        return;
      const auto model_sz = QSize(model_tex_w, model_tex_h);

      m_cur_input = QImage(
          inputs.image.texture.bytes, inputs.image.texture.width,
          inputs.image.texture.height, QImage::Format_RGBA8888);

      if(model_tex_w != inputs.image.texture.width
         || model_tex_h != inputs.image.texture.height)
      {
        m_cur_input = m_cur_input.scaled(
            model_sz, Qt::IgnoreAspectRatio, Qt::FastTransformation);

        input_tex_bytes = m_cur_input.bits();
      }

      blendTextures();

      this->outputs.image.create(model_tex_w, model_tex_h);
      break;
    }
  }

  // Update scheduler if timesteps changed
  if (need_update_scheduler)
  {
    if (!updateScheduler(in_config.t1.value))
      return;
  }

  // Update embeddings if prompt changed
  if (need_update_positive_embeds)
  {
    bool ok = updatePromptEmbeddings(in_config.prompt.value, m_embeddings);
    if(!ok) {
      qDebug("Invalid prompt");
      return;
    }
  }

  if (need_update_negative_embeds)
  {
    bool ok = updatePromptEmbedding(in_config.negative_prompt.value, m_negative_embeddings);
    if(!ok) {
      qDebug("Invalid negative prompt");
      return;
    }
    m_sd.prepare_negative_embeds(m_cached_engine->pipeline->get(), m_negative_embeddings.embeddings,
                                 m_config_state.text_seq_len, m_config_state.text_hidden_dim);
  }

  // Handle seed change
  if (need_reseed)
  {
    m_sd.reseed(m_cached_engine->pipeline->get(), in_config.seed.value);
  }
  if (need_update_guidance)
  {
    m_config_state.guidance_scale = in_config.guidance.value;
    m_sd.set_guidance_scale(m_cached_engine->pipeline->get(), in_config.guidance.value);
  }
  if (need_update_delta)
  {
    m_config_state.delta = in_config.delta.value;
    m_sd.set_delta(m_cached_engine->pipeline->get(), in_config.delta.value);
  }


  // Run inference
  switch (this->inputs.workflow)
  {
    case Workflow::SD_TXT2IMG:
    case Workflow::SD_TXT2IMG_CONTROLNET:
    case Workflow::SD_TXT2IMG_IPADAPTER:
    case Workflow::SDTURBO_TXT2IMG:
    case Workflow::SDXL_TXT2IMG:
    case Workflow::V2V_TXT2IMG:
      m_sd.txt2img(m_cached_engine->pipeline->get(),
                   outputs.image.texture.bytes,
                   in_config.size.value.x,
                   in_config.size.value.y);
      break;

    case Workflow::SD_IMG2IMG:
    case Workflow::SD_IMG2IMG_IPADAPTER:
    case Workflow::SDTURBO_IMG2IMG:
    case Workflow::SDXL_IMG2IMG:
    case Workflow::V2V_IMG2IMG:
      if (input_tex_bytes)
      {
        m_sd.img2img(
            m_cached_engine->pipeline->get(), input_tex_bytes,
            outputs.image.texture.bytes, model_tex_w, model_tex_h);
        if(inputs.feed_prev_in > 0)
        {
#if QT_VERSION > QT_VERSION_CHECK(6, 6, 0)
          m_prev_input.storage.assign(
              (const char*)input_tex_bytes,
              (const char*)input_tex_bytes + m_cur_input.sizeInBytes());
#else
          m_prev_input.storage.clear();
          m_prev_input.storage.insert(
              0, (const char*)input_tex_bytes, m_cur_input.sizeInBytes());
#endif
          m_prev_input.image = QImage(
              (unsigned char*)m_prev_input.storage.data(), model_tex_w, model_tex_h,
              QImage::Format_RGBA8888);
        }

        if(inputs.feed_prev_out > 0)
        {
#if QT_VERSION > QT_VERSION_CHECK(6, 6, 0)
          m_prev_output.storage.assign(
              (const char*)outputs.image.texture.bytes,
              (const char*)outputs.image.texture.bytes + m_cur_input.sizeInBytes());
#else
          m_prev_output.storage.clear();
          m_prev_output.storage.insert(
              0, (const char*)outputs.image.texture.bytes, m_cur_input.sizeInBytes());
#endif
          m_prev_output.image = QImage(
              (unsigned char*)m_prev_output.storage.data(), model_tex_w, model_tex_h,
              QImage::Format_RGBA8888);
        }
      }
      break;
  }

  this->outputs.image.texture.changed = true;
  m_prev_inputs = inputs;
}

}
