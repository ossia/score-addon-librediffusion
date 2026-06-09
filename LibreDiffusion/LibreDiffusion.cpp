/**
 * LibreDiffusion - Ported to StreamDiffusion C API via dynamic loading
 */

#include "LibreDiffusion.hpp"

#include "EngineCache.hpp"
#include "schedulers/lcm_dreamshaper_v7.hpp"
#include "schedulers/sd-turbo.hpp"
#include "schedulers/sdxl-turbo.hpp"

#include <ossia/detail/fmt.hpp>
#include <ossia/detail/small_vector.hpp>

#include <ctre.hpp>
#include <rapidhash.h>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3.hpp>
#include <State/ValueParser.hpp>
#include <QDebug>


#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ranges>

namespace
{
// Monotonic wall-clock seconds for the Phase-C steady render phase (decoupled from score's tick rate).
inline double now_s_steady()
{
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
}

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

  auto view = all_numbers(s) | std::views::transform([](auto&& match) {
    return std::clamp(match.to_number(10), 0, 49);
  });

  std::vector<int> result;
  result.reserve(4);
  std::ranges::copy(view, std::back_inserter(result));
  return result;
}

// True for every workflow that drives a ControlNet (control-aware unet.engine +
// controlnet.engine): SD1.5 / SDXL, txt2img and img2img. ControlNet conditioning
// is orthogonal to txt2img/img2img — only the starting latent differs (txt2img =
// pure noise, img2img = VAE-encoded input frame). Both modes feed the Control/Style
// inlet identically (set_controlnet_cond_rgba).
static bool is_controlnet_workflow(int8_t wf) noexcept
{
  switch (wf)
  {
    case StreamDiffusion::Workflow::SD_TXT2IMG_CONTROLNET:
    case StreamDiffusion::Workflow::SD_IMG2IMG_CONTROLNET:
    case StreamDiffusion::Workflow::SDXL_TXT2IMG_CONTROLNET:
    case StreamDiffusion::Workflow::SDXL_IMG2IMG_CONTROLNET:
      return true;
    default:
      return false;
  }
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

SDFluxStream::SDFluxStream(
    const char* transformer, const char* qwen, const char* vae_decoder,
    const char* vae_encoder, const char* tokenizer_json, int Th, int Tw,
    unsigned long long seed)
{
  const auto& sd = sd::liblibrediffusion::instance();
  if (sd.available && sd.flux2_stream_create)
  {
    m_handle = sd.flux2_stream_create(
        transformer, qwen, vae_decoder, vae_encoder, tokenizer_json, Th, Tw, seed);
  }
}

SDFluxStream::~SDFluxStream()
{
  reset();
}

void SDFluxStream::reset()
{
  if (m_handle)
  {
    const auto& sd = sd::liblibrediffusion::instance();
    if (sd.available && sd.flux2_stream_destroy)
      sd.flux2_stream_destroy(m_handle);
    m_handle = nullptr;
  }
}

SDFluxStream::SDFluxStream(SDFluxStream&& other) noexcept
    : m_handle{other.m_handle}
{
  other.m_handle = nullptr;
}

SDFluxStream& SDFluxStream::operator=(SDFluxStream&& other) noexcept
{
  if (this != &other)
  {
    reset();
    m_handle = other.m_handle;
    other.m_handle = nullptr;
  }
  return *this;
}

SDRife::SDRife(const char* engine_path)
{
  const auto& sd = sd::liblibrediffusion::instance();
  if (sd.available && sd.rife_create && engine_path)
  {
    m_handle = sd.rife_create(engine_path);
  }
}

SDRife::~SDRife()
{
  reset();
}

void SDRife::reset()
{
  if (m_handle)
  {
    const auto& sd = sd::liblibrediffusion::instance();
    if (sd.available && sd.rife_destroy)
      sd.rife_destroy(m_handle);
    m_handle = nullptr;
  }
}

SDRife::SDRife(SDRife&& other) noexcept
    : m_handle{other.m_handle}
{
  other.m_handle = nullptr;
}

SDRife& SDRife::operator=(SDRife&& other) noexcept
{
  if (this != &other)
  {
    reset();
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
    , m_klein_real_tb{KleinRealFrame{}}  // triple_buffer's default ctor is constrained; init explicitly
    , m_klein_job_tb{KleinJob{}}
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
  m_prev_inputs.controlnet_scale.value = {};
  m_prev_inputs.ipadapter_scale.value = {};
}

StreamDiffusion::~StreamDiffusion()
{
  // Phase C: join the producer BEFORE anything it touches (m_klein_stream, RIFE) is destroyed, so no
  // context/stream is used after free.
  stopKleinProducer();
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
    case Workflow::SD_IMG2IMG_CONTROLNET:
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
    case Workflow::SDXL_TXT2IMG_CONTROLNET:
    case Workflow::SDXL_IMG2IMG_CONTROLNET:
      model_type = MODEL_SDXL_TURBO;
      pipeline_mode = MODE_SINGLE_FRAME;
      break;
    case Workflow::V2V_TXT2IMG:
    case Workflow::V2V_IMG2IMG:
      model_type = MODEL_SD_15;
      pipeline_mode = MODE_TEMPORAL_V2V;
      break;
    case Workflow::FLUX2_KLEIN_TXT2IMG:
    case Workflow::FLUX2_KLEIN_IMG2IMG:
      // Handled by the dedicated klein streaming path (runKlein); never reached here.
      return false;
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

  // ControlNet / IP-Adapter feature flags from the workflow. ControlNet needs a
  // control-aware unet.engine + a separate controlnet.engine in the model folder.
  // IP-Adapter needs an IP-variant unet.engine (auto-detected at init); the IP
  // attention is baked into the unet.engine, so no separate engine path.
  m_config_state.controlnet_index = -1;
  m_config_state.controlnet_scale = in_config.controlnet_scale;
  m_config_state.ipadapter_enabled = false;
  m_config_state.ipadapter_scale = in_config.ipadapter_scale;
  m_config_state.ipadapter_num_tokens = 4;  // SD1.5 base IP-Adapter (16 = plus)
  switch (in_config.workflow)
  {
    case Workflow::SD_TXT2IMG_CONTROLNET:
    case Workflow::SD_IMG2IMG_CONTROLNET:
    case Workflow::SDXL_TXT2IMG_CONTROLNET:
    case Workflow::SDXL_IMG2IMG_CONTROLNET:
      // controlnet_index assigned below once the config handle exists.
      break;
    case Workflow::SD_TXT2IMG_IPADAPTER:
    case Workflow::SD_IMG2IMG_IPADAPTER:
      m_config_state.ipadapter_enabled = true;
      break;
    default:
      break;
  }

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

  // CUDA graph: enable for 1-step workflows (SD-Turbo / SDXS / 1-step Hyper + their ControlNet/IP-Adapter
  // variants). Capturable only for denoising_steps==1, cfg-none, non-V2V (the .so re-gates identically and
  // falls back to per-call enqueue otherwise). Measured ~+16% end-to-end, output bit-identical. See
  // CUDA_GRAPH_INTEGRATION_PLAN.md.
  {
    const bool one_step_graphable
        = m_config_state.denoising_steps == 1
          && m_config_state.cfg_type == SD_CFG_NONE
          && pipeline_mode != MODE_TEMPORAL_V2V;
    m_sd.config_set_cuda_graph(config.get(), one_step_graphable ? 1 : 0);
  }

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

  // ControlNet: register the controlnet engine + conditioning scale. Requires the
  // control-aware unet.engine set above. Returns the net's index (>=0) used later
  // by set_controlnet_cond[_rgba]/set_controlnet_scale.
  if(is_controlnet_workflow(in_config.workflow.value))
  {
    if(!m_sd.config_add_controlnet)
    {
      qDebug() << "StreamDiffusion: ControlNet requested but the librediffusion .so "
                  "does not export config_add_controlnet";
      return false;
    }
    std::string controlnet_engine = in_config.model.value + "/controlnet.engine";
    m_config_state.controlnet_index = m_sd.config_add_controlnet(
        config.get(), controlnet_engine.c_str(), m_config_state.controlnet_scale);
    if(m_config_state.controlnet_index < 0)
    {
      qDebug() << "StreamDiffusion: config_add_controlnet failed (missing "
               << controlnet_engine.c_str() << "?)";
      return false;
    }
  }

  // IP-Adapter: configure the baked-in defaults (token count + uniform scale). The
  // IP-variant unet.engine is auto-detected at init; image tokens are fed per-frame.
  if(m_config_state.ipadapter_enabled)
  {
    if(!m_sd.config_set_ipadapter)
    {
      qDebug() << "StreamDiffusion: IP-Adapter requested but the librediffusion .so "
                  "does not export config_set_ipadapter";
      return false;
    }
    m_sd.config_set_ipadapter(
        config.get(), m_config_state.ipadapter_num_tokens, m_config_state.ipadapter_scale);

    // On-device CLIP image encoder + projection: lets the node turn the raw "Control / Style"
    // texture into IP-Adapter tokens (no host-side Python). Loaded only when both engines exist
    // next to the model; otherwise the pipeline falls back to externally-fed tokens.
    if(m_sd.config_set_ipadapter_image_encoder)
    {
      std::string enc = in_config.model.value + "/clip_image_encoder.engine";
      std::string proj = in_config.model.value + "/ip_image_proj.engine";
      m_sd.config_set_ipadapter_image_encoder(config.get(), enc.c_str(), proj.c_str());
    }
  }

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

  // ControlNet / IP-Adapter engines (controlnet.engine + the control-aware or
  // IP-variant unet.engine) are loaded only at pipeline creation (init_engines),
  // NOT at reinit_buffers. A cached pipeline built for another workflow would not
  // have them (or would still have them when switching away). So when the workflow
  // involves ControlNet or IP-Adapter — or the previous one did — force a fresh
  // pipeline instead of a buffer reinit.
  auto is_feature_workflow = [](int8_t wf) {
    return is_controlnet_workflow(wf)
           || wf == Workflow::SD_TXT2IMG_IPADAPTER
           || wf == Workflow::SD_IMG2IMG_IPADAPTER;
  };
  const bool feature_pipeline
      = is_feature_workflow(in_config.workflow.value)
        || is_feature_workflow(m_prev_inputs.workflow.value);
  if (feature_pipeline && m_cached_engine->pipeline)
  {
    delete m_cached_engine->pipeline;
    m_cached_engine->pipeline = nullptr;
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
      case MODEL_FLUX2_KLEIN_4B:
        // klein uses its own flow-match scheduler computed inside the flux2 stream
        // C-API; this SD scheduler table path is never used for klein.
        return false;
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

namespace
{
// Default location of the klein VAE batch-norm constants when the model folder
// does not ship them. These are model constants (per-channel batchnorm over the
// 128 patchified latent channels), 128 fp32 each.
static constexpr const char* k_klein_bn_fallback_dir
    = "/home/jcelerier/ossia/daydream-streamdiffusion/validation/klein_bn";

// Read 128 fp32 from <path>. Returns false if the file is missing/short.
bool read_bn_file(const std::string& path, std::array<float, 128>& out)
{
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return false;
  f.read(reinterpret_cast<char*>(out.data()), 128 * sizeof(float));
  return f.gcount() == static_cast<std::streamsize>(128 * sizeof(float));
}

// Load the klein bn constants: prefer the model folder, fall back to the
// validation dir.
bool load_klein_bn(
    const std::string& model_dir, std::array<float, 128>& mean,
    std::array<float, 128>& sstd)
{
  if (read_bn_file(model_dir + "/bn_mean.bin", mean)
      && read_bn_file(model_dir + "/bn_std.bin", sstd))
    return true;
  if (read_bn_file(std::string(k_klein_bn_fallback_dir) + "/bn_mean.bin", mean)
      && read_bn_file(std::string(k_klein_bn_fallback_dir) + "/bn_std.bin", sstd))
    return true;
  return false;
}
}

bool StreamDiffusion::createKleinStream(const inputs_t& in_config)
{
  if (!m_sd.available || !m_sd.flux2_stream_create)
    return false;

  const std::string& model = in_config.model.value;

  // Resolution -> packed token grid. H = Th*16, W = Tw*16.
  // Round to multiples of 16 (the klein patch+VAE stride).
  int w = std::max(16, (in_config.size.value.x / 16) * 16);
  int h = std::max(16, (in_config.size.value.y / 16) * 16);
  const int Tw = w / 16;
  const int Th = h / 16;

  // Transformer engine: bf16 (Quality) vs fp8_calib (Speed).
  const int quality = static_cast<int>(in_config.klein_quality.value);
  const std::string transformer
      = model
        + (quality == StreamDiffusion::Speed ? "/transformer_fp8_calib.plan"
                                             : "/transformer_bf16.plan");
  const std::string qwen = model + "/qwen3_encoder_bf16.plan";
  const std::string vae_dec = model + "/vae_decoder_bf16.plan";
  const std::string vae_enc = model + "/vae_encoder_bf16.plan";
  const std::string tok = model + "/tokenizer.json";

  // Fixed seed for temporal coherence (FluxRT convention).
  unsigned long long seed
      = static_cast<unsigned long long>(static_cast<uint32_t>(in_config.seed.value));
  if (seed == 0)
    seed = 52ull;

  // Phase C: the producer thread holds m_klein_stream's handle. Drain+join it BEFORE destroying the
  // old stream (assigning a new SDFluxStream frees the old handle) -> no use-after-free on reconfig.
  stopKleinProducer();

  m_klein_stream = SDFluxStream{
      transformer.c_str(), qwen.c_str(), vae_dec.c_str(), vae_enc.c_str(),
      tok.c_str(), Th, Tw, seed};
  if (!m_klein_stream)
  {
    qDebug() << "FLUX.2-klein: failed to create stream pipeline";
    return false;
  }

  // 2 fixed steps for distilled klein.
  if (m_sd.flux2_stream_set_steps)
    m_sd.flux2_stream_set_steps(m_klein_stream.get(), 2);

  // VAE batch-norm constants (required before frame()).
  std::array<float, 128> bn_mean{}, bn_std{};
  if (!load_klein_bn(model, bn_mean, bn_std))
  {
    qDebug() << "FLUX.2-klein: missing bn_mean.bin/bn_std.bin (model folder or fallback)";
    return false;
  }
  if (m_sd.flux2_stream_set_bn)
    m_sd.flux2_stream_set_bn(m_klein_stream.get(), bn_mean.data(), bn_std.data());

  m_klein_model_path = model;
  m_klein_quality = quality;
  m_klein_w = w;
  m_klein_h = h;
  m_klein_prompt.clear();
  m_klein_have_prev = false;
  m_klein_prev_out.assign((size_t)w * h * 4, 0);
  // reset the cached reference + the RIFE display queue on (re)create
  m_klein_ref_set = false;
  m_klein_ref_hash = 0;
  m_klein_queue.clear();
  // async (Phase C): bump generation (stale keyframes from the old gen are dropped) + reset the
  // render-thread keyframe pair / cached sweep / wall-clock phase state. The producer is stopped
  // above and (re)started lazily by runKleinAsync once async mode is requested.
  ++m_klein_gen;
  m_klein_frames.clear();             // render-thread: drop the pending sub-frame FIFO
  m_klein_have_emit = false;
  m_klein_prod_rate = 0.0;            // re-measure production rate after a config change
  m_klein_drain_credit = 0.0;
  m_klein_have_prev_key = false;     // producer-thread: forget the prev keyframe
  m_klein_last_kf_t = 0.0;
  m_klein_kf_interval = 0.0;          // re-measure cadence after a config change (no stale assumption)
  m_klein_last_tick_t = 0.0;
  m_klein_dbg_last_pub_t = 0.0;       // FPS dump: reset deltas so the first reading isn't bogus
  m_klein_dbg_last_emit_t = 0.0;
  m_klein_async_ref_set = false;
  m_klein_async_ref_hash = 0;
  // remember the RIFE engine path so the producer can lazily create its OWN RIFE handle (the producer
  // thread runs RIFE next to diffusion; the render thread does no GPU compute).
  {
    std::string rp = in_config.model.value + "/rife_ifnet_fp16.plan";
    if (!std::filesystem::exists(rp))
      rp = "/media/data2/flux-trt/engine-rife/rife_ifnet_fp16.plan";
    m_klein_rife_path = rp;
  }
  return true;
}

void StreamDiffusion::emitWithRife(const unsigned char* frame_rgba, int w, int h)
{
  const int exp = std::clamp((int)inputs.rife_exp.value, 0, 3);

  // RIFE disabled / no engine / first frame -> emit the real frame directly.
  if (exp <= 0 || !m_rife || !m_sd.rife_interpolate || !m_klein_have_prev)
  {
    this->outputs.image.create(w, h);
    std::memcpy(outputs.image.texture.bytes, frame_rgba, (size_t)w * h * 4);
    outputs.image.texture.changed = true;
    return;
  }

  // Interpolate prev_out -> frame_rgba into 2^exp frames; emit the last (== cur).
  const int n_max = 1 << exp;
  m_rife_scratch.assign((size_t)n_max * w * h * 4, 0);
  int n = 0;
  auto err = m_sd.rife_interpolate(
      m_rife.get(), m_klein_prev_out.data(), frame_rgba, h, w, m_rife_scratch.data(), &n);

  this->outputs.image.create(w, h);
  if (err == LIBREDIFFUSION_SUCCESS && n > 0)
  {
    // Emit the most recent interpolated frame (display order, last == cur).
    const unsigned char* last = m_rife_scratch.data() + (size_t)(n - 1) * w * h * 4;
    std::memcpy(outputs.image.texture.bytes, last, (size_t)w * h * 4);
  }
  else
  {
    std::memcpy(outputs.image.texture.bytes, frame_rgba, (size_t)w * h * 4);
  }
  outputs.image.texture.changed = true;
}

// ---- Phase C: dedicated producer thread -------------------------------------------------------
// Owns the heavy diffusion. The ONLY thread that calls the flux2 C-API once async is on (the TRT
// execution contexts are single-thread). Pulls the newest job (latest-wins), runs set_reference (VAE
// encode) when the ref changed + the ~150ms 2-step denoise+decode on the klein stream's own low-prio
// CUDA stream, and publishes the finished keyframe into the triple_buffer. Never touches Qt / the
// render thread state, so it cannot hitch the render loop.
void StreamDiffusion::kleinProducerLoop(std::stop_token stop)
{
  KleinJob job;
  bool have_job = false;   // the last job we ran; reused to self-rearm when no newer one has arrived
  for (;;)
  {
    // Pull the freshest job if the render thread published one (lock-free, newest-wins). We do NOT block
    // here if we already have a previous job to reuse — that lets the producer run diffusion FLAT-OUT
    // (back-to-back keyframes) instead of stalling one render-tick (~16ms) waiting to be re-poked. The
    // render thread only refreshes the reference/exp; reusing the last job between its updates is correct.
    {
      KleinJob nj;
      if (m_klein_job_tb.consume(nj))   // newest-wins built in
      {
        job = std::move(nj);
        have_job = true;
        m_klein_job_ready.store(false, std::memory_order_release);
      }
    }
    if (!have_job)
    {
      // Nothing ever submitted yet -> block until the render thread signals the first job (lost-wakeup-safe).
      std::unique_lock<std::mutex> lk(m_klein_wake_mtx);
      m_klein_job_cv.wait(lk, [&]{
        return stop.stop_requested() || m_klein_job_ready.load(std::memory_order_acquire);
      });
      if (stop.stop_requested())
        return;
      m_klein_job_ready.store(false, std::memory_order_release);
      continue;   // loop back to consume the job we were just signalled about
    }
    if (stop.stop_requested())
      return;
    if (!job.valid || !m_klein_stream || !m_sd.flux2_stream_frame_cached)
    {
      have_job = false;
      continue;
    }

    m_klein_producer_busy.store(true, std::memory_order_release);
    const int w = job.w, h = job.h; const size_t nb = (size_t)w * h * 4;

    if (job.ref_changed && m_sd.flux2_stream_set_reference)
    {
      if (m_sd.flux2_stream_set_reference(m_klein_stream.get(), job.ref_rgba.data())
          != LIBREDIFFUSION_SUCCESS)
      {
        m_klein_producer_busy.store(false, std::memory_order_release);
        have_job = false;
        continue;
      }
      job.ref_changed = false;  // ref now cached -> a self-rearm reuse skips the (costly) VAE encode
    }

    // 1. diffuse the keyframe (~150ms) — this thread has the GPU largely to itself now (the render
    //    thread does NO GPU compute; it only uploads a finished sub-frame).
    KleinRealFrame out;
    out.w = w; out.h = h; out.gen = job.gen;
    out.rgba.assign(nb, 0);
    if (m_sd.flux2_stream_frame_cached(m_klein_stream.get(), out.rgba.data())
        != LIBREDIFFUSION_SUCCESS)
    {
      m_klein_producer_busy.store(false, std::memory_order_release);
      have_job = false;
      continue;
    }

    // 2. RIFE the sweep ON THIS THREAD (lazily create the producer-side RIFE handle). Interpolate the
    //    previous keyframe -> this one into 2^exp sub-frames (display order, last == cur). RIFE runs
    //    here, sequentially after diffusion, so it never competes with diffusion on a parallel stream.
    if (job.exp > 0 && m_klein_have_prev_key)
    {
      if (!m_klein_producer_rife && !m_klein_rife_path.empty())
      {
        m_klein_producer_rife = SDRife{m_klein_rife_path.c_str()};
        // The handle defaults to enabled=false / exp=0 -> rife_interpolate would use eff_exp=0 and return
        // a single frame (sweep_n=1, NO interpolation). Must arm it explicitly.
        if (m_klein_producer_rife && m_sd.rife_set_enabled)
          m_sd.rife_set_enabled(m_klein_producer_rife.get(), 1);
      }
      // Keep the handle's exp in sync with the job (the user can change rife_exp live).
      if (m_klein_producer_rife && m_sd.rife_set_interpolation_exp)
        m_sd.rife_set_interpolation_exp(m_klein_producer_rife.get(), job.exp);
      if (m_klein_producer_rife && m_sd.rife_interpolate)
      {
        const int n_max = 1 << job.exp;
        out.sweep.assign((size_t)n_max * nb, 0);
        int n = 0;
        if (m_sd.rife_interpolate(m_klein_producer_rife.get(), m_klein_prev_key.data(),
                                  out.rgba.data(), h, w, out.sweep.data(), &n)
                == LIBREDIFFUSION_SUCCESS && n > 0)
          out.sweep_n = n;
        else { out.sweep.clear(); out.sweep_n = 0; }
      }
    }

    // 3. remember this keyframe as the next sweep's prev, then publish the sweep (lock-free).
    m_klein_prev_key = out.rgba;  // copy before move
    m_klein_have_prev_key = true;
    {
      // Live FPS dump: instantaneous PRODUCER (diffusion/keyframe) rate = 1 / time-since-last-publish.
      const double t = now_s_steady();
      const double pfps = (m_klein_dbg_last_pub_t > 0.0)
          ? 1.0 / std::max(1e-6, t - m_klein_dbg_last_pub_t) : 0.0;
      m_klein_dbg_last_pub_t = t;
      qDebug().nospace() << "[klein-PRODUCER] " << pfps << " fps (keyframe)  sweep_n="
                         << out.sweep_n << " exp=" << job.exp;
    }
    m_klein_real_tb.produce(std::move(out));
    m_klein_producer_busy.store(false, std::memory_order_release);
  }
}

void StreamDiffusion::startKleinProducer()
{
  if (m_klein_producer.joinable())
    return;
  m_klein_producer = std::jthread(
      [this](std::stop_token st) { kleinProducerLoop(st); });
}

void StreamDiffusion::stopKleinProducer()
{
  if (!m_klein_producer.joinable())
    return;
  m_klein_producer.request_stop();
  // wake the producer so it observes stop (cv predicate also checks stop_requested)
  { std::lock_guard<std::mutex> lk(m_klein_wake_mtx); m_klein_job_ready.store(true, std::memory_order_release); }
  m_klein_job_cv.notify_all();
  m_klein_producer.join();
  // drain any queued job so a restart starts clean
  { KleinJob drop; while(m_klein_job_tb.consume(drop)) {} }
  m_klein_job_ready.store(false, std::memory_order_release);
  m_klein_producer_busy.store(false, std::memory_order_release);
}

void StreamDiffusion::runKlein(const inputs_t& in_config)
{
  if (!m_sd.available || !m_sd.flux2_stream_create)
  {
    qDebug() << "FLUX.2-klein: library does not export the flux2 streaming API";
    return;
  }

  int w = std::max(16, (in_config.size.value.x / 16) * 16);
  int h = std::max(16, (in_config.size.value.y / 16) * 16);
  const int quality = static_cast<int>(in_config.klein_quality.value);

  // (Re)create the stream pipeline when the model / resolution / quality changes.
  const bool need_new
      = !m_klein_stream || m_klein_model_path != in_config.model.value
        || m_klein_quality != quality || m_klein_w != w || m_klein_h != h;
  if (need_new)
  {
    // createKleinStream() drains+joins the producer before destroying the old stream handle, so
    // there is no use-after-free even if a keyframe was mid-flight. This is HEAVY (engine (re)load +
    // producer join) and runs on the GUI thread -> a freeze here is expected ONLY on real config change.
    if (!createKleinStream(in_config))
      return;
    w = m_klein_w;
    h = m_klein_h;
  }

  // (Re)create the RIFE interpolator on demand (opt-in via rife_exp > 0).
  // IMPORTANT: in ASYNC mode the PRODUCER thread owns its OWN RIFE handle (m_klein_producer_rife) and
  // runs RIFE itself. Loading a SECOND RIFE engine here on the render side would double the RIFE VRAM
  // (~1.3GB each at the wide dynamic profile) and contend on the GPU -> the ~1fps + "rife_create failed"
  // the user saw. So only create the render-side m_rife for the SYNC path.
  const int exp = std::clamp((int)in_config.rife_exp.value, 0, 3);
  const bool async_mode = in_config.klein_async.value
      && m_sd.flux2_stream_set_reference && m_sd.flux2_stream_frame_cached;
  if (!async_mode && exp > 0 && m_sd.rife_create)
  {
    if (!m_rife)
    {
      std::string rife_engine = in_config.model.value + "/rife_ifnet_fp16.plan";
      if (!std::filesystem::exists(rife_engine))
        rife_engine = "/media/data2/flux-trt/engine-rife/rife_ifnet_fp16.plan";
      m_rife = SDRife{rife_engine.c_str()};
    }
    if (m_rife && m_sd.rife_set_enabled)
    {
      m_sd.rife_set_enabled(m_rife.get(), 1);
      if (m_sd.rife_set_interpolation_exp)
        m_sd.rife_set_interpolation_exp(m_rife.get(), exp);
    }
  }
  else if (m_rife && m_sd.rife_set_enabled)
  {
    m_sd.rife_set_enabled(m_rife.get(), 0);
  }

  // Prompt -> (re)encode the cached Qwen embeds (no-op if unchanged).
  if (in_config.prompt.value.empty())
    return;
  if (m_klein_prompt != in_config.prompt.value)
  {
    // set_prompt touches the Qwen TRT context, which in async mode is owned by the producer thread.
    // Drain+join the producer first so the context is never touched from two threads (it restarts
    // lazily in runKleinAsync). This JOIN blocks the GUI thread until the in-flight keyframe finishes
    // (~150ms) -> a freeze. Should fire ONLY when the prompt actually changes; if it logs every tick,
    // the prompt input is oscillating (the real bug).
    if (m_klein_producer.joinable())
      stopKleinProducer();
    if (m_sd.flux2_stream_set_prompt
        && m_sd.flux2_stream_set_prompt(
               m_klein_stream.get(), in_config.prompt.value.c_str())
               < 0)
    {
      qDebug() << "FLUX.2-klein: set_prompt failed";
      return;
    }
    m_klein_prompt = in_config.prompt.value;
    m_klein_queue.clear();            // new prompt -> don't show stale interpolated frames
    m_klein_frames.clear();           // and drop stale async sub-frames
    m_klein_have_emit = false;
    m_klein_drain_credit = 0.0;       // keep prod_rate (cadence unchanged), reset the credit
    m_klein_have_prev_key = false;
  }

  const int exp_now = std::clamp((int)in_config.rife_exp.value, 0, 3);
  if (exp_now != m_klein_last_exp)
  {
    m_klein_queue.clear(); // interpolation factor changed -> flush the queue
    m_klein_last_exp = exp_now;
  }

  // Async (Phase C): diffusion runs on a dedicated producer thread (low-prio stream); the render
  // thread emits one steady-clock-paced, precomputed RIFE sub-frame per tick (fluid, never blocks).
  // Falls back to the sync path below if the cached reference API is absent.
  if (in_config.klein_async.value
      && m_sd.flux2_stream_set_reference && m_sd.flux2_stream_frame_cached)
  {
    runKleinAsync(in_config);
    return;
  }
  // Left async -> ensure the producer is stopped so the sync path owns the contexts exclusively.
  if (m_klein_producer.joinable())
    stopKleinProducer();

  // ---- Task 2: RIFE display decoupling --------------------------------------------------------
  // When interpolation is on (exp>0), we emit ONE frame per tick from a queue and only run a real
  // diffusion when the queue is empty (i.e. once per ~2^exp ticks). Between real frames the GPU is
  // free — diffusion runs only when necessary. When exp==0 the queue holds a single freshly-diffused
  // frame each tick (full diffusion rate, unchanged behaviour).
  if (!m_klein_queue.empty())
  {
    // Still have interpolated frames to show — emit one, no diffusion this tick.
    this->outputs.image.create(w, h);
    std::memcpy(
        outputs.image.texture.bytes, m_klein_queue.front().data(), (size_t)w * h * 4);
    outputs.image.texture.changed = true;
    m_klein_queue.pop_front();
    return;
  }

  // Queue empty -> we need a new REAL frame. Build the reference and (re)encode only if it changed.
  //  - IMG2IMG: the incoming texture (scaled to w x h), as a reference image.
  //  - TXT2IMG: a neutral (black) reference frame (hashes constant -> encoded once).
  static thread_local std::vector<unsigned char> ref_frame;
  ref_frame.assign((size_t)w * h * 4, 0);

  if (in_config.workflow == Workflow::FLUX2_KLEIN_IMG2IMG)
  {
    if (inputs.image.texture.width <= 0 || inputs.image.texture.height <= 0
        || !inputs.image.texture.bytes)
      return;

    QImage in(
        inputs.image.texture.bytes, inputs.image.texture.width,
        inputs.image.texture.height, QImage::Format_RGBA8888);
    if (inputs.image.texture.width != w || inputs.image.texture.height != h)
      in = in.scaled(QSize(w, h), Qt::IgnoreAspectRatio, Qt::FastTransformation);

    const size_t bytes = std::min<size_t>((size_t)w * h * 4, in.sizeInBytes());
    std::memcpy(ref_frame.data(), in.constBits(), bytes);
  }

  // Task 1: VAE-encode the reference ONLY when it changed (hash the bytes, like the IP-Adapter
  // style path). Falls back to the legacy one-shot flux2_stream_frame if the cached API is absent.
  static thread_local std::vector<unsigned char> out_frame;
  out_frame.assign((size_t)w * h * 4, 0);

  if (m_sd.flux2_stream_set_reference && m_sd.flux2_stream_frame_cached)
  {
    const uint64_t rh = rapidhash(ref_frame.data(), (size_t)w * h * 4);
    if (!m_klein_ref_set || rh != m_klein_ref_hash)
    {
      if (m_sd.flux2_stream_set_reference(m_klein_stream.get(), ref_frame.data())
          != LIBREDIFFUSION_SUCCESS)
      {
        qDebug() << "FLUX.2-klein: set_reference failed";
        return;
      }
      m_klein_ref_hash = rh;
      m_klein_ref_set = true;
    }
    auto err = m_sd.flux2_stream_frame_cached(m_klein_stream.get(), out_frame.data());
    if (err != LIBREDIFFUSION_SUCCESS)
    {
      qDebug() << "FLUX.2-klein: stream_frame_cached failed" << (int)err;
      return;
    }
  }
  else
  {
    // Legacy path (older .so): encode every frame.
    auto err = m_sd.flux2_stream_frame(
        m_klein_stream.get(), ref_frame.data(), out_frame.data());
    if (err != LIBREDIFFUSION_SUCCESS)
    {
      qDebug() << "FLUX.2-klein: stream_frame failed" << (int)err;
      return;
    }
  }

  // Build this tick's output queue.
  if (exp_now <= 0 || !m_rife || !m_sd.rife_interpolate || !m_klein_have_prev)
  {
    // No interpolation (or first real frame): emit the single real frame this tick.
    m_klein_queue.clear();
    m_klein_queue.emplace_back(out_frame);
  }
  else
  {
    // Interpolate prev_real -> cur into 2^exp display-ordered frames; queue them all.
    const int n_max = 1 << exp_now;
    m_rife_scratch.assign((size_t)n_max * w * h * 4, 0);
    int n = 0;
    auto rerr = m_sd.rife_interpolate(
        m_rife.get(), m_klein_prev_out.data(), out_frame.data(), h, w,
        m_rife_scratch.data(), &n);
    m_klein_queue.clear();
    if (rerr == LIBREDIFFUSION_SUCCESS && n > 0)
    {
      for (int i = 0; i < n; ++i)
      {
        const unsigned char* f = m_rife_scratch.data() + (size_t)i * w * h * 4;
        m_klein_queue.emplace_back(f, f + (size_t)w * h * 4);
      }
    }
    else
    {
      m_klein_queue.emplace_back(out_frame); // RIFE failed -> just the real frame
    }
  }

  // Remember this real frame for the next RIFE pass, then emit the first queued frame.
  m_klein_prev_out = out_frame;
  m_klein_have_prev = true;

  this->outputs.image.create(w, h);
  std::memcpy(outputs.image.texture.bytes, m_klein_queue.front().data(), (size_t)w * h * 4);
  outputs.image.texture.changed = true;
  m_klein_queue.pop_front();

  m_prev_inputs = inputs;
}

// Phase C — steady-clock paced async klein. The diffusion runs on the dedicated producer thread
// (kleinProducerLoop) on the klein stream's own low-prio CUDA stream; the render thread (this fn,
// called once per score tick) NEVER blocks on it. Each tick:
//   1. build the reference frame, push a latest-wins job to the producer when the sweep is consumed;
//   2. consume a fresh keyframe from the triple_buffer (newest-wins) -> shift prev<-cur, rebuild the
//      RIFE sweep ONCE, rebase the phase + update the keyframe-rate estimate;
//   3. advance the phase by REAL elapsed wall-time scaled by the keyframe rate;
//   4. emit ONE precomputed sub-frame (cheap memcpy) -> constant per-tick cost -> tight pacing.
void StreamDiffusion::runKleinAsync(const inputs_t& in_config)
{
  const int w = m_klein_w, h = m_klein_h;
  const size_t nbytes = (size_t)w * h * 4;
  const int exp = std::clamp((int)in_config.rife_exp.value, 0, 3);

  // The producer is lazily (re)started here (createKleinStream / set_prompt stop it). Cheap no-op when
  // already running, BUT startKleinProducer() -> jthread ctor only the first time; instrument anyway.
  startKleinProducer();

  // Per-tick wall-clock dt drives the credit-based FIFO drain (frames consumed this tick = prod_rate*dt).
  const double tnow = now_s_steady();
  double dt = (m_klein_last_tick_t > 0.0) ? (tnow - m_klein_last_tick_t) : 0.0;
  m_klein_last_tick_t = tnow;
  dt = std::clamp(dt, 0.0, 0.1);   // guard first tick / scheduler hiccup so the cursor never leaps

  // --- 1. Build this tick's reference frame (img2img: input texture; txt2img: black). ---
  static thread_local std::vector<unsigned char> ref_frame;
  ref_frame.assign(nbytes, 0);
  bool have_input = true;
  if (in_config.workflow == Workflow::FLUX2_KLEIN_IMG2IMG)
  {
    if (inputs.image.texture.width <= 0 || inputs.image.texture.height <= 0
        || !inputs.image.texture.bytes)
      have_input = false;
    else
    {
      QImage in(
          inputs.image.texture.bytes, inputs.image.texture.width,
          inputs.image.texture.height, QImage::Format_RGBA8888);
      if (inputs.image.texture.width != w || inputs.image.texture.height != h)
        in = in.scaled(QSize(w, h), Qt::IgnoreAspectRatio, Qt::FastTransformation);
      std::memcpy(ref_frame.data(), in.constBits(),
                  std::min<size_t>(nbytes, in.sizeInBytes()));
    }
  }

  const int pacing = static_cast<int>(in_config.klein_pacing.value);

  // --- 2. Drain the producer's freshly-published SWEEP into the SUB-FRAME FIFO. The render thread emits
  //        ONE sub-frame per tick (sync's model) -> each RIFE sub-frame is shown exactly once, in order
  //        = 0% dup, monotonic motion (the wall-clock fractional cursor repeated ~10% of frames -> choppy).
  //        The pacing mode only controls how much we let the FIFO BUFFER (latency) vs stay fresh:
  //          Smooth     : keep a few sweeps of slack so the FIFO never drains -> motion never stutters.
  //          Fresh      : ~1 keyframe of slack -> low latency, occasional brief hold if the producer slips.
  //          LowLatency : minimal slack -> if the producer falls behind, DROP backlog to stay live. ---
  KleinRealFrame fresh;
  if (m_klein_real_tb.consume(fresh) && fresh.gen == m_klein_gen)
  {
    // ADAPTIVE: measure the keyframe interval AND the PRODUCTION RATE (sub-frames/sec) from the actual
    // arrivals — no hardcoded fps, self-calibrates to any GPU. Reject only physically-impossible gaps.
    const int n = std::max(1, fresh.sweep_n);
    if (m_klein_last_kf_t > 0.0)
    {
      const double gap = tnow - m_klein_last_kf_t;
      if (gap > 1e-3 && gap < 5.0)
      {
        m_klein_kf_interval = (m_klein_kf_interval <= 0.0)
            ? gap : 0.8 * m_klein_kf_interval + 0.2 * gap;
        const double inst_rate = (double)n / gap;   // this sweep's sub-frames over the interval
        m_klein_prod_rate = (m_klein_prod_rate <= 0.0)
            ? inst_rate : 0.8 * m_klein_prod_rate + 0.2 * inst_rate;
      }
    }
    m_klein_last_kf_t = tnow;

    // Latency budget = how many sub-frames we tolerate buffered ahead before trimming. A sweep contributes
    // sweep_n sub-frames; we express the budget in sweeps so it scales with exp.
    const int budget_sweeps = (pacing == StreamDiffusion::Smooth) ? 3
                            : (pacing == StreamDiffusion::Fresh)  ? 2 : 1;
    const size_t max_frames = (size_t)budget_sweeps * n;

    // Append this sweep's sub-frames in display order (sweep[0..n-1]; sweep ends at the new keyframe).
    if (fresh.sweep_n > 1 && fresh.sweep.size() >= (size_t)n * nbytes)
    {
      for (int i = 0; i < n; ++i)
        m_klein_frames.emplace_back(
            fresh.sweep.begin() + (size_t)i * nbytes, fresh.sweep.begin() + (size_t)(i + 1) * nbytes);
    }
    else
    {
      m_klein_frames.emplace_back(fresh.rgba);   // no interpolation -> the single keyframe
    }

    // Trim from the FRONT (drop the stalest queued frames) if we're over budget -> bounds latency. This
    // is where the modes differ: Smooth allows the most slack, LowLatency the least.
    while (m_klein_frames.size() > max_frames)
      m_klein_frames.pop_front();
  }

  // --- 4. Keep the producer SATURATED: submit a new job whenever it's idle (decoupled from display). The
  //        producer diffuses + RIFEs flat-out on its own thread; the render thread never blocks. ---
  // Publish the freshest reference whenever it CHANGES (newest-wins, lock-free). NOT gated on producer-
  // busy: the producer self-rearms and runs flat-out, so it's almost always busy; gating here would
  // starve live-video ref updates. The producer consumes the latest job at the top of each keyframe;
  // between changes it reuses the last job (re-diffusing the same ref is correct + keeps it saturated).
  if (have_input)
  {
    const uint64_t rh = rapidhash(ref_frame.data(), nbytes);
    if (!m_klein_async_ref_set || rh != m_klein_async_ref_hash)
    {
      KleinJob job;
      job.ref_rgba = ref_frame;           // copy: the producer owns its input
      job.ref_changed = true;             // ref genuinely changed -> producer re-encodes once
      job.w = w; job.h = h;
      job.exp = exp;                      // the producer renders the RIFE sweep at this exp
      job.gen = m_klein_gen;
      job.valid = true;
      m_klein_job_tb.produce(std::move(job));
      m_klein_job_ready.store(true, std::memory_order_release);
      m_klein_job_cv.notify_one();
      m_klein_async_ref_hash = rh;
      m_klein_async_ref_set = true;
    }
    else if (exp != m_klein_exp)
    {
      // ref unchanged but interpolation factor changed -> push a job so the producer adopts the new exp.
      KleinJob job;
      job.ref_rgba = ref_frame;
      job.ref_changed = false;            // ref already cached
      job.w = w; job.h = h;
      job.exp = exp;
      job.gen = m_klein_gen;
      job.valid = true;
      m_klein_job_tb.produce(std::move(job));
      m_klein_job_ready.store(true, std::memory_order_release);
      m_klein_job_cv.notify_one();
    }
    m_klein_exp = exp;
  }

  // --- 5/6. PRESENT via CREDIT-BASED EVEN-SPREAD DRAIN. Advance the FIFO by the MEASURED production rate,
  //          not 1-per-tick. credit += prod_rate*dt; pop floor(credit) frames (keep the remainder). This
  //          spreads repeats (content<display) and skips (content>display, bigger GPU) EVENLY across time
  //          instead of bunching them -> smoothness matches sync regardless of the GPU's actual fps. ---
  if (m_klein_prod_rate > 0.0)
    m_klein_drain_credit += m_klein_prod_rate * dt;
  else
    m_klein_drain_credit += 1.0;   // before the rate is known, fall back to 1-per-tick

  int to_pop = (int)m_klein_drain_credit;
  if (to_pop > 0)
  {
    m_klein_drain_credit -= (double)to_pop;
    // Don't let the credit run away if the FIFO is short (cap to what's available + keep fraction sane).
    while (to_pop > 0 && !m_klein_frames.empty())
    {
      m_klein_last_emit = std::move(m_klein_frames.front());
      m_klein_frames.pop_front();
      m_klein_have_emit = true;
      --to_pop;
    }
    // FIFO drained before satisfying the credit -> we're content-starved this instant; drop the unmet
    // credit (don't bank it, or we'd skip-burst when frames arrive) and hold the last frame.
    if (to_pop > 0)
      m_klein_drain_credit = 0.0;
  }
  // else: credit < 1 this tick -> emit NOTHING new, hold the last frame (an EVENLY-spaced repeat).

  if (!m_klein_have_emit)
    return;   // nothing produced yet (producer warming up)

  this->outputs.image.create(w, h);
  std::memcpy(outputs.image.texture.bytes, m_klein_last_emit.data(),
              std::min<size_t>(nbytes, m_klein_last_emit.size()));
  outputs.image.texture.changed = true;

  // Live FPS dump: instantaneous RECEIVER (display/present) rate = 1 / time-since-last-present.
  {
    const double rfps = (m_klein_dbg_last_emit_t > 0.0)
        ? 1.0 / std::max(1e-6, tnow - m_klein_dbg_last_emit_t) : 0.0;
    m_klein_dbg_last_emit_t = tnow;
    qDebug().nospace() << "[klein-RECEIVER] " << rfps << " fps (display)  prod_rate="
                       << m_klein_prod_rate << " fifo=" << (int)m_klein_frames.size();
  }
}

void StreamDiffusion::operator()()
{
  // Check library availability
  if (!m_sd.available)
    return;

  const auto& in_config = this->inputs;
  if(in_config.model.value.empty())
    return;

  // FLUX.2-klein has its own self-contained streaming pipeline (separate engines,
  // tokenizer, scheduler and noise handled inside the C-API). It does not use the
  // SD pipeline / CLIP / EngineCache machinery, so dispatch it early.
  if(in_config.workflow == Workflow::FLUX2_KLEIN_TXT2IMG
     || in_config.workflow == Workflow::FLUX2_KLEIN_IMG2IMG)
  {
    runKlein(in_config);
    return;
  }

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
    case Workflow::FLUX2_KLEIN_TXT2IMG:
    case Workflow::FLUX2_KLEIN_IMG2IMG:
      // Handled by runKlein() above; never reached here.
      return;
    case Workflow::SD_TXT2IMG:
    case Workflow::SD_TXT2IMG_CONTROLNET:
    case Workflow::SDXL_TXT2IMG_CONTROLNET:
    case Workflow::SD_TXT2IMG_IPADAPTER:
    case Workflow::SDTURBO_TXT2IMG:
    case Workflow::SDXL_TXT2IMG:
    case Workflow::V2V_TXT2IMG:
      this->outputs.image.create(model_tex_w, model_tex_h);
      break;
    case Workflow::SD_IMG2IMG:
    case Workflow::SD_IMG2IMG_CONTROLNET:
    case Workflow::SDXL_IMG2IMG_CONTROLNET:
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

  // ControlNet: feed this frame's control map (the "Control / Style" texture),
  // preprocessed externally (canny/depth/pose/...). The C-API uploads the host
  // RGBA->RGB NCHW [0,1] fp16 on-device and tiles it to the UNet batch. Resize to
  // the model resolution first (the controlnet engine expects [1,3,H,W] at H/W).
  if(is_controlnet_workflow(in_config.workflow.value)
     && m_config_state.controlnet_index >= 0 && m_sd.set_controlnet_cond_rgba)
  {
    // Live-adjust the conditioning scale.
    if(m_sd.set_controlnet_scale
       && in_config.controlnet_scale != m_config_state.controlnet_scale)
    {
      m_config_state.controlnet_scale = in_config.controlnet_scale;
      m_sd.set_controlnet_scale(
          m_cached_engine->pipeline->get(), m_config_state.controlnet_index,
          m_config_state.controlnet_scale);
    }

    const auto& ctl = in_config.control.texture;
    if(ctl.bytes && ctl.width > 0 && ctl.height > 0)
    {
      QImage ctl_img(
          ctl.bytes, ctl.width, ctl.height, QImage::Format_RGBA8888);
      if(ctl.width != model_tex_w || ctl.height != model_tex_h)
        ctl_img = ctl_img.scaled(
            QSize(model_tex_w, model_tex_h), Qt::IgnoreAspectRatio,
            Qt::FastTransformation);

      m_sd.set_controlnet_cond_rgba(
          m_cached_engine->pipeline->get(), m_config_state.controlnet_index,
          ctl_img.constBits(), model_tex_h, model_tex_w);
    }
    else
    {
      // No control map this frame -> the controlnet engine would run on stale/zero
      // cond. Skip the frame rather than emit an unconditioned image.
      qDebug() << "StreamDiffusion: ControlNet workflow but no control image on the "
                  "'Control / Style' input";
      return;
    }
  }

  // IP-Adapter: feed the raw "Control / Style" texture through the on-device CLIP image
  // encoder + projection (set_ipadapter_image) to produce the image tokens the IP-variant
  // unet.engine consumes. The style is static, so we only re-encode when the texture content
  // changes (hash) — not every frame. The per-layer style scale stays host-adjustable.
  if(m_config_state.ipadapter_enabled)
  {
    if(m_sd.set_ipadapter_scale
       && in_config.ipadapter_scale != m_config_state.ipadapter_scale)
    {
      m_config_state.ipadapter_scale = in_config.ipadapter_scale;
      m_sd.set_ipadapter_scale(
          m_cached_engine->pipeline->get(), m_config_state.ipadapter_scale);
    }

    if(m_sd.set_ipadapter_image)
    {
      const auto& style = in_config.control.texture;
      if(style.bytes && style.width > 0 && style.height > 0)
      {
        // Fingerprint the style image; re-encode only on change (first frame or new image).
        const std::size_t nbytes = std::size_t(style.width) * style.height * 4;
        const uint64_t h = rapidhash(style.bytes, nbytes);
        if(!m_config_state.ipadapter_image_set || h != m_config_state.ipadapter_style_hash)
        {
          m_sd.set_ipadapter_image(
              m_cached_engine->pipeline->get(), style.bytes, style.height, style.width);
          m_config_state.ipadapter_style_hash = h;
          m_config_state.ipadapter_image_set = true;
        }
      }
      else if(!m_config_state.ipadapter_image_set)
      {
        // No style image yet and none ever set -> the IP-variant unet.engine would throw on
        // run (no tokens). Skip this frame rather than crash; the user must wire a style image.
        qDebug() << "StreamDiffusion: IP-Adapter workflow but no style image on the "
                    "'Control / Style' input";
        return;
      }
    }
  }

  // Run inference
  switch (this->inputs.workflow)
  {
    case Workflow::FLUX2_KLEIN_TXT2IMG:
    case Workflow::FLUX2_KLEIN_IMG2IMG:
      // Handled by runKlein() above; never reached here.
      return;
    case Workflow::SD_TXT2IMG:
    case Workflow::SD_TXT2IMG_CONTROLNET:
    case Workflow::SDXL_TXT2IMG_CONTROLNET:
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
    case Workflow::SD_IMG2IMG_CONTROLNET:
    case Workflow::SDXL_IMG2IMG_CONTROLNET:
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
