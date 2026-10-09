/**
 * StreamDiffusion node: drives the librediffusion C API (loaded at runtime) for every model
 * family through one frame pipeline. Per tick the render thread turns the inputs into an AsyncJob;
 * produceFrame() renders it (+ RIFE) either inline (sync) or on the producer thread (Async).
 */

#include "LibreDiffusion.hpp"

#include "EngineCache.hpp"
#include "ModelBuilder.hpp"
#include "schedulers/lcm_dreamshaper_v7.hpp"
#include "schedulers/sd-turbo.hpp"
#include "schedulers/sdxl-turbo.hpp"

#include <boost/container/small_vector.hpp>

// rapidhash is a libossia 3rdparty single-header. Use the host's copy when it is on the include
// path (score dev build), otherwise the vendored copy.
#if __has_include(<rapidhash.h>)
#include <rapidhash.h>
#else
#include "compat/rapidhash.h"
#endif
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>

namespace
{
// Monotonic wall-clock seconds for the paced presentation (independent of the host's tick rate).
inline double now_s_steady()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Non-throwing std::filesystem::exists: a symlink loop, an unreadable parent or a dead mount raises
// filesystem_error, and the host's render path has no try/catch around operator().
inline bool file_exists(const std::string& p) noexcept
{
  std::error_code ec;
  return std::filesystem::exists(p, ec) && !ec;
}

inline uint64_t hash_bytes(const void* p, size_t n) noexcept
{
  return n ? rapidhash(p, n) : 0;
}

// Upper bound for the Resolution port: a preset or an automation can deliver anything, and halp's
// texture create() computes width * height * 4 in `int`.
constexpr int k_max_resolution = 8192;
}

// -------------------------------------------------------------------------------------------------
// Prompt language: "(some text: 0.1), (other text: 0.5)" -> weighted sub-prompts blended in
// embedding space.
// -------------------------------------------------------------------------------------------------
namespace lo
{
struct WeightedPromptElement
{
  std::string text;
  double value;
};

std::optional<std::vector<WeightedPromptElement>> parse_input_string(std::string_view str);
}

BOOST_FUSION_ADAPT_STRUCT(lo::WeightedPromptElement, (std::string, text)(double, value))

namespace lo
{
namespace x3 = boost::spirit::x3;

struct TextContentTag;
struct NumberTag;
struct WeightedPromptElementTag;
struct DataListTag;

const x3::rule<TextContentTag, std::string> text_content = "text_content";
const x3::rule<NumberTag, double> number = "number";
const x3::rule<WeightedPromptElementTag, WeightedPromptElement> data_item = "data_item";
const x3::rule<DataListTag, std::vector<WeightedPromptElement>> data_list = "data_list";

auto const text_content_def = x3::lexeme[*(x3::char_ - ':')];
auto const number_def = x3::double_;
auto const data_item_def = '(' >> text_content >> ':' >> number >> ')';
auto const data_list_def = data_item % ',';

BOOST_SPIRIT_DEFINE(text_content, number, data_item, data_list);

// Blend weights multiply every element of the conditioning tensor and are narrowed to float on
// the way to blend_embeds, so anything beyond this is either a typo or an overflow.
constexpr double k_max_prompt_weight = 1e6;

std::optional<std::vector<WeightedPromptElement>> parse_input_string(std::string_view str)
{
  std::vector<WeightedPromptElement> result_data;
  auto iterator = str.begin();
  auto const end_iterator = str.end();

  const auto success
      = x3::phrase_parse(iterator, end_iterator, data_list, x3::ascii::space, result_data);
  if(!(success && iterator == end_iterator))
    return std::nullopt;

  for(auto& e : result_data)
  {
    // x3::double_ parses "nan" and "inf", and a huge weight becomes inf once narrowed to float.
    if(!std::isfinite(e.value) || std::abs(e.value) > k_max_prompt_weight)
      return std::nullopt;
    // `*(char_ - ':')` accepts '\0', and every consumer reaches CLIP through c_str().
    std::erase(e.text, '\0');
  }
  return result_data;
}

// Parse the "Timesteps" control into scheduler indices (slots in a 50-entry table).
//
// Returns a (possibly empty) list when the string was understood -- empty means "nothing typed
// yet" -- and std::nullopt when it holds something that is not a usable index. Separators are
// commas and whitespace; a token must parse completely; non-finite values are rejected; finite
// out-of-range values are clamped into [0, 49]; "[15, 25]" is tolerated.
static std::optional<std::vector<int>> get_steps(std::string_view in)
{
  auto ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  auto sep = [&](unsigned char c) { return ws(c) || c == ','; };

  std::size_t b = 0, e = in.size();
  while(b < e && ws(in[b]))
    ++b;
  while(e > b && ws(in[e - 1]))
    --e;
  std::string_view s = in.substr(b, e - b);
  if(s.size() >= 2 && s.front() == '[' && s.back() == ']')
    s = s.substr(1, s.size() - 2);

  std::vector<int> result;
  std::size_t pos = 0;
  while(pos < s.size())
  {
    while(pos < s.size() && sep(s[pos]))
      ++pos;
    if(pos >= s.size())
      break;
    std::size_t te = pos;
    while(te < s.size() && !sep(s[te]))
      ++te;

    const std::string tok{s.substr(pos, te - pos)};
    char* parse_end = nullptr;
    const double d = std::strtod(tok.c_str(), &parse_end);
    if(parse_end != tok.c_str() + tok.size())
      return std::nullopt;
    if(!std::isfinite(d))
      return std::nullopt;

    result.push_back(std::clamp(static_cast<int>(std::clamp(d, -1e9, 1e9)), 0, 49));
    pos = te;
  }
  return result;
}

// Parse the Timesteps control as FLUX FlowMatch sigmas (native scale): comma-separated floats,
// high -> low, each in (0, 1]. Empty when the field is not a sigma list (e.g. the SD-style
// "15, 25" default), which means "the model's natural 2-step schedule".
static std::vector<float> get_sigmas(const std::string& s)
{
  std::vector<float> out;
  std::size_t i = 0;
  while(i < s.size())
  {
    std::size_t j = s.find(',', i);
    if(j == std::string::npos)
      j = s.size();
    const std::string tok = s.substr(i, j - i);
    char* end = nullptr;
    const float v = std::strtof(tok.c_str(), &end);
    if(end != tok.c_str())
      out.push_back(v);
    i = j + 1;
  }
  for(float v : out)
    if(!(v > 0.f && v <= 1.f))
      return {};
  return out;
}

// Cross-attention width declared by a bundle's manifest, or 0 when there is no usable manifest.
// SD1.5 encodes to 768, SD2.1 to 1024, and the workflow enum cannot tell them apart. A scan and
// not a JSON parse: one integer out of a file we generate ourselves.
static int bundle_embedding_dim(const std::string& model_dir)
{
  std::ifstream f{model_dir + "/bundle.json", std::ios::binary};
  if(!f)
    return 0;
  const std::string text{std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{}};

  const auto key = text.find("\"embedding_dim\"");
  if(key == std::string::npos)
    return 0;
  auto p = text.find(':', key);
  if(p == std::string::npos)
    return 0;
  ++p;
  while(p < text.size() && (text[p] == ' ' || text[p] == '\t'))
    ++p;
  int dim = 0;
  const auto begin = p;
  while(p < text.size() && text[p] >= '0' && text[p] <= '9')
  {
    dim = dim * 10 + (text[p] - '0');
    if(dim > 65536)
      return 0;
    ++p;
  }
  return p > begin ? dim : 0;
}

// Read 128 fp32 (a klein VAE batch-norm constant) from <path>.
static bool read_bn_file(const std::string& path, std::array<float, 128>& out)
{
  std::ifstream f(path, std::ios::binary);
  if(!f)
    return false;
  f.read(reinterpret_cast<char*>(out.data()), 128 * sizeof(float));
  return f.gcount() == static_cast<std::streamsize>(128 * sizeof(float));
}

// The RIFE engine for a bundle: its own, else one shared by the bundles of the parent folder.
static std::string rife_engine_path(const std::string& model_dir)
{
  for(const std::string candidate :
      {model_dir + "/rife_ifnet_fp16.plan", model_dir + "/../rife_ifnet_fp16.plan"})
    if(file_exists(candidate))
      return candidate;
  return {};
}

static const char* build_state_name(BuildState s) noexcept
{
  switch(s)
  {
    case BuildState::Idle:
      return "idle";
    case BuildState::Extracting:
      return "extracting";
    case BuildState::Running:
      return "running";
    case BuildState::Done:
      return "done";
    case BuildState::Failed:
      return "failed";
  }
  return "";
}

// -------------------------------------------------------------------------------------------------
// Workflow classification
// -------------------------------------------------------------------------------------------------
StreamDiffusion::Family StreamDiffusion::familyOf(Workflow wf) noexcept
{
  switch(wf)
  {
    case FLUX2_KLEIN_TXT2IMG:
    case FLUX2_KLEIN_IMG2IMG:
    case FLUX2_KLEIN_INPAINT:
      return Family::Klein;
    case IMG2IMG_TURBO:
      return Family::Turbo;
    default:
      return Family::SD;
  }
}

bool StreamDiffusion::isImg2Img(Workflow wf) noexcept
{
  switch(wf)
  {
    case SD_IMG2IMG:
    case SD_IMG2IMG_CONTROLNET:
    case SD_IMG2IMG_IPADAPTER:
    case SDTURBO_IMG2IMG:
    case SDXL_IMG2IMG:
    case SDXL_IMG2IMG_CONTROLNET:
    case V2V_IMG2IMG:
    case FLUX2_KLEIN_IMG2IMG:
    case FLUX2_KLEIN_INPAINT:
    case IMG2IMG_TURBO:
      return true;
    default:
      return false;
  }
}

// ControlNet conditioning is orthogonal to txt2img/img2img: only the starting latent differs.
bool StreamDiffusion::isControlNet(Workflow wf) noexcept
{
  switch(wf)
  {
    case SD_TXT2IMG_CONTROLNET:
    case SD_IMG2IMG_CONTROLNET:
    case SDXL_TXT2IMG_CONTROLNET:
    case SDXL_IMG2IMG_CONTROLNET:
      return true;
    default:
      return false;
  }
}

bool StreamDiffusion::isIPAdapter(Workflow wf) noexcept
{
  return wf == SD_TXT2IMG_IPADAPTER || wf == SD_IMG2IMG_IPADAPTER;
}

// -------------------------------------------------------------------------------------------------
// Lifetime
// -------------------------------------------------------------------------------------------------
StreamDiffusion::StreamDiffusion() noexcept
    : m_sd{lib()}
{
  // halp::texture_output's constructor sets changed = true with bytes == nullptr.
  outputs.image.texture.changed = false;
}

StreamDiffusion::~StreamDiffusion()
{
  // The producer touches every pipeline handle: join it before anything is destroyed.
  releaseFamily();
  if(m_cached_engine)
    EngineCache::instance().release(m_cached_engine);
}

bool StreamDiffusion::is_available() noexcept
{
  return lib().available;
}

StreamDiffusion::SetupKey StreamDiffusion::setupKey(const inputs_t& in)
{
  SetupKey k;
  k.model = in.model.value;
  k.prompt = in.prompt.value;
  k.negative_prompt = in.negative_prompt.value;
  k.timesteps = in.t1.value;
  k.width = in.size.value.x;
  k.height = in.size.value.y;
  k.gpu = in.gpu.value;
  k.seed = in.seed.value;
  k.workflow = in.workflow.value;
  k.cfg = in.cfg.value;
  k.klein_quality = in.klein_quality.value;
  k.add_noise = in.add_noise.value;
  k.denoise_batch = in.denoise_batch.value;
  k.valid = true;
  return k;
}

// Without this a failing configuration re-attempts a full engine load on every tick.
bool StreamDiffusion::setupBlocked(const inputs_t& in) const
{
  return m_failed_setup.valid && m_failed_setup == setupKey(in);
}

// The latent grid is width/step by height/step (8 for SD, 16 for klein); round DOWN to the nearest
// multiple, floor `step`. Diagnoses once per distinct bad value.
bool StreamDiffusion::resolveResolution(const inputs_t& in, int step, int& w, int& h)
{
  const int rw = in.size.value.x;
  const int rh = in.size.value.y;
  const bool usable = (rw > 0 && rh > 0);

  w = usable ? std::max(step, std::min(rw, k_max_resolution) / step * step) : rw;
  h = usable ? std::max(step, std::min(rh, k_max_resolution) / step * step) : rh;

  if((!usable || w != rw || h != rh) && (rw != m_reported_size_w || rh != m_reported_size_h))
  {
    m_reported_size_w = rw;
    m_reported_size_h = rh;
    std::fprintf(
        stderr, "StreamDiffusion: Resolution %dx%d -- %s (usable range %d..%d, multiples of %d)\n",
        rw, rh, usable ? "adjusted" : "frame skipped", step, k_max_resolution, step);
  }
  return usable;
}

// A NaN never equals itself, so every `m_prev_inputs.x != in.x` change gate would fire on every
// tick for a NaN knob; inf/NaN also reach int(feed_prev * 256.f), where the cast is UB.
void StreamDiffusion::sanitizeControls()
{
  const auto fix = [](float& v, float fallback) {
    if(!std::isfinite(v))
      v = fallback;
  };
  fix(inputs.guidance.value, 1.0f);
  fix(inputs.delta.value, 1.0f);
  fix(inputs.feed_prev_in.value, 0.0f);
  fix(inputs.feed_prev_out.value, 0.0f);
  fix(inputs.controlnet_scale.value, 0.6f);
  fix(inputs.ipadapter_scale.value, 0.7f);
  fix(inputs.lora_scale.value, 1.0f);

  // These two feed fixed-point blend arithmetic; the others reach the library as-is.
  inputs.feed_prev_in.value = std::clamp(inputs.feed_prev_in.value, 0.0f, 1.0f);
  inputs.feed_prev_out.value = std::clamp(inputs.feed_prev_out.value, 0.0f, 1.0f);
  inputs.gpu.value = std::max(0, inputs.gpu.value);
}

// -------------------------------------------------------------------------------------------------
// Per tick
// -------------------------------------------------------------------------------------------------
void StreamDiffusion::operator()()
{
  sanitizeControls();
  // One-shot values are consumed by the tick that received them, whatever the host does with them
  // afterwards (score clears them after the run; other back-ends may not).
  const bool build_requested = inputs.build.value.has_value();
  const bool triggered = inputs.trigger.value.has_value();
  inputs.build.value.reset();
  inputs.trigger.value.reset();

  // The builder is pure host code: engines can be built before the runtime library is usable.
  builderTick(inputs, build_requested);
  if(!m_sd.available)
    return;

  const inputs_t& in = inputs;
  const Family family = familyOf(in.workflow.value);
  if(family != m_family || in.gpu.value != m_device)
  {
    releaseFamily();
    m_family = family;
    if(in.gpu.value != m_device)
    {
      // The cached SD engine is bound to the old device; hand it back and let configureSD acquire
      // or build one for the new device.
      m_device = in.gpu.value;
      if(m_cached_engine)
      {
        EngineCache::instance().release(m_cached_engine);
        m_cached_engine = nullptr;
      }
    }
  }

  if(in.model.value.empty())
    return;
  if(setupBlocked(in))
    return;
  if(!configure(in))
  {
    noteSetupFailure(in);
    return;
  }
  noteSetupSuccess();

  renderTick(in, triggered);
  m_prev_inputs = inputs;
}

bool StreamDiffusion::configure(const inputs_t& in)
{
  switch(m_family)
  {
    case Family::SD:
      return configureSD(in);
    case Family::Klein:
      return configureKlein(in);
    case Family::Turbo:
      return configureTurbo(in);
  }
  return false;
}

void StreamDiffusion::builderTick(const inputs_t& in, bool build_requested)
{
  auto& builder = ModelBuilder::instance();
  if(build_requested)
  {
    BuildRequest request;
    request.python_cache = in.python_cache.value;
    request.build_folder
        = in.build_folder.value.empty() ? in.model.value : in.build_folder.value;
    request.options = in.build_options.value;
    request.gpu = in.gpu.value;
    std::string error;
    if(!builder.start(std::move(request), error))
      std::fprintf(stderr, "StreamDiffusion: build not started: %s\n", error.c_str());
  }

  const BuildStatus status = builder.status();
  std::string text = build_state_name(status.state);
  if(!status.message.empty())
    text += ": " + status.message;
  if(text != m_build_status)
  {
    m_build_status = text;
    outputs.build_status.value = text;
    std::fprintf(stderr, "StreamDiffusion: build %s\n", text.c_str());
    // A finished build may be exactly the bundle a blocked setup was waiting for.
    if(status.state == BuildState::Done)
      noteSetupSuccess();
  }
}

// The frame's inputs, at model resolution. False when nothing can be rendered this tick.
bool StreamDiffusion::buildJob(const inputs_t& in, AsyncJob& job)
{
  const Workflow wf = in.workflow.value;
  job.w = m_w;
  job.h = m_h;
  job.exp = std::clamp(in.rife_exp.value, 0, 3);
  job.gen = m_gen;
  const size_t nbytes = (size_t)m_w * m_h * 4;

  if(isImg2Img(wf))
  {
    // An unconnected "In" port delivers a width and height while `bytes` is still null.
    const auto& t = in.image.texture;
    if(t.width <= 0 || t.height <= 0 || !t.bytes)
      return false;
    m_cur_input = rgba_image(t.bytes, t.width, t.height);
    if(t.width != m_w || t.height != m_h)
      m_cur_input = m_cur_input.scaled({m_w, m_h});
    blendFeedback(m_cur_input, in);
    job.ref_rgba = m_cur_input.px;
  }
  else if(m_family == Family::Klein)
  {
    job.ref_rgba.assign(nbytes, 0);  // txt2img: a neutral reference, encoded once
  }
  job.ref_hash = hash_bytes(job.ref_rgba.data(), job.ref_rgba.size());

  if(isControlNet(wf) || isIPAdapter(wf))
  {
    const auto& c = in.control.texture;
    if(c.bytes && c.width > 0 && c.height > 0)
    {
      // The ControlNet engine expects the model geometry; the IP-Adapter image encoder resizes itself.
      rgba_image ctl(c.bytes, c.width, c.height);
      if(isControlNet(wf) && (c.width != m_w || c.height != m_h))
        ctl = ctl.scaled({m_w, m_h});
      job.control_w = ctl.w;
      job.control_h = ctl.h;
      job.control_rgba = std::move(ctl.px);
      job.control_hash = hash_bytes(job.control_rgba.data(), job.control_rgba.size());
      m_reported_no_control = false;
    }
    else if(isControlNet(wf))
    {
      // The ControlNet engine would run on stale or zero conditioning: skip the frame. (An
      // IP-Adapter style is static and may stay applied; renderSD decides.)
      if(!m_reported_no_control)
      {
        m_reported_no_control = true;
        std::fprintf(
            stderr, "StreamDiffusion: ControlNet workflow but no image on the 'Control / Style' input\n");
      }
      return false;
    }
  }

  job.controlnet_scale = in.controlnet_scale.value;
  job.ipadapter_scale = in.ipadapter_scale.value;
  job.guidance = in.guidance.value;
  job.delta = in.delta.value;
  job.lora_scale = in.lora_scale.value;
  job.seed = in.seed.value;
  if(m_family == Family::Turbo)
    job.ehs = in.ehs.value;

  // Everything that changes the output, widened to one type so there is no padding to hash.
  const auto bits = [](float f) { return (uint64_t)std::bit_cast<uint32_t>(f); };
  const uint64_t parts[] = {
      job.ref_hash,
      job.control_hash,
      hash_bytes(job.ehs.data(), job.ehs.size() * sizeof(float)),
      bits(job.controlnet_scale),
      bits(job.ipadapter_scale),
      bits(job.guidance),
      bits(job.delta),
      bits(job.lora_scale),
      (uint64_t)(uint32_t)job.seed,
      (uint64_t)job.w,
      (uint64_t)job.h,
      (uint64_t)job.exp};
  job.key = hash_bytes(parts, sizeof parts);
  return true;
}

// Feed prev. input / output: blend the previous input and/or output frame into this one.
void StreamDiffusion::blendFeedback(rgba_image& cur, const inputs_t& in)
{
  const int a = std::clamp(int(in.feed_prev_in.value * 256.f), 0, 256);
  const int b = std::clamp(int(in.feed_prev_out.value * 256.f), 0, 256 - a);
  const bool use_in = a > 0 && m_prev_input.size() == cur.size();
  const bool use_out = b > 0 && m_prev_output.size() == cur.size();
  if(!use_in && !use_out)
    return;
  const int wa = use_in ? a : 0;
  const int wb = use_out ? b : 0;
  const int wc = 256 - wa - wb;
  const uint8_t* pin = m_prev_input.constBits();
  const uint8_t* pout = m_prev_output.constBits();
  uint8_t* px = cur.bits();
  const size_t n = cur.sizeInBytes();
  for(size_t i = 0; i < n; i += 4)
    for(int c = 0; c < 3; ++c)
      px[i + c] = (wc * px[i + c] + wa * (use_in ? pin[i + c] : 0) + wb * (use_out ? pout[i + c] : 0)
                   + 128)
                  >> 8;
}

void StreamDiffusion::renderTick(const inputs_t& in, bool triggered)
{
  const double tnow = now_s_steady();
  // Capped so a stall (engine load, blocked host) does not burst-drain the FIFO afterwards.
  const double dt = std::clamp(m_last_tick_t > 0.0 ? tnow - m_last_tick_t : 0.0, 0.0, 0.1);
  m_last_tick_t = tnow;

  const bool manual = in.manual.value;
  const bool fire = triggered;
  const int budget_sweeps = in.pacing.value == Smooth ? 3 : in.pacing.value == Fresh ? 2 : 1;

  // Settle the producer first: (re)starting or stopping it bumps the generation the job carries.
  if(in.async_mode.value)
    ensureProducer(m_continuous && !manual);
  else if(m_producer && m_producer->running())
    stopProducer();

  AsyncJob job;
  const bool have_job = buildJob(in, job);

  auto adopt = [&](const AsyncFrame& frame) {
    m_consumer.on_keyframe(frame, tnow, budget_sweeps, m_gen);
    if(frame.gen != m_gen)
      return;
    if(in.feed_prev_in.value > 0)
      m_prev_input = m_cur_input;
    if(in.feed_prev_out.value > 0)
      m_prev_output = rgba_image(frame.rgba.data(), frame.w, frame.h);
  };

  const unsigned char* out = nullptr;
  size_t out_bytes = 0;
  bool have_frame = false;
  if(in.async_mode.value)
  {
    if(have_job && (manual ? fire : job.key != m_submitted_key))
    {
      m_submitted_key = job.key;
      m_producer->submit(std::move(job));
    }
    AsyncFrame fresh;
    if(m_producer->consume(fresh))
      adopt(fresh);
    have_frame = m_consumer.present(dt, out, out_bytes);
  }
  else
  {
    // Render when the FIFO ran dry (every tick without interpolation, every 2^exp ticks with it).
    if(have_job && (manual ? fire : m_consumer.empty()))
    {
      AsyncFrame frame;
      if(produceFrame(job, frame))
        adopt(frame);
    }
    have_frame = m_consumer.present_next(out, out_bytes);
  }

  // Publish only a NEW frame: a failed render or a held frame leaves the host's texture alone.
  const bool publish = have_frame && m_consumer.advanced();
  if(publish)
  {
    const size_t nbytes = (size_t)m_w * m_h * 4;
    outputs.image.create(m_w, m_h);
    std::memcpy(outputs.image.texture.bytes, out, std::min(nbytes, out_bytes));
  }
  outputs.image.texture.changed = publish;
}

void StreamDiffusion::ensureProducer(bool continuous)
{
  if(!m_producer)
    m_producer = std::make_unique<AsyncFrameProducer<AsyncJob, AsyncFrame>>(
        [this](AsyncJob& job, AsyncFrame& out) { return produceFrame(job, out); });
  if(m_producer->running() && m_producer->continuous() != continuous)
    stopProducer();
  if(!m_producer->running())
    m_producer->start(continuous);
}

// Drain + join the producer and forget every frame of the current configuration. After this the
// render thread owns the pipelines again.
void StreamDiffusion::stopProducer()
{
  if(m_producer)
    m_producer->stop();
  ++m_gen;
  m_consumer.reset();
  m_prev_key.clear();
  m_applied_ref_hash = 0;
  m_applied_control_hash = 0;
  m_reported_no_style = false;
  m_submitted_key = 0;
  m_last_tick_t = 0.0;
}

void StreamDiffusion::releaseFamily()
{
  stopProducer();
  m_rife.reset();
  m_rife_failed_path.clear();
  m_model_dir.clear();
  m_w = m_h = 0;

  m_klein_stream.reset();
  m_klein_quality = -1;
  m_klein_seed = 0;
  m_klein_prompt.clear();
  m_klein_sched.clear();
  m_klein_mask_hash = 0;

  m_i2it.reset();
  m_i2it_clip.reset();
  m_i2it_embeddings.reset();
  m_i2it_prompt.clear();
}

// -------------------------------------------------------------------------------------------------
// Rendering (producer thread when Async is on, render thread otherwise)
// -------------------------------------------------------------------------------------------------
bool StreamDiffusion::produceFrame(AsyncJob& job, AsyncFrame& out)
{
  const size_t nbytes = (size_t)job.w * job.h * 4;
  if(nbytes == 0)
    return false;
  out.w = job.w;
  out.h = job.h;
  out.gen = job.gen;
  out.rgba.assign(nbytes, 0);
  const double t0 = now_s_steady();
  if(!renderFrame(job, out.rgba.data()))
    return false;
  if(job.exp > 0 && m_prev_key.size() == nbytes)
    interpolate(job, out);
  out.produce_seconds = now_s_steady() - t0;
  m_prev_key = out.rgba;
  return true;
}

bool StreamDiffusion::renderFrame(const AsyncJob& job, unsigned char* out_rgba)
{
  switch(m_family)
  {
    case Family::SD:
      return renderSD(job, out_rgba);
    case Family::Klein:
      return renderKlein(job, out_rgba);
    case Family::Turbo:
      return renderTurbo(job, out_rgba);
  }
  return false;
}

bool StreamDiffusion::renderSD(const AsyncJob& job, unsigned char* out_rgba)
{
  if(!m_cached_engine || !m_cached_engine->pipeline)
    return false;
  auto pipe = m_cached_engine->pipeline->get();
  SDConfigState& s = m_config_state;

  if(s.controlnet_index >= 0)
  {
    if(job.control_hash != m_applied_control_hash)
    {
      m_sd.set_controlnet_cond_rgba(
          pipe, s.controlnet_index, job.control_rgba.data(), job.control_h, job.control_w);
      m_applied_control_hash = job.control_hash;
    }
    if(job.controlnet_scale != s.controlnet_scale)
    {
      s.controlnet_scale = job.controlnet_scale;
      m_sd.set_controlnet_scale(pipe, s.controlnet_index, job.controlnet_scale);
    }
  }
  if(s.ipadapter_enabled)
  {
    // The style is static: re-encode only when the image changes, keep it when it is unplugged.
    if(!job.control_rgba.empty() && job.control_hash != m_applied_control_hash)
    {
      m_sd.set_ipadapter_image(pipe, job.control_rgba.data(), job.control_h, job.control_w);
      m_applied_control_hash = job.control_hash;
      s.ipadapter_tokens = true;
    }
    if(!s.ipadapter_tokens)
    {
      // An IP-variant UNet with no tokens at all: skip rather than run on nothing.
      if(!m_reported_no_style)
      {
        m_reported_no_style = true;
        std::fprintf(
            stderr, "StreamDiffusion: IP-Adapter workflow but no style image on the 'Control / Style' input\n");
      }
      return false;
    }
    if(job.ipadapter_scale != s.ipadapter_scale)
    {
      s.ipadapter_scale = job.ipadapter_scale;
      m_sd.set_ipadapter_scale(pipe, job.ipadapter_scale);
    }
  }

  // The cheap live parameters: pushed from here so an automated knob never stops the producer.
  if(!s.seeded || job.seed != s.seed)
  {
    m_sd.reseed(pipe, job.seed);
    s.seed = job.seed;
    s.seeded = true;
  }
  if(job.guidance != s.guidance_scale)
  {
    s.guidance_scale = job.guidance;
    m_sd.set_guidance_scale(pipe, job.guidance);
  }
  if(job.delta != s.delta)
  {
    s.delta = job.delta;
    m_sd.set_delta(pipe, job.delta);
  }
  if(job.lora_scale != s.lora_scale)
  {
    // Uniform across the runtime-LoRA slots; a no-op for engines without a lora_scale input.
    const int slots = m_sd.num_runtime_loras(pipe);
    for(int i = 0; i < slots; ++i)
      m_sd.set_lora_scale(pipe, i, job.lora_scale);
    s.lora_scale = job.lora_scale;
  }

  const auto err = job.ref_rgba.empty()
                       ? m_sd.txt2img(pipe, out_rgba, job.w, job.h)
                       : m_sd.img2img(pipe, job.ref_rgba.data(), out_rgba, job.w, job.h);
  if(err != LIBREDIFFUSION_SUCCESS)
    std::fprintf(
        stderr, "StreamDiffusion: %s failed (%d)\n", job.ref_rgba.empty() ? "txt2img" : "img2img",
        (int)err);
  return err == LIBREDIFFUSION_SUCCESS;
}

bool StreamDiffusion::renderKlein(const AsyncJob& job, unsigned char* out_rgba)
{
  if(!m_klein_stream)
    return false;
  // VAE-encode the reference only when it changed (txt2img's black reference: once).
  if(job.ref_hash != m_applied_ref_hash)
  {
    if(m_sd.flux2_stream_set_reference(m_klein_stream.get(), job.ref_rgba.data())
       != LIBREDIFFUSION_SUCCESS)
    {
      std::fprintf(stderr, "FLUX.2-klein: set_reference failed\n");
      return false;
    }
    m_applied_ref_hash = job.ref_hash;
  }
  const auto err = m_sd.flux2_stream_frame_cached(m_klein_stream.get(), out_rgba);
  if(err != LIBREDIFFUSION_SUCCESS)
    std::fprintf(stderr, "FLUX.2-klein: frame failed (%d)\n", (int)err);
  return err == LIBREDIFFUSION_SUCCESS;
}

bool StreamDiffusion::renderTurbo(const AsyncJob& job, unsigned char* out_rgba)
{
  if(!m_i2it)
    return false;
  const size_t frame_bytes = (size_t)m_sd.img2img_turbo_frame_bytes(m_i2it.get());
  const size_t ehs_elems = (size_t)m_sd.img2img_turbo_ehs_elements(m_i2it.get());
  if(job.ref_rgba.size() < frame_bytes)
    return false;

  // The "Embedding" port overrides; otherwise the prompt's CLIP embedding on the device.
  librediffusion_error_t err;
  if(job.ehs.size() >= ehs_elems)
    err = m_sd.img2img_turbo_frame_sized(
        m_i2it.get(), job.ref_rgba.data(), job.ref_rgba.size(), job.ehs.data(), job.ehs.size(),
        out_rgba, frame_bytes);
  else if(m_i2it_embeddings)
    err = m_sd.img2img_turbo_frame_dev_sized(
        m_i2it.get(), job.ref_rgba.data(), job.ref_rgba.size(), m_i2it_embeddings.embeddings,
        out_rgba, frame_bytes);
  else
    return false;  // nothing to translate without a text embedding
  if(err != LIBREDIFFUSION_SUCCESS)
    std::fprintf(stderr, "img2img-turbo: frame failed (%d)\n", (int)err);
  return err == LIBREDIFFUSION_SUCCESS;
}

// RIFE the sweep prev -> cur into out.sweep. The engine is loaded once per model/device; a bundle
// without one (or an engine that cannot run this geometry) leaves the keyframe alone. A plan that
// appears later (e.g. built by the exporter) is picked up: the lookup is a stat per keyframe.
void StreamDiffusion::interpolate(const AsyncJob& job, AsyncFrame& out)
{
  if(!m_rife)
  {
    const std::string path = rife_engine_path(m_model_dir);
    if(path.empty() || path == m_rife_failed_path)
      return;
    m_rife = SDRife{path.c_str(), m_device};
    if(!m_rife)
    {
      m_rife_failed_path = path;
      std::fprintf(stderr, "StreamDiffusion: %s could not be loaded; interpolation off\n", path.c_str());
      return;
    }
    m_sd.rife_set_enabled(m_rife.get(), 1);
  }

  m_sd.rife_set_interpolation_exp(m_rife.get(), job.exp);
  const size_t needed = m_sd.rife_required_out_bytes(m_rife.get(), job.h, job.w);
  out.sweep.assign(needed, 0);
  int n = 0;
  if(m_sd.rife_interpolate_sized(
         m_rife.get(), m_prev_key.data(), out.rgba.data(), job.h, job.w, out.sweep.data(),
         out.sweep.size(), &n)
         == LIBREDIFFUSION_SUCCESS
     && n > 0)
  {
    out.sweep_n = n;
  }
  else
  {
    out.sweep.clear();
    out.sweep_n = 0;
  }
}

// -------------------------------------------------------------------------------------------------
// SD family configuration
// -------------------------------------------------------------------------------------------------
bool StreamDiffusion::configureSD(const inputs_t& in)
{
  const auto new_t1 = get_steps(in.t1.value);
  if(!new_t1 || new_t1->empty())
  {
    std::fprintf(
        stderr, "StreamDiffusion: Timesteps \"%s\" %s\n", in.t1.value.c_str(),
        new_t1 ? "contains no step" : "could not be parsed");
    return false;
  }
  if(in.prompt.value.empty())
    return false;

  const inputs_t& prev = m_prev_inputs;
  const auto prev_t1 = get_steps(prev.t1.value);
  const std::size_t n_prev_t1 = prev_t1 ? prev_t1->size() : 0;

  // State-based, not only input-based: a failed switch to another family (nothing is recorded in
  // m_prev_inputs on failure) must not leave a pipeline without geometry.
  const bool need_rebuild
      = !m_cached_engine || !m_cached_engine->pipeline || m_w == 0
        || m_model_dir != in.model.value || n_prev_t1 != new_t1->size()
        || prev.add_noise.value != in.add_noise.value
        || prev.denoise_batch.value != in.denoise_batch.value
        || prev.workflow.value != in.workflow.value
        || prev.size.value.x != in.size.value.x || prev.size.value.y != in.size.value.y
        || prev.cfg.value != in.cfg.value
        || std::signbit(prev.guidance.value - 1.f) != std::signbit(in.guidance.value - 1.f);
  const bool need_scheduler = need_rebuild || prev.t1.value != in.t1.value;
  const bool need_positive
      = need_rebuild || prev.prompt.value != in.prompt.value || m_embeddings.empty();
  const bool need_negative = need_rebuild || prev.negative_prompt.value != in.negative_prompt.value
                             || !m_negative_embeddings;

  // Every step below mutates the pipeline the producer may be driving. (Seed, guidance, delta and
  // LoRA scale travel in the job and are applied by the producer itself.)
  if(need_rebuild || need_scheduler || need_positive || need_negative)
    stopProducer();

  if(need_rebuild && !createSDPipeline(in, *new_t1))
    return false;
  auto pipe = m_cached_engine->pipeline->get();

  if(need_scheduler && !updateScheduler(in.t1.value))
    return false;
  if(need_positive && !updatePromptEmbeddings(in.prompt.value, m_embeddings))
  {
    std::fprintf(stderr, "StreamDiffusion: invalid prompt\n");
    return false;
  }
  if(need_negative)
  {
    if(!updatePromptEmbedding(in.negative_prompt.value, m_negative_embeddings, false)
       || m_sd.prepare_negative_embeds(
              pipe, m_negative_embeddings.embeddings, m_config_state.text_seq_len,
              m_config_state.text_hidden_dim)
              != LIBREDIFFUSION_SUCCESS)
    {
      std::fprintf(stderr, "StreamDiffusion: invalid negative prompt\n");
      return false;
    }
  }
  return true;
}

bool StreamDiffusion::createSDPipeline(const inputs_t& in, std::vector<int> timestep_indices)
{
  int width = 0, height = 0;
  if(!resolveResolution(in, 8, width, height))
    return false;

  librediffusion_model_type_t model_type = MODEL_SD_15;
  librediffusion_pipeline_mode_t pipeline_mode = MODE_SINGLE_FRAME;
  switch(in.workflow.value)
  {
    case SDTURBO_TXT2IMG:
    case SDTURBO_IMG2IMG:
      model_type = MODEL_SD_TURBO;
      break;
    case SDXL_TXT2IMG:
    case SDXL_IMG2IMG:
    case SDXL_TXT2IMG_CONTROLNET:
    case SDXL_IMG2IMG_CONTROLNET:
      model_type = MODEL_SDXL_TURBO;
      break;
    case V2V_TXT2IMG:
    case V2V_IMG2IMG:
      pipeline_mode = MODE_TEMPORAL_V2V;
      break;
    default:
      break;
  }

  const std::string& model = in.model.value;
  if(m_cached_engine
     && (m_cached_engine->model_path != model || m_cached_engine->pipeline_mode != pipeline_mode))
  {
    EngineCache::instance().release(m_cached_engine);
    m_cached_engine = nullptr;
  }
  if(!m_cached_engine)
  {
    m_cached_engine = EngineCache::instance().acquire(model, pipeline_mode, m_device);
    if(!m_cached_engine)
    {
      auto engine = std::make_unique<CachedEngine>();
      engine->model_path = model;
      engine->pipeline_mode = pipeline_mode;
      engine->device = m_device;
      engine->clip1 = new SDClip{(model + "/clip.engine").c_str(), m_device};
      if(!*engine->clip1)
        return false;
      m_cached_engine = EngineCache::instance().store(std::move(engine));
    }
  }
  // SD / SDXL / ControlNet / IP-Adapter share MODE_SINGLE_FRAME, so an entry created by an SD
  // workflow lacks the clip2 an SDXL workflow needs; build it on demand.
  if(model_type == MODEL_SDXL_TURBO && !m_cached_engine->clip2)
  {
    auto* clip2 = new SDClip{(model + "/clip2.engine").c_str(), m_device};
    if(!*clip2)
    {
      std::fprintf(stderr, "StreamDiffusion: SDXL workflow but %s/clip2.engine could not be loaded\n",
                   model.c_str());
      delete clip2;
      return false;
    }
    m_cached_engine->clip2 = clip2;
  }

  SDConfigState& s = m_config_state;
  s.model_type = model_type;
  s.pipeline_mode = pipeline_mode;
  s.width = width;
  s.height = height;
  s.batch_size = 1;
  s.timestep_indices = std::move(timestep_indices);
  s.denoising_steps = (int)s.timestep_indices.size();
  s.controlnet_index = -1;
  s.controlnet_scale = in.controlnet_scale.value;
  s.ipadapter_enabled = isIPAdapter(in.workflow.value);
  s.ipadapter_scale = in.ipadapter_scale.value;
  s.lora_scale = -1.f;  // pushed again by the first frame: a new engine may have LoRA slots
  s.seeded = false;
  s.ipadapter_tokens = false;
  s.do_add_noise = in.add_noise.value;
  s.delta = in.delta.value;
  s.text_seq_len = 77;
  s.pooled_embedding_dim = 1280;
  switch(model_type)
  {
    case MODEL_SD_TURBO:  // genuinely single-step
      s.use_denoising_batch = false;
      s.denoising_steps = 1;
      s.cfg_type = SD_CFG_NONE;
      s.guidance_scale = 0.0f;
      s.text_hidden_dim = 1024;
      s.clip_pad_token = 0;
      break;
    case MODEL_SDXL_TURBO:
      // The whole SDXL family (Hyper-SDXL / Lightning / LCM-LoRA / VegaRT are multi-step); the
      // engine's batch profile constrains the step count.
      s.use_denoising_batch = in.denoise_batch.value;
      s.cfg_type = SD_CFG_NONE;
      s.guidance_scale = 0.0f;
      s.text_hidden_dim = 2048;
      s.clip_pad_token = 0;
      break;
    default:
      s.use_denoising_batch = in.denoise_batch.value;
      switch(in.cfg.value)
      {
        case None:
          s.cfg_type = SD_CFG_NONE;
          break;
        case Full:
          s.cfg_type = SD_CFG_FULL;
          break;
        case Self:
          s.cfg_type = SD_CFG_SELF;
          break;
        case Initialize:
          s.cfg_type = SD_CFG_INITIALIZE;
          break;
      }
      s.guidance_scale = in.guidance.value;
      s.text_hidden_dim = 768;
      s.clip_pad_token = 49407;
      // SD2.1 shares this path but encodes to 1024: take the manifest's width when it declares one.
      if(const int dim = bundle_embedding_dim(model); dim > 0)
        s.text_hidden_dim = dim;
      break;
  }

  SDConfig config;
  if(!config)
    return false;
  // The validating setters keep the previous value on rejection while everything downstream
  // reports success, so the pipeline would run at a geometry nobody asked for. Refuse instead.
  auto accepted = [](librediffusion_error_t err, const char* what) {
    if(err == LIBREDIFFUSION_SUCCESS)
      return true;
    std::fprintf(stderr, "StreamDiffusion: %s rejected the request (%d)\n", what, (int)err);
    return false;
  };
  m_sd.config_set_device(config.get(), m_device);
  m_sd.config_set_model_type(config.get(), model_type);
  m_sd.config_set_pipeline_mode(config.get(), pipeline_mode);
  if(!accepted(
         m_sd.config_set_dimensions(config.get(), width, height, width / 8, height / 8),
         "config_set_dimensions"))
    return false;
  if(!accepted(m_sd.config_set_batch_size(config.get(), s.batch_size), "config_set_batch_size"))
    return false;
  if(!accepted(
         m_sd.config_set_denoising_steps(config.get(), s.denoising_steps),
         "config_set_denoising_steps"))
    return false;
  m_sd.config_set_guidance_scale(config.get(), s.guidance_scale);
  m_sd.config_set_delta(config.get(), s.delta);
  m_sd.config_set_add_noise(config.get(), s.do_add_noise ? 1 : 0);
  m_sd.config_set_denoising_batch(config.get(), s.use_denoising_batch ? 1 : 0);
  m_sd.config_set_cfg_type(config.get(), static_cast<librediffusion_cfg_type_t>(s.cfg_type));
  // CUDA graph: capturable only for one step, cfg-none, non-V2V (the library re-gates identically).
  m_sd.config_set_cuda_graph(
      config.get(),
      (s.denoising_steps == 1 && s.cfg_type == SD_CFG_NONE && pipeline_mode != MODE_TEMPORAL_V2V)
          ? 1
          : 0);
  if(!accepted(
         m_sd.config_set_text_config(
             config.get(), s.text_seq_len, s.text_hidden_dim, s.clip_pad_token),
         "config_set_text_config"))
    return false;
  if(model_type == MODEL_SDXL_TURBO
     && !accepted(
         m_sd.config_set_sdxl_config(config.get(), s.pooled_embedding_dim, 6),
         "config_set_sdxl_config"))
    return false;

  m_sd.config_set_unet_engine(config.get(), (model + "/unet.engine").c_str());
  m_sd.config_set_vae_encoder(config.get(), (model + "/vae_encoder.engine").c_str());
  m_sd.config_set_vae_decoder(config.get(), (model + "/vae_decoder.engine").c_str());

  if(isControlNet(in.workflow.value))
  {
    const std::string controlnet = model + "/controlnet.engine";
    s.controlnet_index
        = m_sd.config_add_controlnet(config.get(), controlnet.c_str(), s.controlnet_scale);
    if(s.controlnet_index < 0)
    {
      std::fprintf(
          stderr, "StreamDiffusion: config_add_controlnet failed (missing %s?)\n",
          controlnet.c_str());
      return false;
    }
  }
  if(s.ipadapter_enabled)
  {
    // SD1.5 base IP-Adapter: 4 tokens. The image encoder engines let the node turn the raw style
    // texture into tokens on-device.
    m_sd.config_set_ipadapter(config.get(), 4, s.ipadapter_scale);
    m_sd.config_set_ipadapter_image_encoder(
        config.get(), (model + "/clip_image_encoder.engine").c_str(),
        (model + "/ip_image_proj.engine").c_str());
  }
  m_sd.config_set_timestep_indices(
      config.get(), s.timestep_indices.data(), s.timestep_indices.size());

  // StreamV2V: kvo_cache extended self-attention, banking the previous 2 frames (profile max 4).
  // Feature injection / similarity are inert for the extended-attention engines.
  if(pipeline_mode == MODE_TEMPORAL_V2V)
    m_sd.config_set_temporal_params(config.get(), 1, in.add_noise.value ? 1 : 0, 0.8f, 0.78f, 1, 2);

  // ControlNet / IP-Adapter engines are loaded only at pipeline creation, not at reinit_buffers, so
  // a pipeline that has them, or needs them, cannot be reused across such a change.
  const bool wants_features = isControlNet(in.workflow.value) || isIPAdapter(in.workflow.value);
  if((wants_features || m_cached_engine->has_features) && m_cached_engine->pipeline)
  {
    delete m_cached_engine->pipeline;
    m_cached_engine->pipeline = nullptr;
  }
  if(m_cached_engine->pipeline && *m_cached_engine->pipeline)
  {
    // The engines cannot be reloaded here, so a geometry outside their profiles is refused.
    if(!accepted(
           m_sd.pipeline_reinit_buffers(m_cached_engine->pipeline->get(), config.get()),
           "pipeline_reinit_buffers"))
      return false;
  }
  else
  {
    delete m_cached_engine->pipeline;
    m_cached_engine->pipeline = new SDPipeline{config.get()};
    if(!*m_cached_engine->pipeline)
      return false;
    m_cached_engine->has_features = wants_features;
  }

  m_model_dir = model;
  m_w = width;
  m_h = height;
  m_continuous = s.use_denoising_batch || pipeline_mode == MODE_TEMPORAL_V2V;
  m_rife.reset();
  m_rife_failed_path.clear();
  m_embeddings.clear();
  m_negative_embeddings.reset();
  return true;
}

bool StreamDiffusion::updatePromptEmbedding(
    const std::string& prompt, SDXLEmbeddings& embeddings, bool positive)
{
  // The CLIP calls overwrite the device pointers with a fresh allocation; release what is held.
  embeddings.reset();
  auto pipe = m_cached_engine->pipeline->get();
  if(m_config_state.model_type == MODEL_SDXL_TURBO)
  {
    if(m_sd.clip_compute_embeddings_sdxl(
           m_cached_engine->clip1->get(), m_cached_engine->clip2->get(), prompt.c_str(),
           m_config_state.batch_size, m_config_state.height, m_config_state.width, nullptr,
           &embeddings.embeddings, &embeddings.pooled_embeds, &embeddings.time_ids)
       != LIBREDIFFUSION_SUCCESS)
      return false;
    // The pooled conditioning is the positive prompt's; the negative one only feeds CFG.
    return !positive
           || m_sd.prepare_sdxl_conditioning(pipe, embeddings.pooled_embeds, embeddings.time_ids)
                  == LIBREDIFFUSION_SUCCESS;
  }
  return m_sd.clip_compute_embeddings(
             m_cached_engine->clip1->get(), prompt.c_str(), m_config_state.clip_pad_token,
             nullptr, &embeddings.embeddings)
         == LIBREDIFFUSION_SUCCESS;
}

bool StreamDiffusion::updatePromptEmbeddings(
    const std::string& prompt, std::vector<SDXLEmbeddings>& embeddings)
{
  embeddings.clear();
  auto pipe = m_cached_engine->pipeline->get();
  if(auto weights = parse_input_string(prompt))
  {
    boost::container::small_vector<float, 8> bweight;
    boost::container::small_vector<librediffusion_half_t*, 8> bembeds;
    for(const auto& [text, weight] : *weights)
    {
      SDXLEmbeddings e;
      // blend_embeds does not null-check: a null device pointer there kills the CUDA context.
      if(!updatePromptEmbedding(text, e, true) || !e.embeddings)
      {
        embeddings.clear();
        return false;
      }
      bembeds.push_back(e.embeddings);
      embeddings.push_back(std::move(e));
      bweight.push_back(weight);
    }
    if(bembeds.empty())
      return false;
    return m_sd.blend_embeds(
               pipe, bembeds.data(), bweight.data(), bembeds.size(), m_config_state.text_seq_len,
               m_config_state.text_hidden_dim)
           == LIBREDIFFUSION_SUCCESS;
  }

  SDXLEmbeddings e;
  if(!updatePromptEmbedding(prompt, e, true) || !e.embeddings)
    return false;
  embeddings.push_back(std::move(e));
  return m_sd.prepare_embeds(
             pipe, embeddings.front().embeddings, m_config_state.text_seq_len,
             m_config_state.text_hidden_dim)
         == LIBREDIFFUSION_SUCCESS;
}

bool StreamDiffusion::updateScheduler(const std::string& timestep_str)
{
  auto timestep_indices = get_steps(timestep_str);
  if(!timestep_indices || timestep_indices->empty())
    return false;
  if(m_config_state.model_type == MODEL_SD_TURBO)  // genuinely single-step
    timestep_indices->resize(1);
  m_config_state.timestep_indices = std::move(*timestep_indices);
  m_config_state.denoising_steps = (int)m_config_state.timestep_indices.size();

  std::span<const int> table_timesteps;
  std::span<const streamdiffusion::TimestepParams> table_params;
  using namespace streamdiffusion;
  switch(m_config_state.model_type)
  {
    case MODEL_SD_TURBO:
      table_timesteps = SCHEDULER_STABILITYAI_SD_TURBO::TIMESTEP_VALUES;
      table_params = SCHEDULER_STABILITYAI_SD_TURBO::TIMESTEP_PARAMS;
      break;
    case MODEL_SDXL_TURBO:
      table_timesteps = SCHEDULER_STABILITYAI_SDXL_TURBO::TIMESTEP_VALUES;
      table_params = SCHEDULER_STABILITYAI_SDXL_TURBO::TIMESTEP_PARAMS;
      break;
    default:
      table_timesteps = SCHEDULER_SIMIANLUO_LCM_DREAMSHAPER_V7::TIMESTEP_VALUES;
      table_params = SCHEDULER_SIMIANLUO_LCM_DREAMSHAPER_V7::TIMESTEP_PARAMS;
      break;
  }

  std::vector<float> timesteps, alpha, beta, c_skip, c_out;
  for(int idx : m_config_state.timestep_indices)
  {
    if(idx < 0 || idx >= std::ssize(table_params) || idx >= std::ssize(table_timesteps))
      continue;
    timesteps.push_back(static_cast<float>(table_timesteps[idx]));
    alpha.push_back(table_params[idx].alpha_prod_t_sqrt);
    beta.push_back(table_params[idx].beta_prod_t_sqrt);
    c_skip.push_back(table_params[idx].c_skip);
    c_out.push_back(table_params[idx].c_out);
  }
  if(timesteps.empty())
    return false;

  const auto err = m_sd.prepare_scheduler(
      m_cached_engine->pipeline->get(), timesteps.data(), alpha.data(), beta.data(), c_skip.data(),
      c_out.data(), timesteps.size());
  if(err != LIBREDIFFUSION_SUCCESS)
    std::fprintf(stderr, "StreamDiffusion: prepare_scheduler failed (%d)\n", (int)err);
  return err == LIBREDIFFUSION_SUCCESS;
}

// -------------------------------------------------------------------------------------------------
// FLUX.2-klein configuration
// -------------------------------------------------------------------------------------------------
// FluxRT's fixed-seed convention: 0 -> 52.
static unsigned long long klein_seed(int seed) noexcept
{
  const auto s = static_cast<unsigned long long>(static_cast<uint32_t>(seed));
  return s == 0 ? 52ull : s;
}

bool StreamDiffusion::configureKlein(const inputs_t& in)
{
  int w = 0, h = 0;
  if(!resolveResolution(in, 16, w, h))
    return false;
  if(in.prompt.value.empty())
    return false;

  // The seed is baked in at creation and the engines come back from the library's own cache.
  if(!m_klein_stream || m_model_dir != in.model.value
     || m_klein_quality != in.klein_quality.value || m_w != w || m_h != h
     || m_klein_seed != klein_seed(in.seed.value))
  {
    if(!createKleinStream(in))
      return false;
  }
  auto stream = m_klein_stream.get();

  // set_prompt runs the Qwen encoder, set_schedule / set_mask mutate what the producer reads.
  if(m_klein_prompt != in.prompt.value)
  {
    stopProducer();
    if(m_sd.flux2_stream_set_prompt(stream, in.prompt.value.c_str()) < 0)
    {
      std::fprintf(stderr, "FLUX.2-klein: set_prompt failed\n");
      return false;
    }
    m_klein_prompt = in.prompt.value;
  }

  // Timesteps -> FlowMatch sigma schedule (list length = steps, first value = start noise level);
  // anything that is not a sigma list means the natural 2-step schedule.
  if(m_klein_sched != in.t1.value)
  {
    stopProducer();
    const auto sigmas = get_sigmas(in.t1.value);
    if(!sigmas.empty())
      m_sd.flux2_stream_set_schedule(stream, sigmas.data(), (int)sigmas.size());
    else
      m_sd.flux2_stream_set_steps(stream, 2);
    m_klein_sched = in.t1.value;
  }

  // Inpaint: the "Control / Style" texture is the mask (white = regenerate, black = keep).
  const auto& mt = in.control.texture;
  const bool inpaint = in.workflow.value == FLUX2_KLEIN_INPAINT && mt.bytes && mt.width > 0
                       && mt.height > 0;
  const uint64_t mask_hash = inpaint ? hash_bytes(mt.bytes, (size_t)mt.width * mt.height * 4) : 0;
  if(mask_hash != m_klein_mask_hash)
  {
    stopProducer();
    if(inpaint)
    {
      rgba_image mask(mt.bytes, mt.width, mt.height);
      if(mt.width != m_w || mt.height != m_h)
        mask = mask.scaled({m_w, m_h});
      m_sd.flux2_stream_set_mask(stream, mask.constBits(), m_h, m_w);
    }
    else
      m_sd.flux2_stream_set_mask(stream, nullptr, 0, 0);
    m_klein_mask_hash = mask_hash;
  }
  return true;
}

bool StreamDiffusion::createKleinStream(const inputs_t& in)
{
  int w = 0, h = 0;
  if(!resolveResolution(in, 16, w, h))
    return false;
  const std::string& model = in.model.value;

  // VAE batch-norm constants (128 fp32 each), vendored into every bundle by the exporter.
  std::array<float, 128> bn_mean{}, bn_std{};
  if(!read_bn_file(model + "/bn_mean.bin", bn_mean) || !read_bn_file(model + "/bn_std.bin", bn_std))
  {
    std::fprintf(stderr, "FLUX.2-klein: %s lacks bn_mean.bin / bn_std.bin\n", model.c_str());
    return false;
  }

  // The producer holds the old stream's handle: drain + join before it is freed, and free it
  // before the new one is created so both never share the VRAM.
  stopProducer();
  m_klein_stream.reset();
  const std::string transformer
      = model
        + (in.klein_quality.value == Speed ? "/transformer_fp8_calib.plan"
                                           : "/transformer_bf16.plan");
  m_klein_stream = SDFluxStream{
      transformer.c_str(), (model + "/qwen3_encoder_bf16.plan").c_str(),
      (model + "/vae_decoder_bf16.plan").c_str(), (model + "/vae_encoder_bf16.plan").c_str(),
      (model + "/tokenizer.json").c_str(), h / 16, w / 16, klein_seed(in.seed.value), m_device};
  if(!m_klein_stream)
  {
    std::fprintf(stderr, "FLUX.2-klein: failed to create the stream pipeline\n");
    return false;
  }
  m_sd.flux2_stream_set_steps(m_klein_stream.get(), 2);
  m_sd.flux2_stream_set_bn(m_klein_stream.get(), bn_mean.data(), bn_std.data());

  m_model_dir = model;
  m_w = w;
  m_h = h;
  m_klein_quality = in.klein_quality.value;
  m_klein_seed = klein_seed(in.seed.value);
  m_klein_prompt.clear();
  m_klein_sched.clear();
  m_klein_mask_hash = 0;
  m_continuous = false;  // fixed-seed noise + cached reference: identical inputs, identical frame
  m_rife.reset();
  m_rife_failed_path.clear();
  return true;
}

// -------------------------------------------------------------------------------------------------
// img2img-turbo configuration
// -------------------------------------------------------------------------------------------------
bool StreamDiffusion::configureTurbo(const inputs_t& in)
{
  const std::string& model = in.model.value;
  if(!m_i2it || m_model_dir != model)
  {
    stopProducer();
    m_i2it.reset();
    m_i2it = SDImg2ImgTurbo{
        (model + "/unet.engine").c_str(), (model + "/vae_encoder.engine").c_str(),
        (model + "/vae_decoder.engine").c_str(), m_device};
    if(!m_i2it)
    {
      std::fprintf(stderr, "img2img-turbo: create failed for %s\n", model.c_str());
      return false;
    }
    // The geometry is the engines' own, not the Resolution port's.
    int w = 0, h = 0;
    if(m_sd.img2img_turbo_frame_size(m_i2it.get(), &w, &h) != LIBREDIFFUSION_SUCCESS || w <= 0
       || h <= 0)
    {
      std::fprintf(stderr, "img2img-turbo: engine reports an unusable geometry %dx%d\n", w, h);
      m_i2it.reset();
      return false;
    }
    m_w = w;
    m_h = h;
    m_model_dir = model;
    m_continuous = false;
    m_rife.reset();
    m_rife_failed_path.clear();
    // Prompt path: sd-turbo CLIP (1024-dim, pad 0); optional, the Embedding port overrides it.
    m_i2it_clip = SDClip{(model + "/clip.engine").c_str(), m_device};
    m_i2it_embeddings.reset();
    m_i2it_prompt.clear();
  }

  if(m_i2it_clip && m_i2it_prompt != in.prompt.value)
  {
    stopProducer();
    m_i2it_embeddings.reset();
    if(m_sd.clip_compute_embeddings(
           m_i2it_clip.get(), in.prompt.value.c_str(), 0, nullptr, &m_i2it_embeddings.embeddings)
       != LIBREDIFFUSION_SUCCESS)
      m_i2it_embeddings.embeddings = nullptr;
    m_i2it_prompt = in.prompt.value;
  }
  return true;
}

}
