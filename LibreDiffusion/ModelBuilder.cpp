#include "ModelBuilder.hpp"

#include "EmbeddedExporter.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace lo
{
namespace fs = std::filesystem;

namespace
{
#if defined(_WIN32)
constexpr const char* k_uv = "uv.exe";
#else
constexpr const char* k_uv = "uv";
#endif

std::string env_or(const char* name, std::string fallback)
{
  if(const char* v = std::getenv(name); v && *v)
    return v;
  return fallback;
}
}

ModelBuilder& ModelBuilder::instance()
{
  static ModelBuilder builder;
  return builder;
}

ModelBuilder::~ModelBuilder()
{
  if(m_thread.joinable())
  {
    m_thread.request_stop();
    m_thread.join();
  }
}

std::string ModelBuilder::default_python_cache()
{
  // Windows: the venv path is what overflows MAX_PATH once torch is in it, so stay very short.
#if defined(_WIN32)
  return env_or("SystemDrive", "C:") + "\\lrd";
#else
  return env_or("XDG_CACHE_HOME", env_or("HOME", "/tmp") + "/.cache") + "/librediffusion";
#endif
}

// Split on whitespace; single or double quotes group. No escape character, so Windows paths pass
// through untouched.
std::vector<std::string> ModelBuilder::tokenize(std::string_view options)
{
  std::vector<std::string> out;
  std::string cur;
  bool in_token = false;
  char quote = 0;
  for(const char c : options)
  {
    if(quote)
    {
      if(c == quote)
        quote = 0;
      else
        cur.push_back(c);
    }
    else if(c == '\'' || c == '"')
    {
      quote = c;
      in_token = true;
    }
    else if(c == ' ' || c == '\t' || c == '\n' || c == '\r')
    {
      if(in_token)
        out.push_back(std::move(cur));
      cur.clear();
      in_token = false;
    }
    else
    {
      cur.push_back(c);
      in_token = true;
    }
  }
  if(in_token)
    out.push_back(std::move(cur));
  return out;
}

// The child's CUDA_VISIBLE_DEVICES for the node's device ordinal. The ordinal is relative to the
// host's own CUDA_VISIBLE_DEVICES when one is set, so pick that entry rather than the number.
static std::string visible_device(int gpu)
{
  const char* host = std::getenv("CUDA_VISIBLE_DEVICES");
  if(!host || !*host)
    return std::to_string(gpu);
  std::string_view list{host};
  for(int i = 0; !list.empty(); ++i)
  {
    const auto comma = list.find(',');
    const auto entry = list.substr(0, comma);
    if(i == gpu)
      return std::string{entry};
    if(comma == std::string_view::npos)
      break;
    list.remove_prefix(comma + 1);
  }
  return std::to_string(gpu);
}

Process::Environment ModelBuilder::environment(const BuildRequest& r)
{
  const std::string cache = r.python_cache;
  const char sep = fs::path::preferred_separator;
  Process::Environment env{
      {"UV_CACHE_DIR", cache + sep + "uv"},
      {"UV_PYTHON_INSTALL_DIR", cache + sep + "py"},
      {"UV_PROJECT_ENVIRONMENT", cache + sep + "venv"},
      {"UV_LINK_MODE", "copy"},  // cache and venv may sit on different drives
      {"UV_PYTHON_DOWNLOADS", "automatic"},
      {"UV_NO_PROGRESS", "1"},
      {"CUDA_VISIBLE_DEVICES", visible_device(r.gpu)},
      {"PYTHONUTF8", "1"},
      {"PYTHONUNBUFFERED", "1"},
  };
  if(!std::getenv("HF_HOME"))
    env.emplace_back("HF_HOME", cache + sep + "hf");
  return env;
}

bool ModelBuilder::start(BuildRequest request, std::string& error)
{
  if(embedded_exporter_id().empty())
  {
    error = "this build of the plugin does not embed the exporter";
    return false;
  }
  if(request.build_folder.empty())
  {
    error = "no build folder";
    return false;
  }
  if(request.python_cache.empty())
    request.python_cache = default_python_cache();

  {
    std::lock_guard lock{m_mutex};
    if(m_status.state == BuildState::Extracting || m_status.state == BuildState::Running)
    {
      error = "a build is already running";
      return false;
    }
    m_status = {BuildState::Extracting, "extracting the exporter"};
  }
  // The previous build's thread has published its final state, so it is (about to be) done.
  if(m_thread.joinable())
    m_thread.join();
  m_thread = std::jthread(
      [this, request = std::move(request)](std::stop_token st) { run(st, request); });
  return true;
}

BuildStatus ModelBuilder::status() const
{
  std::lock_guard lock{m_mutex};
  return m_status;
}

void ModelBuilder::set_status(BuildState state, std::string message)
{
  std::lock_guard lock{m_mutex};
  m_status = {state, std::move(message)};
}

// Writes the embedded exporter under <python cache>/exporter-<id>/ once; a stamp file marks a
// complete extraction so a crash half-way is redone rather than trusted.
std::string ModelBuilder::extract(const std::string& python_cache)
{
  const fs::path dir = fs::path{python_cache} / ("exporter-" + std::string{embedded_exporter_id()});
  const fs::path stamp = dir / ".complete";
  if(fs::exists(stamp))
    return dir.string();

  fs::create_directories(dir);
  for(const EmbeddedFile& f : embedded_exporter_files())
  {
    const fs::path target = dir / fs::path{std::string{f.path}}.make_preferred();
    fs::create_directories(target.parent_path());
    std::ofstream out{target, std::ios::binary | std::ios::trunc};
    out.write(f.bytes.data(), (std::streamsize)f.bytes.size());
    if(!out)
      throw std::runtime_error("cannot write " + target.string());
    out.close();
    if(f.executable)
      fs::permissions(
          target, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
          fs::perm_options::add);
  }
  std::ofstream{stamp} << embedded_exporter_id();
  return dir.string();
}

void ModelBuilder::run(std::stop_token stop, BuildRequest request)
{
  try
  {
    const fs::path exporter = extract(request.python_cache);
    fs::create_directories(request.build_folder);
    const std::string log = (fs::path{request.build_folder} / "build.log").string();

    std::vector<std::string> argv{
        (exporter / k_uv).string(), "run", "--project", exporter.string(), "python",
        (exporter / "train-lora.py").string()};
    for(auto& t : tokenize(request.options))
      argv.push_back(std::move(t));
    argv.push_back("--output");
    argv.push_back(request.build_folder);

    std::string error;
    auto process = Process::start(argv, environment(request), log, error);
    if(!process)
    {
      set_status(BuildState::Failed, error);
      return;
    }
    set_status(BuildState::Running, "building into " + request.build_folder + " (log: " + log + ")");

    while(process->running())
    {
      if(stop.stop_requested())
      {
        process->terminate();
        set_status(BuildState::Failed, "cancelled");
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if(process->exit_code() == 0)
      set_status(BuildState::Done, request.build_folder);
    else
      set_status(
          BuildState::Failed,
          "exporter exited with " + std::to_string(process->exit_code()) + ", see " + log);
  }
  catch(const std::exception& e)
  {
    set_status(BuildState::Failed, e.what());
  }
}

}
