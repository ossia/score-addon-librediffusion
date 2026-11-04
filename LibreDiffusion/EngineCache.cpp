#include "EngineCache.hpp"
#include "LibreDiffusion.hpp"

#include <algorithm>
#include <cstdio>

namespace lo
{
CachedEngine::CachedEngine() = default;

CachedEngine::~CachedEngine()
{
  delete pipeline;
  delete clip1;
  delete clip2;
}

CachedEngine::CachedEngine(CachedEngine&& other) noexcept
    : model_path{std::move(other.model_path)}
    , pipeline_mode{other.pipeline_mode}
    , device{other.device}
    , pipeline{other.pipeline}
    , clip1{other.clip1}
    , clip2{other.clip2}
    , has_features{other.has_features}
    , in_use{other.in_use}
{
  other.pipeline = nullptr;
  other.clip1 = nullptr;
  other.clip2 = nullptr;
}

CachedEngine& CachedEngine::operator=(CachedEngine&& other) noexcept
{
  if (this != &other)
  {
    delete pipeline;
    delete clip1;
    delete clip2;

    model_path = std::move(other.model_path);
    pipeline_mode = other.pipeline_mode;
    device = other.device;
    pipeline = other.pipeline;
    clip1 = other.clip1;
    clip2 = other.clip2;
    has_features = other.has_features;
    in_use = other.in_use;

    other.pipeline = nullptr;
    other.clip1 = nullptr;
    other.clip2 = nullptr;
  }
  return *this;
}

EngineCache& EngineCache::instance()
{
  // Never destroyed. The first call happens before any CUDA call, so a function-local static
  // would be destroyed after the CUDA runtime's own exit-time teardown and its pipelines would
  // then release their streams and TensorRT contexts into a dead driver (segfault on quit). The
  // process exit reclaims the device memory anyway.
  static EngineCache* const cache = new EngineCache;
  return *cache;
}

CachedEngine* EngineCache::acquire(
    const std::string& model_path, int pipeline_mode, int device)
{
  std::lock_guard lock{m_mutex};

  for (auto& engine : m_engines)
  {
    if (!engine->in_use && engine->model_path == model_path
        && engine->pipeline_mode == pipeline_mode && engine->device == device)
    {
      engine->in_use = true;
      return engine.get();
    }
  }
  return nullptr;
}

CachedEngine* EngineCache::store(std::unique_ptr<CachedEngine> engine)
{
  std::lock_guard lock{m_mutex};

  engine->in_use = true;
  m_engines.push_back(std::move(engine));
  return m_engines.back().get();
}

void EngineCache::release(CachedEngine* engine)
{
  if (!engine)
    return;

  std::lock_guard lock{m_mutex};
  engine->in_use = false;
  // Most recently parked last.
  auto it = std::find_if(m_engines.begin(), m_engines.end(), [&](auto& e) { return e.get() == engine; });
  if (it != m_engines.end())
    std::rotate(it, it + 1, m_engines.end());
}

bool EngineCache::evict_parked(int device)
{
  std::unique_ptr<CachedEngine> dropped;
  {
    std::lock_guard lock{m_mutex};
    auto it = std::find_if(m_engines.begin(), m_engines.end(), [&](auto& e) {
      return !e->in_use && e->device == device;
    });
    if (it == m_engines.end())
      return false;
    dropped = std::move(*it);
    m_engines.erase(it);
  }
  std::fprintf(stderr, "StreamDiffusion: out of device memory, unloading %s\n",
               dropped->model_path.c_str());
  return true;
}

void EngineCache::clear()
{
  std::lock_guard lock{m_mutex};
  m_engines.clear();
}

}
