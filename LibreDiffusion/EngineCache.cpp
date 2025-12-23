#include "EngineCache.hpp"
#include "LibreDiffusion.hpp"
#include <QDebug>

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
    , pipeline{other.pipeline}
    , clip1{other.clip1}
    , clip2{other.clip2}
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
    pipeline = other.pipeline;
    clip1 = other.clip1;
    clip2 = other.clip2;
    in_use = other.in_use;

    other.pipeline = nullptr;
    other.clip1 = nullptr;
    other.clip2 = nullptr;
  }
  return *this;
}

EngineCache& EngineCache::instance()
{
  static EngineCache cache;
  return cache;
}

CachedEngine* EngineCache::acquire(const std::string& model_path, int pipeline_mode)
{
  std::lock_guard lock{m_mutex};

  for (auto& engine : m_engines)
  {
    if (!engine->in_use && engine->model_path == model_path
        && engine->pipeline_mode == pipeline_mode)
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
}

void EngineCache::clear()
{
  std::lock_guard lock{m_mutex};
  m_engines.clear();
}

}
