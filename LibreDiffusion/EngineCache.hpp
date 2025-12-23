#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace lo
{
class SDPipeline;
class SDClip;

struct CachedEngine
{
  std::string model_path;
  int pipeline_mode{0};  // sd_pipeline_mode_t: 0=SINGLE_FRAME, 1=TEMPORAL_V2V
  bool denoising_batch{};
  SDPipeline* pipeline{};
  SDClip* clip1{};
  SDClip* clip2{};  // Only for SDXL
  bool in_use{false};

  CachedEngine();
  ~CachedEngine();
  CachedEngine(const CachedEngine&) = delete;
  CachedEngine& operator=(const CachedEngine&) = delete;
  CachedEngine(CachedEngine&&) noexcept;
  CachedEngine& operator=(CachedEngine&&) noexcept;
};

class EngineCache
{
public:
  static EngineCache& instance();

  CachedEngine* acquire(const std::string& model_path, int pipeline_mode);
  CachedEngine* store(std::unique_ptr<CachedEngine> engine);
  void release(CachedEngine* engine);
  void clear();

private:
  EngineCache() = default;
  ~EngineCache() = default;
  EngineCache(const EngineCache&) = delete;
  EngineCache& operator=(const EngineCache&) = delete;

  std::mutex m_mutex;
  std::vector<std::unique_ptr<CachedEngine>> m_engines;
};

}
