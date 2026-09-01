#pragma once

// Builds TensorRT engine bundles with the embedded exporter (uv + train-lora.py), out of process.
//
// The builder belongs to the process, not to a node: score destroys and re-creates the node on
// every transport start/stop, and a build takes minutes to hours. One build runs at a time; the
// child dies with the host. This code never touches CUDA or the librediffusion library.

#include "Process.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace lo
{
struct BuildRequest
{
  std::string python_cache;  // root for uv's cache, the managed Python, the venv and the exporter
  std::string build_folder;  // train-lora.py --output
  std::string options;       // the remaining train-lora.py arguments, shell-style
  int gpu{0};                // CUDA_VISIBLE_DEVICES for the child
};

enum class BuildState : int8_t
{
  Idle,
  Extracting,
  Running,
  Done,
  Failed
};

struct BuildStatus
{
  BuildState state{BuildState::Idle};
  std::string message;
};

class ModelBuilder
{
public:
  static ModelBuilder& instance();

  // Starts a build; false (with `error`) when one is already running or the request is unusable.
  bool start(BuildRequest request, std::string& error);
  BuildStatus status() const;

  // Where uv, its cache and the venv live for a given Python cache root.
  static std::string default_python_cache();
  static std::vector<std::string> tokenize(std::string_view options);
  static Process::Environment environment(const BuildRequest& request);

private:
  ModelBuilder() = default;
  ~ModelBuilder();

  void run(std::stop_token stop, BuildRequest request);
  void set_status(BuildState state, std::string message);
  static std::string extract(const std::string& python_cache);  // -> exporter directory

  mutable std::mutex m_mutex;
  BuildStatus m_status;
  std::jthread m_thread;
};
}
