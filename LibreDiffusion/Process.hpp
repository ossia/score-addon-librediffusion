#pragma once

// A child process whose stdout and stderr are appended to a log file. No shell is involved:
// arguments are passed as a vector and quoted by us. The child (and, on Windows, everything it
// spawns) is killed when the owner is destroyed, so a build can never outlive the host.

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lo
{
class Process
{
public:
  using Environment = std::vector<std::pair<std::string, std::string>>;

  Process() = default;
  ~Process();
  Process(Process&&) noexcept;
  Process& operator=(Process&&) noexcept;
  Process(const Process&) = delete;
  Process& operator=(const Process&) = delete;

  // argv[0] is the executable path. `env` is merged over the parent's environment.
  // On failure returns nullopt and describes why in `error`.
  static std::optional<Process> start(
      const std::vector<std::string>& argv, const Environment& env, const std::string& log_path,
      std::string& error);

  // Polls the child; false once it has exited.
  bool running();
  // The exit code, meaningful once running() returned false. -1 when killed by a signal.
  int exit_code() const noexcept { return m_exit_code; }
  // Asks the child to stop (SIGTERM / TerminateProcess through the job object).
  void terminate() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
  int m_exit_code{-1};
};
}
