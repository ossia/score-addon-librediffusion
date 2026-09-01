#include "Process.hpp"

#include <algorithm>
#include <cstring>
#include <string_view>

#if defined(_WIN32)
#include <windows.h>

#include <cwctype>
#include <map>
#else
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#endif

namespace lo
{

#if defined(_WIN32)
namespace
{
std::wstring widen(const std::string& s)
{
  if(s.empty())
    return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
  return w;
}

// CommandLineToArgvW's inverse: quote an argument the way CRT-based programs parse it.
std::wstring quote_argument(const std::wstring& arg)
{
  if(!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos)
    return arg;
  std::wstring out = L"\"";
  for(auto it = arg.begin();; ++it)
  {
    std::size_t backslashes = 0;
    while(it != arg.end() && *it == L'\\')
    {
      ++it;
      ++backslashes;
    }
    if(it == arg.end())
    {
      out.append(backslashes * 2, L'\\');
      break;
    }
    if(*it == L'"')
      out.append(backslashes * 2 + 1, L'\\');
    else
      out.append(backslashes, L'\\');
    out.push_back(*it);
  }
  out.push_back(L'"');
  return out;
}

// The parent's environment with `env` merged in, as the NUL-separated block CreateProcess wants.
std::wstring environment_block(const Process::Environment& env)
{
  struct case_insensitive
  {
    bool operator()(const std::wstring& a, const std::wstring& b) const
    {
      return std::lexicographical_compare(
          a.begin(), a.end(), b.begin(), b.end(),
          [](wchar_t x, wchar_t y) { return std::towupper(x) < std::towupper(y); });
    }
  };
  std::map<std::wstring, std::wstring, case_insensitive> vars;
  if(wchar_t* block = GetEnvironmentStringsW())
  {
    for(wchar_t* p = block; *p;)
    {
      std::wstring entry{p};
      p += entry.size() + 1;
      const auto eq = entry.find(L'=', 1); // drive-letter entries start with '='
      if(eq != std::wstring::npos)
        vars[entry.substr(0, eq)] = entry.substr(eq + 1);
    }
    FreeEnvironmentStringsW(block);
  }
  for(const auto& [k, v] : env)
    vars[widen(k)] = widen(v);

  std::wstring out;
  for(const auto& [k, v] : vars)
  {
    out += k;
    out += L'=';
    out += v;
    out.push_back(L'\0');
  }
  out.push_back(L'\0');
  return out;
}
}

struct Process::Impl
{
  HANDLE process{nullptr};
  HANDLE job{nullptr};
  ~Impl()
  {
    if(job)
      CloseHandle(job); // KILL_ON_JOB_CLOSE ends whatever is still running
    if(process)
      CloseHandle(process);
  }
};

std::optional<Process> Process::start(
    const std::vector<std::string>& argv, const Environment& env, const std::string& log_path,
    std::string& error)
{
  if(argv.empty())
  {
    error = "no executable";
    return std::nullopt;
  }

  SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  HANDLE log = CreateFileW(
      widen(log_path).c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &inheritable,
      OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if(log == INVALID_HANDLE_VALUE)
  {
    error = "cannot open log file " + log_path;
    return std::nullopt;
  }
  HANDLE null_in = CreateFileW(
      L"NUL", GENERIC_READ, 0, &inheritable, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

  std::wstring cmdline;
  for(const auto& a : argv)
  {
    if(!cmdline.empty())
      cmdline += L' ';
    cmdline += quote_argument(widen(a));
  }
  std::wstring block = environment_block(env);

  // Inherit exactly the three standard handles: the host's other inheritable handles (files,
  // sockets, pipes) must not leak into a child that lives for hours, and a concurrent
  // CreateProcess elsewhere in the host must not pick up our log handle.
  HANDLE inherited[] = {log, null_in};
  SIZE_T attr_size = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
  std::vector<unsigned char> attr_storage(attr_size);
  auto* attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr_storage.data());
  InitializeProcThreadAttributeList(attrs, 1, 0, &attr_size);
  UpdateProcThreadAttribute(
      attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited, sizeof(inherited), nullptr, nullptr);

  STARTUPINFOEXW si{};
  si.StartupInfo.cb = sizeof(si);
  si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  si.StartupInfo.hStdInput = null_in;
  si.StartupInfo.hStdOutput = log;
  si.StartupInfo.hStdError = log;
  si.lpAttributeList = attrs;
  PROCESS_INFORMATION pi{};
  const BOOL ok = CreateProcessW(
      widen(argv[0]).c_str(), cmdline.data(), nullptr, nullptr, TRUE,
      CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT,
      block.data(), nullptr, &si.StartupInfo, &pi);
  const DWORD create_error = GetLastError();
  DeleteProcThreadAttributeList(attrs);
  CloseHandle(log);
  if(null_in != INVALID_HANDLE_VALUE)
    CloseHandle(null_in);
  if(!ok)
  {
    error = "CreateProcess failed (" + std::to_string(create_error) + ") for " + argv[0];
    return std::nullopt;
  }

  Process p;
  p.m_impl = std::make_unique<Impl>();
  p.m_impl->process = pi.hProcess;
  p.m_impl->job = CreateJobObjectW(nullptr, nullptr);
  if(p.m_impl->job)
  {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(
        p.m_impl->job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
    AssignProcessToJobObject(p.m_impl->job, pi.hProcess);
  }
  ResumeThread(pi.hThread);
  CloseHandle(pi.hThread);
  return p;
}

bool Process::running()
{
  if(!m_impl || !m_impl->process)
    return false;
  if(WaitForSingleObject(m_impl->process, 0) == WAIT_TIMEOUT)
    return true;
  DWORD code = 0;
  GetExitCodeProcess(m_impl->process, &code);
  m_exit_code = (int)code;
  CloseHandle(m_impl->process);
  m_impl->process = nullptr;
  return false;
}

void Process::terminate() noexcept
{
  if(m_impl && m_impl->job)
    TerminateJobObject(m_impl->job, 1);
  else if(m_impl && m_impl->process)
    TerminateProcess(m_impl->process, 1);
}

#else // POSIX

struct Process::Impl
{
  pid_t pid{-1};
};

std::optional<Process> Process::start(
    const std::vector<std::string>& argv, const Environment& env, const std::string& log_path,
    std::string& error)
{
  if(argv.empty())
  {
    error = "no executable";
    return std::nullopt;
  }

  std::vector<std::string> env_storage;
  for(char** e = environ; e && *e; ++e)
  {
    const std::string_view entry{*e};
    const auto key = entry.substr(0, entry.find('='));
    const bool overridden = std::any_of(
        env.begin(), env.end(), [&](const auto& kv) { return kv.first == key; });
    if(!overridden)
      env_storage.emplace_back(entry);
  }
  for(const auto& [k, v] : env)
    env_storage.push_back(k + "=" + v);

  std::vector<char*> c_argv, c_env;
  for(const auto& a : argv)
    c_argv.push_back(const_cast<char*>(a.c_str()));
  c_argv.push_back(nullptr);
  for(const auto& e : env_storage)
    c_env.push_back(const_cast<char*>(e.c_str()));
  c_env.push_back(nullptr);

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(
      &actions, 1, log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  posix_spawn_file_actions_adddup2(&actions, 1, 2);
  posix_spawnattr_t attr;
  posix_spawnattr_init(&attr);
  // Own process group, so terminate() reaches the children uv spawns as well.
  posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
  posix_spawnattr_setpgroup(&attr, 0);

  pid_t pid = -1;
  const int rc = posix_spawn(&pid, argv[0].c_str(), &actions, &attr, c_argv.data(), c_env.data());
  posix_spawn_file_actions_destroy(&actions);
  posix_spawnattr_destroy(&attr);
  if(rc != 0)
  {
    error = std::string{"posix_spawn failed: "} + std::strerror(rc) + " for " + argv[0];
    return std::nullopt;
  }
  Process p;
  p.m_impl = std::make_unique<Impl>();
  p.m_impl->pid = pid;
  return p;
}

bool Process::running()
{
  if(!m_impl || m_impl->pid <= 0)
    return false;
  int status = 0;
  const pid_t r = waitpid(m_impl->pid, &status, WNOHANG);
  if(r == 0)
    return true;
  m_exit_code = (r > 0 && WIFEXITED(status)) ? WEXITSTATUS(status) : -1;
  m_impl->pid = -1;
  return false;
}

void Process::terminate() noexcept
{
  if(m_impl && m_impl->pid > 0)
    kill(-m_impl->pid, SIGTERM);
}
#endif

Process::~Process()
{
  if(m_impl)
  {
    terminate();
#if !defined(_WIN32)
    if(m_impl->pid > 0)
    {
      kill(-m_impl->pid, SIGKILL);
      int status = 0;
      waitpid(m_impl->pid, &status, 0);
    }
#endif
  }
}

Process::Process(Process&&) noexcept = default;
Process& Process::operator=(Process&&) noexcept = default;

}
