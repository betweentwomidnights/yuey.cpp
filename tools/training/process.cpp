#include "training/process.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

namespace yue2::training {
namespace {

// Splits output into lines like Python's text mode: \r\n, \r and \n all end one.
class Lines {
public:
    explicit Lines(std::function<void(const std::string &)> on_line) : on_line_(std::move(on_line)) {}
    void feed(const char * data, std::size_t size) {
        for (std::size_t i = 0; i < size; ++i) {
            const char c = data[i];
            if (carriage_ && c == '\n') { carriage_ = false; continue; }
            carriage_ = c == '\r';
            if (c == '\r' || c == '\n') { emit(); continue; }
            pending_.push_back(c);
        }
    }
    void finish() { if (!pending_.empty()) emit(); }

private:
    void emit() {
        auto line = std::move(pending_);
        pending_.clear();
        on_line_(line);
    }
    std::function<void(const std::string &)> on_line_;
    std::string pending_;
    bool carriage_ = false;
};

// Forwards lines to the log, then to the caller; after a caller's exception
// it keeps logging and reports the exception at the end.
struct Forward {
    std::ofstream & log;
    const std::function<void(const std::string &)> & on_line;
    std::exception_ptr error;
    void operator()(const std::string & line) {
        log << line << '\n';
        log.flush();
        if (error) return;
        try { on_line(line); } catch (...) { error = std::current_exception(); }
    }
};

std::ofstream open_log(const std::filesystem::path & log, bool append) {
    if (log.has_parent_path()) std::filesystem::create_directories(log.parent_path());
    // Text mode, like Python's log handles: "\n" becomes "\r\n" on Windows.
    std::ofstream stream(log, append ? std::ios::app : std::ios::trunc);
    if (!stream) throw std::runtime_error("Cannot open log " + log.u8string());
    return stream;
}

} // namespace

#ifdef _WIN32
namespace {

std::wstring wide(const std::string & text) {
    if (text.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring output(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), output.data(), size);
    return output;
}

// Python's subprocess.list2cmdline: quote only an argument with a space or tab
// (or an empty one), since cmd.exe does not unquote its own switches.
// Backslashes are literal unless they precede a quote.
std::wstring quote(const std::wstring & value) {
    const bool quoted = value.empty() || value.find_first_of(L" \t") != std::wstring::npos;
    std::wstring output = quoted ? L"\"" : L"";
    std::size_t slashes = 0;
    for (const wchar_t c : value) {
        if (c == L'\\') { ++slashes; continue; }
        output.append(slashes * (c == L'"' ? 2 : 1), L'\\');
        slashes = 0;
        if (c == L'"') output += L'\\';
        output += c;
    }
    output.append(slashes * (quoted ? 2 : 1), L'\\');
    return quoted ? output + L"\"" : output;
}

// Starts program with the given inheritable standard output and error; the
// caller owns the returned process handle. Null when it cannot start.
HANDLE spawn(const std::filesystem::path & program, const std::vector<std::string> & arguments,
             const std::filesystem::path & cwd, HANDLE output, HANDLE error) {
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = INVALID_HANDLE_VALUE;
    startup.hStdOutput = output;
    startup.hStdError = error;
    auto command = quote(program.wstring());
    for (const auto & argument : arguments) command += L" " + quote(wide(argument));
    PROCESS_INFORMATION process{};
    const auto directory = cwd.empty() ? std::wstring() : cwd.wstring();
    if (!CreateProcessW(program.wstring().c_str(), command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                        directory.empty() ? nullptr : directory.c_str(), &startup, &process)) {
        return nullptr;
    }
    CloseHandle(process.hThread);
    return process.hProcess;
}

int exit_code(HANDLE process) {
    WaitForSingleObject(process, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process, &code);
    CloseHandle(process);
    return static_cast<int>(code);
}

} // namespace

std::optional<std::string> capture_output(const std::filesystem::path & program, const std::vector<std::string> & arguments,
                                          int timeout_seconds, int * exit_code_out) {
    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    HANDLE read_end = nullptr, write_end = nullptr;
    if (!CreatePipe(&read_end, &write_end, &inherit, 0)) return std::nullopt;
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);
    HANDLE null = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr);
    HANDLE process = spawn(program, arguments, {}, write_end, null);
    CloseHandle(write_end);
    if (null != INVALID_HANDLE_VALUE) CloseHandle(null);
    if (!process) { CloseHandle(read_end); return std::nullopt; }
    std::string output;
    std::thread reader([&]() {
        char buffer[4096];
        DWORD got = 0;
        while (ReadFile(read_end, buffer, sizeof(buffer), &got, nullptr) && got) output.append(buffer, got);
    });
    const bool finished = WaitForSingleObject(process, static_cast<DWORD>(timeout_seconds) * 1000) == WAIT_OBJECT_0;
    if (!finished) TerminateProcess(process, 1);
    const int code = exit_code(process);
    if (exit_code_out) *exit_code_out = code;
    reader.join();
    CloseHandle(read_end);
    if (!finished) return std::nullopt;
    return output;
}

int run_to_log(const std::filesystem::path & program, const std::vector<std::string> & arguments,
               const std::filesystem::path & cwd, const std::filesystem::path & log, bool append) {
    { open_log(log, append); }
    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    HANDLE file = CreateFileW(log.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open log " + log.u8string());
    HANDLE process = spawn(program, arguments, cwd, file, file);
    CloseHandle(file);
    if (!process) throw std::runtime_error("Cannot start " + program.u8string());
    return exit_code(process);
}

int run_streaming(const std::filesystem::path & program, const std::vector<std::string> & arguments,
                  const std::filesystem::path & cwd, const std::filesystem::path & log,
                  const std::function<void(const std::string &)> & on_line) {
    auto stream = open_log(log, true);
    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    HANDLE read_end = nullptr, write_end = nullptr;
    if (!CreatePipe(&read_end, &write_end, &inherit, 0)) throw std::runtime_error("Cannot create a pipe");
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);
    HANDLE process = spawn(program, arguments, cwd, write_end, write_end);
    CloseHandle(write_end);
    if (!process) { CloseHandle(read_end); throw std::runtime_error("Cannot start " + program.u8string()); }
    Forward forward{stream, on_line, nullptr};
    Lines lines(std::ref(forward));
    char buffer[4096];
    DWORD got = 0;
    while (ReadFile(read_end, buffer, sizeof(buffer), &got, nullptr) && got) lines.feed(buffer, got);
    lines.finish();
    CloseHandle(read_end);
    const int code = exit_code(process);
    if (forward.error) std::rethrow_exception(forward.error);
    return code;
}

void set_environment(const char * name, const char * value) { _putenv_s(name, value); }

std::filesystem::path current_executable() {
    std::wstring path(32768, L'\0');
    const auto size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    path.resize(size);
    return path;
}

std::string own_process_identity() {
    FILETIME created, exited, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) throw std::runtime_error("Cannot identify the training worker");
    return std::to_string((static_cast<unsigned long long>(created.dwHighDateTime) << 32) | created.dwLowDateTime);
}

void own_children() {
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) ||
        !AssignProcessToJobObject(job, GetCurrentProcess())) {
        throw std::runtime_error("Cannot establish training child-process ownership");
    }
    // The handle stays open until this process exits; children inherit the job.
}

std::optional<std::filesystem::path> find_program(const std::string & name) {
    const char * path = std::getenv("PATH");
    if (!path) return std::nullopt;
    std::string entries = path;
    for (std::size_t start = 0; start <= entries.size();) {
        const auto end = std::min(entries.find(';', start), entries.size());
        const auto folder = entries.substr(start, end - start);
        if (!folder.empty()) {
            const auto candidate = std::filesystem::u8path(folder) / (name + ".exe");
            std::error_code error;
            if (std::filesystem::is_regular_file(candidate, error)) return candidate;
        }
        start = end + 1;
    }
    return std::nullopt;
}

Child::Child(const std::filesystem::path & program, const std::vector<std::string> & arguments,
             const std::filesystem::path & cwd, const std::filesystem::path & log) {
    { open_log(log, false); }
    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    HANDLE file = CreateFileW(log.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, &inherit,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open log " + log.u8string());
    HANDLE process = spawn(program, arguments, cwd, file, file);
    CloseHandle(file);
    if (!process) throw std::runtime_error("Cannot start " + program.u8string());
    handle_ = process;
    pid_ = static_cast<long>(GetProcessId(process));
}

Child::~Child() {
    if (handle_) {
        if (!code_) WaitForSingleObject(handle_, INFINITE);
        CloseHandle(handle_);
    }
}

bool Child::running() {
    if (code_) return false;
    if (WaitForSingleObject(handle_, 0) != WAIT_OBJECT_0) return true;
    DWORD code = 1;
    GetExitCodeProcess(handle_, &code);
    code_ = static_cast<int>(code);
    return false;
}

int Child::wait() {
    if (!code_) {
        WaitForSingleObject(handle_, INFINITE);
        running();
    }
    return *code_;
}

void Child::terminate() { if (!code_) TerminateProcess(handle_, 1); }

std::optional<std::pair<double, double>> Child::peak_memory_gib() const {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(handle_, reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters), sizeof(counters))) return std::nullopt;
    return std::make_pair(counters.PeakWorkingSetSize / 1073741824.0, counters.PeakPagefileUsage / 1073741824.0);
}
#else
namespace {

// Starts program with standard output and error on output_fd; -1 when it cannot start.
pid_t spawn(const std::filesystem::path & program, const std::vector<std::string> & arguments,
            const std::filesystem::path & cwd, int output_fd, int error_fd, int * exec_error_fd) {
    int status_pipe[2];
    if (pipe(status_pipe) != 0) return -1;
    fcntl(status_pipe[1], F_SETFD, FD_CLOEXEC);
    const pid_t child = fork();
    if (child < 0) { close(status_pipe[0]); close(status_pipe[1]); return -1; }
    if (child == 0) {
        close(status_pipe[0]);
        const int null = open("/dev/null", O_RDWR);
        if (null >= 0) dup2(null, STDIN_FILENO);
        dup2(output_fd, STDOUT_FILENO);
        dup2(error_fd >= 0 ? error_fd : null, STDERR_FILENO);
        if (!cwd.empty() && chdir(cwd.c_str()) != 0) { const char failed = 1; (void)!write(status_pipe[1], &failed, 1); _exit(127); }
        std::vector<char *> argv{const_cast<char *>(program.c_str())};
        for (const auto & argument : arguments) argv.push_back(const_cast<char *>(argument.c_str()));
        argv.push_back(nullptr);
        execv(program.c_str(), argv.data());
        const char failed = 1;
        (void)!write(status_pipe[1], &failed, 1);
        _exit(127);
    }
    close(status_pipe[1]);
    *exec_error_fd = status_pipe[0];
    return child;
}

// True when the child reported that exec failed (the status pipe closes on success).
bool exec_failed(int status_fd) {
    char failed = 0;
    const auto got = read(status_fd, &failed, 1);
    close(status_fd);
    return got == 1;
}

int wait_exit(pid_t child) {
    int status = 0;
    waitpid(child, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
}

} // namespace

std::optional<std::string> capture_output(const std::filesystem::path & program, const std::vector<std::string> & arguments,
                                          int timeout_seconds, int * exit_code) {
    int pipe_ends[2];
    if (pipe(pipe_ends) != 0) return std::nullopt;
    int status_fd = -1;
    const pid_t child = spawn(program, arguments, {}, pipe_ends[1], -1, &status_fd);
    close(pipe_ends[1]);
    if (child < 0) { close(pipe_ends[0]); return std::nullopt; }
    if (exec_failed(status_fd)) { close(pipe_ends[0]); wait_exit(child); return std::nullopt; }
    std::string output;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    bool timed_out = false;
    for (;;) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) { timed_out = true; break; }
        pollfd watch{pipe_ends[0], POLLIN, 0};
        if (poll(&watch, 1, static_cast<int>(left)) <= 0) continue;
        char buffer[4096];
        const auto got = read(pipe_ends[0], buffer, sizeof(buffer));
        if (got <= 0) break;
        output.append(buffer, static_cast<std::size_t>(got));
    }
    close(pipe_ends[0]);
    if (timed_out) kill(child, SIGKILL);
    const int code = wait_exit(child);
    if (exit_code) *exit_code = code;
    if (timed_out) return std::nullopt;
    return output;
}

int run_to_log(const std::filesystem::path & program, const std::vector<std::string> & arguments,
               const std::filesystem::path & cwd, const std::filesystem::path & log, bool append) {
    { open_log(log, append); }
    const int file = open(log.c_str(), O_WRONLY | O_APPEND);
    if (file < 0) throw std::runtime_error("Cannot open log " + log.u8string());
    int status_fd = -1;
    const pid_t child = spawn(program, arguments, cwd, file, file, &status_fd);
    close(file);
    if (child < 0) throw std::runtime_error("Cannot start " + program.u8string());
    if (exec_failed(status_fd)) { wait_exit(child); throw std::runtime_error("Cannot start " + program.u8string()); }
    return wait_exit(child);
}

int run_streaming(const std::filesystem::path & program, const std::vector<std::string> & arguments,
                  const std::filesystem::path & cwd, const std::filesystem::path & log,
                  const std::function<void(const std::string &)> & on_line) {
    auto stream = open_log(log, true);
    int pipe_ends[2];
    if (pipe(pipe_ends) != 0) throw std::runtime_error("Cannot create a pipe");
    int status_fd = -1;
    const pid_t child = spawn(program, arguments, cwd, pipe_ends[1], pipe_ends[1], &status_fd);
    close(pipe_ends[1]);
    if (child < 0) { close(pipe_ends[0]); throw std::runtime_error("Cannot start " + program.u8string()); }
    if (exec_failed(status_fd)) {
        close(pipe_ends[0]);
        wait_exit(child);
        throw std::runtime_error("Cannot start " + program.u8string());
    }
    Forward forward{stream, on_line, nullptr};
    Lines lines(std::ref(forward));
    char buffer[4096];
    for (;;) {
        const auto got = read(pipe_ends[0], buffer, sizeof(buffer));
        if (got <= 0) break;
        lines.feed(buffer, static_cast<std::size_t>(got));
    }
    lines.finish();
    close(pipe_ends[0]);
    const int code = wait_exit(child);
    if (forward.error) std::rethrow_exception(forward.error);
    return code;
}

void set_environment(const char * name, const char * value) { setenv(name, value, 1); }

std::filesystem::path current_executable() {
#ifdef __APPLE__
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string path(size, '\0');
    if (_NSGetExecutablePath(path.data(), &size) != 0) throw std::runtime_error("Cannot find this executable");
    return std::filesystem::canonical(path.c_str());
#else
    return std::filesystem::read_symlink("/proc/self/exe");
#endif
}

std::string own_process_identity() {
    std::ifstream in("/proc/self/stat");
    std::string line;
    std::getline(in, line);
    const auto end = line.rfind(')');
    if (end == std::string::npos) throw std::runtime_error("Cannot identify the training worker");
    std::istringstream fields(line.substr(end + 2));
    std::string field;
    for (int i = 3; i <= 22; ++i) fields >> field;
    return field;
}

void own_children() {}

std::optional<std::filesystem::path> find_program(const std::string & name) {
    const char * path = std::getenv("PATH");
    if (!path) return std::nullopt;
    std::string entries = path;
    for (std::size_t start = 0; start <= entries.size();) {
        const auto end = std::min(entries.find(':', start), entries.size());
        const auto folder = entries.substr(start, end - start);
        if (!folder.empty()) {
            const auto candidate = std::filesystem::path(folder) / name;
            if (access(candidate.c_str(), X_OK) == 0) return candidate;
        }
        start = end + 1;
    }
    return std::nullopt;
}

Child::Child(const std::filesystem::path & program, const std::vector<std::string> & arguments,
             const std::filesystem::path & cwd, const std::filesystem::path & log) {
    { open_log(log, false); }
    const int file = open(log.c_str(), O_WRONLY | O_APPEND);
    if (file < 0) throw std::runtime_error("Cannot open log " + log.u8string());
    int status_fd = -1;
    const pid_t child = spawn(program, arguments, cwd, file, file, &status_fd);
    close(file);
    if (child < 0) throw std::runtime_error("Cannot start " + program.u8string());
    pid_ = static_cast<long>(child);
    if (exec_failed(status_fd)) { code_ = wait_exit(child); throw std::runtime_error("Cannot start " + program.u8string()); }
}

Child::~Child() { if (!code_ && pid_ > 0) wait(); }

bool Child::running() {
    if (code_) return false;
    int status = 0;
    if (waitpid(static_cast<pid_t>(pid_), &status, WNOHANG) == 0) return true;
    code_ = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    return false;
}

int Child::wait() {
    if (!code_) code_ = wait_exit(static_cast<pid_t>(pid_));
    return *code_;
}

void Child::terminate() { if (!code_) kill(static_cast<pid_t>(pid_), SIGTERM); }

std::optional<std::pair<double, double>> Child::peak_memory_gib() const { return std::nullopt; }
#endif

} // namespace yue2::training
