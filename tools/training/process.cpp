#include "training/process.h"

#include <chrono>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace yue2::training {

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

} // namespace

std::optional<std::string> capture_output(const std::filesystem::path & program, const std::vector<std::string> & arguments,
                                          int timeout_seconds) {
    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    HANDLE read_end = nullptr, write_end = nullptr;
    if (!CreatePipe(&read_end, &write_end, &inherit, 0)) return std::nullopt;
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);
    HANDLE null = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = INVALID_HANDLE_VALUE;
    startup.hStdOutput = write_end;
    startup.hStdError = null;
    auto command = quote(program.wstring());
    for (const auto & argument : arguments) command += L" " + quote(wide(argument));
    PROCESS_INFORMATION process{};
    const bool started = CreateProcessW(program.wstring().c_str(), command.data(), nullptr, nullptr, TRUE,
                                        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    CloseHandle(write_end);
    if (null != INVALID_HANDLE_VALUE) CloseHandle(null);
    if (!started) { CloseHandle(read_end); return std::nullopt; }
    std::string output;
    std::thread reader([&]() {
        char buffer[4096];
        DWORD got = 0;
        while (ReadFile(read_end, buffer, sizeof(buffer), &got, nullptr) && got) output.append(buffer, got);
    });
    const bool finished = WaitForSingleObject(process.hProcess, static_cast<DWORD>(timeout_seconds) * 1000) == WAIT_OBJECT_0;
    if (!finished) {
        TerminateProcess(process.hProcess, 1);
        WaitForSingleObject(process.hProcess, INFINITE);
    }
    reader.join();
    CloseHandle(read_end);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    if (!finished) return std::nullopt;
    return output;
}
#else
std::optional<std::string> capture_output(const std::filesystem::path & program, const std::vector<std::string> & arguments,
                                          int timeout_seconds) {
    int pipe_ends[2];
    if (pipe(pipe_ends) != 0) return std::nullopt;
    const pid_t child = fork();
    if (child < 0) { close(pipe_ends[0]); close(pipe_ends[1]); return std::nullopt; }
    if (child == 0) {
        dup2(pipe_ends[1], STDOUT_FILENO);
        const int null = open("/dev/null", O_RDWR);
        if (null >= 0) { dup2(null, STDERR_FILENO); dup2(null, STDIN_FILENO); }
        close(pipe_ends[0]);
        close(pipe_ends[1]);
        std::vector<char *> argv{const_cast<char *>(program.c_str())};
        for (const auto & argument : arguments) argv.push_back(const_cast<char *>(argument.c_str()));
        argv.push_back(nullptr);
        execv(program.c_str(), argv.data());
        _exit(127);
    }
    close(pipe_ends[1]);
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
    int status = 0;
    waitpid(child, &status, 0);
    if (timed_out || (WIFEXITED(status) && WEXITSTATUS(status) == 127)) return std::nullopt;
    return output;
}
#endif

} // namespace yue2::training
