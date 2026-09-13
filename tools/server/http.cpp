#include "http.h"

#include "json.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cerrno>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using SocketHandle = SOCKET;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using SocketHandle = int;
constexpr SocketHandle kInvalidSocket = -1;
#endif

namespace yue2::server {
namespace {

std::string lower_ascii(std::string value) {
    for (auto & character : value) {
        character = static_cast<char>(
            std::tolower(static_cast<unsigned char>(character)));
    }
    return value;
}

std::string trim(std::string value) {
    const auto space = [](unsigned char character) {
        return std::isspace(character) != 0;
    };
    value.erase(value.begin(), std::find_if(
        value.begin(), value.end(), [&](char character) { return !space(character); }));
    value.erase(std::find_if(
        value.rbegin(), value.rend(), [&](char character) { return !space(character); }).base(),
        value.end());
    return value;
}

const char * status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 409: return "Conflict";
        case 413: return "Payload Too Large";
        case 415: return "Unsupported Media Type";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "Error";
    }
}

class SocketRuntime {
public:
    SocketRuntime() {
#ifdef _WIN32
        WSADATA data{};
        const auto result = WSAStartup(MAKEWORD(2, 2), &data);
        if (result != 0) throw std::runtime_error("WSAStartup failed: " + std::to_string(result));
#endif
    }
    ~SocketRuntime() {
#ifdef _WIN32
        WSACleanup();
#endif
    }
};

void close_socket(SocketHandle socket) {
    if (socket == kInvalidSocket) return;
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
}

class UniqueSocket {
public:
    explicit UniqueSocket(SocketHandle socket = kInvalidSocket) : socket_(socket) {}
    ~UniqueSocket() { close_socket(socket_); }
    UniqueSocket(const UniqueSocket &) = delete;
    UniqueSocket & operator=(const UniqueSocket &) = delete;
    UniqueSocket(UniqueSocket && other) noexcept
        : socket_(std::exchange(other.socket_, kInvalidSocket)) {}
    SocketHandle get() const noexcept { return socket_; }
private:
    SocketHandle socket_;
};

void set_timeouts(SocketHandle socket, int receive_ms, int send_ms) {
#ifdef _WIN32
    const DWORD receive = static_cast<DWORD>(std::max(0, receive_ms));
    const DWORD send = static_cast<DWORD>(std::max(0, send_ms));
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO,
        reinterpret_cast<const char *>(&receive), sizeof(receive));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO,
        reinterpret_cast<const char *>(&send), sizeof(send));
#else
    const auto timeout = [](int milliseconds) {
        timeval value{};
        value.tv_sec = std::max(0, milliseconds) / 1000;
        value.tv_usec = (std::max(0, milliseconds) % 1000) * 1000;
        return value;
    };
    const auto receive = timeout(receive_ms);
    const auto send = timeout(send_ms);
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &receive, sizeof(receive));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &send, sizeof(send));
#endif
}

void send_all(SocketHandle socket, const std::string & data) {
    std::size_t offset = 0;
    while (offset < data.size()) {
        const auto remaining = data.size() - offset;
#ifdef _WIN32
        const int sent = send(
            socket, data.data() + offset,
            static_cast<int>(std::min<std::size_t>(
                remaining, static_cast<std::size_t>(std::numeric_limits<int>::max()))), 0);
#else
        const auto sent = send(socket, data.data() + offset, remaining, MSG_NOSIGNAL);
#endif
        if (sent <= 0) throw std::runtime_error("socket send failed");
        offset += static_cast<std::size_t>(sent);
    }
}

std::size_t parse_content_length(const std::string & text) {
    unsigned long long value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("invalid Content-Length header");
    }
    return static_cast<std::size_t>(value);
}

HttpRequest read_request(SocketHandle socket, const HttpServerOptions & options) {
    std::string bytes;
    bytes.reserve(std::min<std::size_t>(options.max_header_bytes, 8192));
    std::array<char, 8192> buffer{};
    std::size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
#ifdef _WIN32
        const int received = recv(socket, buffer.data(), static_cast<int>(buffer.size()), 0);
#else
        const auto received = recv(socket, buffer.data(), buffer.size(), 0);
#endif
        if (received <= 0) throw std::invalid_argument("connection closed before HTTP headers");
        bytes.append(buffer.data(), static_cast<std::size_t>(received));
        header_end = bytes.find("\r\n\r\n");
        if (header_end == std::string::npos && bytes.size() > options.max_header_bytes) {
            throw std::invalid_argument("HTTP headers exceed configured limit");
        }
    }
    if (header_end > options.max_header_bytes) {
        throw std::invalid_argument("HTTP headers exceed configured limit");
    }

    HttpRequest request;
    std::istringstream stream(bytes.substr(0, header_end));
    std::string line;
    if (!std::getline(stream, line)) throw std::invalid_argument("missing HTTP request line");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::string version;
    std::istringstream request_line(line);
    request_line >> request.method >> request.path >> version;
    std::string extra;
    if (request.method.empty() || request.path.empty() ||
        version.rfind("HTTP/1.", 0) != 0 || request_line >> extra) {
        throw std::invalid_argument("invalid HTTP request line");
    }
    const auto query = request.path.find('?');
    if (query != std::string::npos) {
        request.query = request.path.substr(query + 1);
        request.path.resize(query);
    }
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto colon = line.find(':');
        if (colon == std::string::npos) throw std::invalid_argument("invalid HTTP header line");
        const auto name = lower_ascii(trim(line.substr(0, colon)));
        const auto value = trim(line.substr(colon + 1));
        if (name.empty()) throw std::invalid_argument("empty HTTP header name");
        if (!request.headers.emplace(name, value).second) {
            throw std::invalid_argument("duplicate HTTP header: " + name);
        }
    }
    if (request.headers.count("transfer-encoding")) {
        throw std::invalid_argument("Transfer-Encoding is not supported; send Content-Length");
    }
    std::size_t body_size = 0;
    if (const auto found = request.headers.find("content-length"); found != request.headers.end()) {
        body_size = parse_content_length(found->second);
    }
    if (body_size > options.max_request_body_bytes) {
        throw std::length_error("request body exceeds configured limit");
    }
    request.body = bytes.substr(header_end + 4);
    if (request.body.size() > body_size) request.body.resize(body_size);
    while (request.body.size() < body_size) {
#ifdef _WIN32
        const int received = recv(socket, buffer.data(), static_cast<int>(buffer.size()), 0);
#else
        const auto received = recv(socket, buffer.data(), buffer.size(), 0);
#endif
        if (received <= 0) throw std::invalid_argument("connection closed during HTTP body");
        const auto need = body_size - request.body.size();
        request.body.append(buffer.data(), std::min<std::size_t>(need, received));
    }
    return request;
}

bool safe_header(const std::string & name, const std::string & value) {
    if (name.empty() || value.find_first_of("\r\n") != std::string::npos) return false;
    for (const unsigned char character : name) {
        if (!std::isalnum(character) && character != '-' && character != '_') return false;
    }
    return true;
}

std::string serialize(const HttpResponse & response) {
    std::ostringstream header;
    header << "HTTP/1.1 " << response.status << ' ' << status_text(response.status) << "\r\n"
           << "Content-Type: " << response.content_type << "\r\n"
           << "Content-Length: " << response.body.size() << "\r\n"
           << "Connection: close\r\n";
    for (const auto & entry : response.headers) {
        if (!safe_header(entry.first, entry.second)) {
            throw std::logic_error("unsafe response header");
        }
        header << entry.first << ": " << entry.second << "\r\n";
    }
    header << "\r\n";
    auto output = header.str();
    output += response.body;
    return output;
}

UniqueSocket bind_socket(const HttpServerOptions & options) {
    UniqueSocket socket(socket(AF_INET, SOCK_STREAM, 0));
    if (socket.get() == kInvalidSocket) throw std::runtime_error("could not create listen socket");
    int yes = 1;
#ifdef _WIN32
    setsockopt(socket.get(), SOL_SOCKET, SO_REUSEADDR,
        reinterpret_cast<const char *>(&yes), sizeof(yes));
#else
    setsockopt(socket.get(), SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(options.port));
    if (inet_pton(AF_INET, options.host.c_str(), &address.sin_addr) != 1) {
        throw std::invalid_argument("server host must be an IPv4 address");
    }
    if (bind(socket.get(), reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        throw std::runtime_error("could not bind " + options.host + ':' + std::to_string(options.port));
    }
    if (listen(socket.get(), 16) != 0) throw std::runtime_error("could not listen on socket");
    return socket;
}

bool wait_for_client(SocketHandle socket) {
    fd_set read;
    FD_ZERO(&read);
    FD_SET(socket, &read);
    timeval timeout{};
    timeout.tv_usec = 250000;
#ifdef _WIN32
    const auto result = select(0, &read, nullptr, nullptr, &timeout);
#else
    const auto result = select(socket + 1, &read, nullptr, nullptr, &timeout);
#endif
    if (result < 0) {
#ifdef _WIN32
        if (WSAGetLastError() == WSAEINTR) return false;
#else
        if (errno == EINTR) return false;
#endif
        throw std::runtime_error("server select failed");
    }
    return result > 0;
}

void handle_client(
    SocketHandle accepted,
    const HttpServerOptions & options,
    const HttpHandler & handler) {
    UniqueSocket socket(accepted);
    set_timeouts(socket.get(), options.receive_timeout_ms, options.send_timeout_ms);
    HttpResponse response;
    try {
        const auto request = read_request(socket.get(), options);
        response = handler(request);
    } catch (const std::length_error & error) {
        response = error_response(413, error.what(), "request_too_large");
    } catch (const std::invalid_argument & error) {
        response = error_response(400, error.what(), "invalid_request_error");
    } catch (const std::exception & error) {
        response = error_response(500, error.what(), "server_error");
    }
    try {
        send_all(socket.get(), serialize(response));
    } catch (const std::exception & error) {
        std::cerr << "[yue2-server] response failed: " << error.what() << '\n';
    }
}

} // namespace

HttpResponse json_response(std::string body, int status) {
    return {status, "application/json", std::move(body), {}};
}

HttpResponse error_response(
    int status, const std::string & message, const std::string & type) {
    return json_response(
        "{\"error\":{\"message\":" + json::quote(message) +
        ",\"type\":" + json::quote(type) + "}}", status);
}

void serve_http(
    const HttpServerOptions & options,
    const HttpHandler & handler,
    const std::atomic<bool> & stop_requested) {
    if (options.port < 1 || options.port > 65535 || options.max_header_bytes < 1024 ||
        options.max_request_body_bytes == 0) {
        throw std::invalid_argument("invalid HTTP server options");
    }
    SocketRuntime runtime;
    auto listening = bind_socket(options);
    struct Worker {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> done;
    };
    // Finished connection threads are joined as the loop runs. Polling clients
    // open a connection every second or two, so leaving them joinable until
    // shutdown would accumulate thread handles for the life of the server.
    std::vector<Worker> workers;
    const auto reap = [&workers]() {
        for (auto worker = workers.begin(); worker != workers.end();) {
            if (worker->done->load()) {
                worker->thread.join();
                worker = workers.erase(worker);
            } else {
                ++worker;
            }
        }
    };
    std::cout << "yue2-server listening on http://" << options.host << ':' << options.port << '\n';
    while (!stop_requested.load()) {
        reap();
        if (!wait_for_client(listening.get())) continue;
        sockaddr_in address{};
#ifdef _WIN32
        int length = sizeof(address);
#else
        socklen_t length = sizeof(address);
#endif
        const auto client = accept(
            listening.get(), reinterpret_cast<sockaddr *>(&address), &length);
        if (client == kInvalidSocket) {
            if (stop_requested.load()) break;
            continue;
        }
        auto done = std::make_shared<std::atomic<bool>>(false);
        workers.push_back({std::thread([client, &options, &handler, done]() {
            struct MarkDone {
                std::atomic<bool> & flag;
                ~MarkDone() { flag.store(true); }
            } mark{*done};
            handle_client(client, options, handler);
        }), done});
    }
    for (auto & worker : workers) if (worker.thread.joinable()) worker.thread.join();
}

} // namespace yue2::server
