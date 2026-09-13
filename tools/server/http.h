#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <string>

namespace yue2::server {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string query;
    std::map<std::string, std::string> headers;
    std::string body;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
    std::map<std::string, std::string> headers;
};

using HttpHandler = std::function<HttpResponse(const HttpRequest &)>;

struct HttpServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::uint64_t max_request_body_bytes = 512ULL * 1024 * 1024;
    std::size_t max_header_bytes = 64 * 1024;
    int receive_timeout_ms = 30000;
    int send_timeout_ms = 30000;
};

HttpResponse json_response(std::string body, int status = 200);
HttpResponse error_response(int status, const std::string & message, const std::string & type);

// Blocking HTTP/1.1 server. Each accepted connection is handled on a joined
// worker thread; responses close the connection. Content-Length bodies are
// bounded before allocation and transfer-coding is deliberately rejected.
void serve_http(
    const HttpServerOptions & options,
    const HttpHandler & handler,
    const std::atomic<bool> & stop_requested);

} // namespace yue2::server
