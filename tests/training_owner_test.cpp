#include "server/training.h"
#include <cassert>
#include <iostream>

int main() {
    using namespace yue2::server;
    namespace fs=std::filesystem;
#ifdef _WIN32
    const auto pid=GetCurrentProcessId();
#else
    const auto pid=getpid();
#endif
    const auto root=fs::temp_directory_path()/("yuey-owner-test-"+std::to_string(pid));
    assert(!fs::exists(root)); fs::create_directories(root/"runs/live");
    const auto runtime=root/"runtime.json";
    training::write(runtime,"{\"run_root\":"+json::quote((root/"runs").string())+"}");
    training::write(root/"runs/live/request.json","{\"id\":\"live\",\"dataset\":\"fixture\"}");
    const auto identity=training::process_identity(pid); assert(!identity.empty());
    const auto owner=root/"runs/live/owner.json";
    training::write(owner,"{\"pid\":"+std::to_string(pid)+",\"identity\":"+json::quote(identity)+"}");
    training::Manager first(runtime); assert(first.active());
    // A fresh manager adopts a live worker without launching another process.
    training::Manager adopted(runtime); assert(adopted.active());
    bool reserved=false;
    auto reserve=[&]() { reserved=true; return true; };
    HttpRequest request; request.method="POST"; request.path="/training/api/runs/live/resume"; request.body="{}";
    assert(adopted.handle(request,reserve).status==409); assert(!reserved);
    request.path="/training/api/runs/live/stop";
    assert(adopted.handle(request,reserve).status==200);
    assert(fs::exists(root/"runs/live/stop.request"));
    assert(fs::exists(root/"runs/live/data/dora/stop.request"));
    assert(adopted.active()); // A stop request never releases ownership.
    training::write(owner,"{\"pid\":"+std::to_string(pid)+",\"identity\":\"wrong-creation-time\"}");
    assert(!adopted.active()); // PID reuse cannot resurrect an unrelated worker.
    for(const auto &id:{"../outside","a/b","a\\b",".","..","","CON","com1","LPT9"}) assert(!training::identifier(id));
    fs::create_directories(root/"models"); training::write(root/"models/common.gguf","fixture");
    auto metadata=json::parse("{\"companion\":{\"name\":\"common.gguf\",\"path\":\"missing-install/common.gguf\",\"strength\":1}}");
    assert(training::catalog_companion(metadata,root/"models",root/"runs")==root/"models/common.gguf");
    bool rejected=false;
    try { training::catalog_companion(json::parse("{\"companion\":{\"strength\":0.5}}"),root/"models",root/"runs"); }
    catch(const std::invalid_argument &) { rejected=true; }
    assert(rejected);
    assert(fs::canonical(root).parent_path()==fs::canonical(fs::temp_directory_path()));
    fs::remove_all(root);
    std::cout<<"Training owner adoption, stop and PID-reuse guards passed\n";
}
