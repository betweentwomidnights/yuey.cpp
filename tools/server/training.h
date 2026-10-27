// Local training process registry. The worker survives an HTTP-server restart;
// the server adopts its PID + creation identity before accepting GPU work.
#pragma once
#include "json.h"
#include "http.h"
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <iterator>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/file.h>
#include <fcntl.h>
#endif

namespace yue2::server::training {
namespace fs = std::filesystem;
// Serialize admissions from multiple local servers sharing a run registry.
// The file remains on disk; the OS releases the lock if a server exits.
class RegistryLock {
#ifdef _WIN32
    HANDLE handle_=INVALID_HANDLE_VALUE;
#else
    int handle_=-1;
#endif
public:
    explicit RegistryLock(const fs::path &root) {
#ifdef _WIN32
        handle_=CreateFileW((root/"admission.lock").c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(handle_==INVALID_HANDLE_VALUE) throw std::runtime_error("Training registry is busy; retry shortly");
#else
        handle_=open((root/"admission.lock").c_str(),O_CREAT|O_RDWR,0600);
        if(handle_<0) throw std::runtime_error("Cannot open training registry lock");
        if(flock(handle_,LOCK_EX|LOCK_NB)!=0) { close(handle_); handle_=-1; throw std::runtime_error("Training registry is busy; retry shortly"); }
#endif
    }
    ~RegistryLock() {
#ifdef _WIN32
        if(handle_!=INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
        if(handle_>=0) close(handle_);
#endif
    }
    RegistryLock(const RegistryLock &)=delete;
    RegistryLock &operator=(const RegistryLock &)=delete;
};
inline std::string read(const fs::path &p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot read training file: " + p.string());
    return {std::istreambuf_iterator<char>(in), {}};
}
inline void write(const fs::path &p, const std::string &s) {
    auto tmp = p; tmp += ".partial";
    { std::ofstream out(tmp, std::ios::binary); out << s; if (!out) throw std::runtime_error("Cannot write training file"); }
#ifdef _WIN32
    if (!MoveFileExW(tmp.c_str(), p.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Cannot commit training file");
#else
    fs::rename(tmp, p);
#endif
}
inline bool identifier(const std::string &s) {
    std::string upper=s;
    for(auto &c:upper) c=static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if(upper=="CON" || upper=="PRN" || upper=="AUX" || upper=="NUL" ||
       (upper.size()==4 && (upper.rfind("COM",0)==0 || upper.rfind("LPT",0)==0) && upper[3]>='1' && upper[3]<='9')) return false;
    return !s.empty() && s.size() <= 64 && s != "." && s != ".." &&
        s.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") == std::string::npos;
}
inline fs::path catalog_companion(const json::Value &metadata, const fs::path &models, const fs::path &adapters) {
    const auto *companion=metadata.find("companion");
    if(!companion) throw std::invalid_argument("trained style has no companion metadata");
    if(json::number(*companion,"strength",1)!=1) throw std::invalid_argument("unsupported trained companion strength");
    fs::path path(json::string(*companion,"path"));
    const auto filename=json::string(*companion,"name",path.filename().string());
    if(!fs::is_regular_file(path)) {
        path=models/filename;
        if(!fs::is_regular_file(path)) path=adapters/filename;
    }
    if(!fs::is_regular_file(path)) throw std::invalid_argument("install the trained style's companion: "+filename);
    return path;
}
inline std::string process_identity(unsigned long pid) {
#ifdef _WIN32
    HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, pid);
    if (!process) return {};
    FILETIME created, exited, kernel, user;
    const bool alive = WaitForSingleObject(process, 0) == WAIT_TIMEOUT && GetProcessTimes(process, &created, &exited, &kernel, &user);
    CloseHandle(process);
    return alive ? std::to_string((static_cast<unsigned long long>(created.dwHighDateTime) << 32) | created.dwLowDateTime) : "";
#else
    // Reap children we launched; adopted workers are not our children.
    int status; if (waitpid(pid, &status, WNOHANG) == static_cast<pid_t>(pid)) return {};
    if (kill(pid, 0) != 0) return {};
    std::ifstream in("/proc/" + std::to_string(pid) + "/stat"); std::string line; std::getline(in, line);
    if (line.empty()) return {}; auto end = line.rfind(')'); std::istringstream fields(line.substr(end+2));
    std::string field; for (int i=3; i<=22; ++i) fields >> field; return field;
#endif
}
#ifdef _WIN32
inline std::wstring quote(const std::wstring &s) {
    std::wstring out=L"\""; size_t slashes=0;
    for (wchar_t c:s) { if(c==L'\\') { ++slashes; continue; }
        out.append(slashes * (c==L'"' ? 2 : 1), L'\\'); slashes=0;
        if(c==L'"') out+=L'\\'; out+=c; }
    out.append(slashes*2,L'\\'); return out+L'"';
}
#endif
inline unsigned long launch(const fs::path &python, const fs::path &script, const fs::path &run,
                            const std::string &action, const fs::path &cwd) {
#ifdef _WIN32
    std::wstring command=quote(python.wstring())+L" "+quote(script.wstring())+L" --run "+quote(run.wstring())+L" --action "+fs::path(action).wstring();
    STARTUPINFOW startup{}; startup.cb=sizeof(startup); PROCESS_INFORMATION process{};
    if (!CreateProcessW(python.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                        nullptr, cwd.c_str(), &startup, &process))
        throw std::runtime_error("Cannot launch training Python; check --training-config");
    CloseHandle(process.hThread); CloseHandle(process.hProcess); return process.dwProcessId;
#else
    const pid_t pid=fork(); if(pid<0) throw std::runtime_error("Cannot fork training worker");
    if(pid==0) { chdir(cwd.c_str()); execl(python.c_str(),python.c_str(),script.c_str(),"--run",run.c_str(),"--action",action.c_str(),nullptr); _exit(127); }
    return pid;
#endif
}

class Manager {
    fs::path config_path_, root_;
    std::mutex mutex_;
    json::Value config() const { return json::parse(read(config_path_)); }
    fs::path run(const std::string &id) const {
        if (!identifier(id)) throw std::invalid_argument("Invalid training run ID");
        auto p=root_/id; if (!fs::exists(p/"request.json")) throw std::invalid_argument("Unknown training run"); return p;
    }
    bool alive(const fs::path &p) const {
        if (!fs::exists(p/"owner.json")) return false;
        auto owner=json::parse(read(p/"owner.json"));
        const auto identity=process_identity(json::u32(owner,"pid",0));
        return !identity.empty() && identity==json::string(owner,"identity");
    }
public:
    explicit Manager(const fs::path &path):config_path_(path) {
#if !defined(_WIN32) && !defined(__linux__)
        if(!path.empty()) throw std::invalid_argument("Local training ownership currently supports Windows and Linux");
#endif
        if (!path.empty()) {
            auto c=config(); const auto root=json::string(c,"run_root");
            if(root.empty()) throw std::invalid_argument("training-config requires run_root");
            root_=fs::absolute(root); fs::create_directories(root_);
        }
    }
    bool active() { std::lock_guard<std::mutex> lock(mutex_); return active_locked(); }
    bool active_locked() const {
        if(root_.empty()) return false;
        for(const auto &entry:fs::directory_iterator(root_)) if(entry.is_directory() && alive(entry.path())) return true;
        return false;
    }
    HttpResponse handle(const HttpRequest &request, const std::function<bool()> &reserve) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(config_path_.empty()) return json_response("{\"configured\":false,\"runs\":[],\"message\":\"Launch with --training-config FILE; see docs/training.md\"}",request.method=="GET"?200:503);
        const std::string prefix="/training/api/runs";
        if(request.path=="/training/api" || (request.path==prefix && request.method=="GET")) {
            std::string body="{\"configured\":true,\"runs\":["; bool first=true;
            for(const auto &entry:fs::directory_iterator(root_)) {
                auto p=entry.path(); if(!entry.is_directory() || !fs::exists(p/"request.json")) continue;
                auto state=fs::exists(p/"state.json")?json::parse(read(p/"state.json")):json::parse("{}");
                auto req=json::parse(read(p/"request.json"));
                if(!first) body+=","; first=false;
                body+="{\"id\":"+json::quote(p.filename().string())+",\"request\":"+json::stringify(req)+
                    ",\"state\":"+json::stringify(state)+",\"active\":"+(alive(p)?"true":"false")+"}";
            }
            return json_response(body+"]}");
        }
        if(request.method!="POST") return error_response(405,"POST required","training");
        RegistryLock registry(root_);
        auto body=json::parse(request.body.empty()?"{}":request.body);
        fs::path p; std::string action;
        if(request.path==prefix) {
            const auto id=json::string(body,"id"); if(!identifier(id)) throw std::invalid_argument("Use 1–64 letters, numbers, hyphens or underscores for the run ID");
            if(fs::exists(root_/id)) throw std::invalid_argument("Run name already exists");
            const auto name=json::string(body,"name",id);
            if(name.empty() || name.size()>240) throw std::invalid_argument("Style name is required and must be short");
            for(const auto &entry:fs::directory_iterator(root_)) {
                const auto file=entry.path()/"request.json";
                if(entry.is_directory() && fs::exists(file) && json::string(json::parse(read(file)),"name")==name)
                    throw std::invalid_argument("Style name already exists");
            }
            const auto models=json::string(config(),"models_dir");
            if(models.empty()) throw std::invalid_argument("training-config requires models_dir");
            if(fs::exists(fs::path(models)/(id+"-F32-LoRA.gguf"))) throw std::invalid_argument("Adapter filename already exists");
            const auto dataset=json::string(body,"dataset"); if(!fs::is_directory(dataset)) throw std::invalid_argument("Dataset folder does not exist");
            // Only product settings are accepted; runtime/model paths come from
            // the server's trusted configuration, never the browser payload.
            for(const auto &[key,value]:body.object)
                if(key!="id" && key!="dataset" && key!="name" && key!="trigger" && key!="window_seconds" &&
                   key!="ar_steps" && key!="nar_steps" && key!="rank" && key!="family" && key!="ar_lr" &&
                   key!="nar_lr" && key!="kl_weight" && key!="seed" && key!="eval_every") throw std::invalid_argument("Unknown training setting: "+key);
            if(active_locked() || !reserve()) return error_response(409,"The device is in use","training");
            p=root_/id; fs::create_directory(p);
            write(p/"request.json",json::stringify(body)); write(p/"runtime.json",json::stringify(config())); action="prepare";
        } else {
            if(request.path.rfind(prefix+"/",0)!=0) return error_response(404,"Unknown training route","training");
            auto tail=request.path.substr(prefix.size()+1); auto slash=tail.find('/');
            if(slash==std::string::npos) throw std::invalid_argument("Training action required");
            p=run(tail.substr(0,slash)); action=tail.substr(slash+1);
            if(action=="stop") {
                if(!alive(p)) throw std::invalid_argument("Run is not active");
                write(p/"stop.request","stop\n");
                // Both native expert trainers stop after checkpointing an update.
                for(const auto &folder:{p/"data/lora",p/"data/dora",p/"nar-run"}) {
                    fs::create_directories(folder); write(folder/"stop.request","stop\n");
                }
                return json_response("{\"success\":true,\"status\":\"stopping\"}");
            }
            if(action!="train" && action!="resume" && action!="publish") throw std::invalid_argument("Unknown training action");
            if(active_locked() || !reserve()) return error_response(409,"The device is in use","training");
            for(const auto &file:{p/"stop.request",p/"data/lora/stop.request",p/"data/dora/stop.request",p/"nar-run/stop.request"}) fs::remove(file);
        }
        auto c=config();
        const auto pid=launch(fs::absolute(json::string(c,"python")),fs::absolute(json::string(c,"workspace"))/"tools/training_workflow.py",p,action,fs::absolute(json::string(c,"workspace")));
        const auto identity=process_identity(pid);
        write(p/"owner.json","{\"pid\":"+std::to_string(pid)+",\"identity\":"+json::quote(identity)+"}");
        return json_response("{\"success\":true,\"id\":"+json::quote(p.filename().string())+",\"pid\":"+std::to_string(pid)+"}",202);
    }
};
} // namespace yue2::server::training
