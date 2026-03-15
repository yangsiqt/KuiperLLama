#ifndef KUIPER_INCLUDE_BASE_PROFILER_H_
#define KUIPER_INCLUDE_BASE_PROFILER_H_

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace base {

struct ProfileStats {
  double total_ms = 0.0;
  int call_count = 0;
  double min_ms = 1e9;
  double max_ms = 0.0;

  double avg_ms() const { return call_count > 0 ? total_ms / call_count : 0.0; }
  double pct(double total) const { return total > 0 ? total_ms / total * 100.0 : 0.0; }
};

class OpProfiler {
 public:
  static OpProfiler& instance() {
    static OpProfiler inst;
    return inst;
  }

  void set_stream(cudaStream_t s) { stream_ = s; }
  void enable() { enabled_ = true; }
  void disable() { enabled_ = false; }
  bool is_enabled() const { return enabled_; }
  cudaStream_t stream() const { return stream_; }

  void begin(const char* name) {
    if (!enabled_) return;
#ifdef ENABLE_NVTX
    nvtxRangePushA(name);
#endif
    Record rec;
    rec.name = name;
    cudaEventCreate(&rec.start);
    cudaEventCreate(&rec.stop);
    cudaEventRecord(rec.start, stream_);
    pending_.push_back(rec);
  }

  void end() {
    if (!enabled_ || pending_.empty()) return;
#ifdef ENABLE_NVTX
    nvtxRangePop();
#endif
    cudaEventRecord(pending_.back().stop, stream_);
    completed_.push_back(pending_.back());
    pending_.pop_back();
  }

  void collect() {
    if (completed_.empty()) return;
    cudaStreamSynchronize(stream_);
    for (auto& rec : completed_) {
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, rec.start, rec.stop);
      auto& stat = stats_[rec.name];
      stat.total_ms += ms;
      stat.call_count++;
      stat.min_ms = std::min(stat.min_ms, static_cast<double>(ms));
      stat.max_ms = std::max(stat.max_ms, static_cast<double>(ms));
      cudaEventDestroy(rec.start);
      cudaEventDestroy(rec.stop);
    }
    completed_.clear();
  }

  void reset() {
    for (auto& rec : pending_) {
      cudaEventDestroy(rec.start);
      cudaEventDestroy(rec.stop);
    }
    for (auto& rec : completed_) {
      cudaEventDestroy(rec.start);
      cudaEventDestroy(rec.stop);
    }
    pending_.clear();
    completed_.clear();
    stats_.clear();
  }

  const std::map<std::string, ProfileStats>& stats() const { return stats_; }

  double total_ms() const {
    double total = 0.0;
    for (auto& [name, stat] : stats_) {
      total += stat.total_ms;
    }
    return total;
  }

  std::string markdown_table() const {
    double total = total_ms();
    std::string s;
    s += "| Operator | Calls | Total(ms) | Avg(ms) | Min(ms) | Max(ms) | Pct(%) |\n";
    s += "|----------|------:|----------:|--------:|--------:|--------:|-------:|\n";

    std::vector<std::pair<std::string, ProfileStats>> sorted(stats_.begin(), stats_.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });

    for (auto& [name, stat] : sorted) {
      char buf[256];
      snprintf(buf, sizeof(buf), "| %-20s | %5d | %9.2f | %7.3f | %7.3f | %7.3f | %5.1f%% |\n",
               name.c_str(), stat.call_count, stat.total_ms, stat.avg_ms(), stat.min_ms,
               stat.max_ms, stat.pct(total));
      s += buf;
    }
    return s;
  }

  void print_report() const {
    printf("\n## Operator Profiling Report\n\n");
    printf("%s", markdown_table().c_str());
    printf("\nTotal operator time: %.2f ms\n", total_ms());
  }

 private:
  OpProfiler() = default;

  struct Record {
    std::string name;
    cudaEvent_t start{};
    cudaEvent_t stop{};
  };

  bool enabled_ = false;
  cudaStream_t stream_ = nullptr;
  std::vector<Record> pending_;
  std::vector<Record> completed_;
  std::map<std::string, ProfileStats> stats_;
};

class ProfileScope {
 public:
  explicit ProfileScope(const char* name) {
    auto& p = OpProfiler::instance();
    if (!p.is_enabled()) return;
    active_ = true;
    p.begin(name);
  }

  ~ProfileScope() {
    if (!active_) return;
    OpProfiler::instance().end();
  }

  ProfileScope(const ProfileScope&) = delete;
  ProfileScope& operator=(const ProfileScope&) = delete;

 private:
  bool active_ = false;
};

#define PROFILE_CONCAT_IMPL(a, b) a##b
#define PROFILE_CONCAT(a, b) PROFILE_CONCAT_IMPL(a, b)
#define PROFILE_SCOPE(name) ::base::ProfileScope PROFILE_CONCAT(_prof_, __LINE__)(name)

}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_PROFILER_H_
