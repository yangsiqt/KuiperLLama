#include <base/base.h>
#include <base/profiler.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include "model/qwen3.h"

static FILE* g_log_file = nullptr;

static void tee_printf(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  va_list args_copy;
  va_copy(args_copy, args);
  vprintf(fmt, args);
  if (g_log_file) {
    vfprintf(g_log_file, fmt, args_copy);
  }
  va_end(args_copy);
  va_end(args);
}

static std::string create_log_file(const char* project_root, const char* tag,
                                   int prompt_tokens, int max_tokens, int bench_runs) {
  std::string logs_dir = std::string(project_root) + "/logs";
  mkdir(logs_dir.c_str(), 0755);

  auto now = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;
  struct tm tm_buf;
  localtime_r(&t, &tm_buf);

  char filename[512];
  snprintf(filename, sizeof(filename),
           "%s/bench_%s_p%d_d%d_r%d_%04d%02d%02d_%02d%02d%02d.md",
           logs_dir.c_str(), tag,
           prompt_tokens, max_tokens, bench_runs,
           tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
           tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);

  g_log_file = fopen(filename, "w");
  if (!g_log_file) {
    fprintf(stderr, "Warning: cannot create log file %s\n", filename);
  }
  return std::string(filename);
}

struct BenchConfig {
  std::string checkpoint_path;
  std::string tokenizer_path;
  std::string prompt =
      "Explain the history and evolution of artificial intelligence from its origins in the 1950s "
      "to modern deep learning. Cover key milestones including the Dartmouth Conference, expert "
      "systems, the AI winters, the rise of machine learning, neural networks, and the transformer "
      "architecture. Discuss how large language models like GPT and LLaMA work at a high level.";
  int max_tokens = 512;
  int warmup_runs = 2;
  int bench_runs = 10;
  bool enable_profiling = true;
};

struct BenchResult {
  int prompt_tokens = 0;
  int generated_tokens = 0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double total_ms = 0.0;
  double first_token_ms = 0.0;
  size_t peak_mem_used_bytes = 0;
  size_t total_mem_bytes = 0;
};

static size_t get_gpu_memory_used() {
  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  return total_bytes - free_bytes;
}

static size_t get_gpu_memory_total() {
  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  return total_bytes;
}

std::string fill_template(const std::string& content) {
  const std::string format = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
  std::string result = format;
  size_t pos = result.find("%s");
  if (pos != std::string::npos) {
    result.replace(pos, 2, content);
  }
  return result;
}

BenchResult run_benchmark(const model::Qwen3Model& model, const std::string& prompt,
                          int max_tokens, bool profile) {
  BenchResult result;
  auto& profiler = base::OpProfiler::instance();

  const std::string& sentence = fill_template(prompt);
  auto tokens = model.encode(sentence);
  result.prompt_tokens = static_cast<int>(tokens.size());

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  int total_steps = result.prompt_tokens + max_tokens;

  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  size_t peak_mem = get_gpu_memory_used();

  if (profile) {
    profiler.reset();
    profiler.enable();
  }

  auto t_start = std::chrono::high_resolution_clock::now();
  auto t_first_token = t_start;
  bool first_token_recorded = false;

  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < result.prompt_tokens - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      if (!first_token_recorded) {
        cudaDeviceSynchronize();
        t_first_token = std::chrono::high_resolution_clock::now();
        first_token_recorded = true;
      }
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
      if (next != 151645 && next != 151644) {
        words.push_back(next);
      }
    }

    size_t cur_mem = get_gpu_memory_used();
    if (cur_mem > peak_mem) peak_mem = cur_mem;

    if (model.is_sentence_ending(next)) break;
    if (is_prompt) next = tokens.at(pos + 1);
    pos += 1;
  }

  cudaDeviceSynchronize();
  auto t_end = std::chrono::high_resolution_clock::now();

  if (profile) {
    profiler.collect();
    profiler.disable();
  }

  result.generated_tokens = static_cast<int>(words.size());
  result.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  result.prefill_ms = std::chrono::duration<double, std::milli>(t_first_token - t_start).count();
  result.decode_ms = std::chrono::duration<double, std::milli>(t_end - t_first_token).count();
  result.first_token_ms = result.prefill_ms;
  result.peak_mem_used_bytes = peak_mem;
  result.total_mem_bytes = get_gpu_memory_total();

  return result;
}

void print_markdown_report(const std::vector<BenchResult>& results, const BenchConfig& config) {
  tee_printf("\n# Benchmark Report\n\n");

#ifdef USE_NAIVE_MHA
  tee_printf("**MHA Mode: Naive (baseline)**\n\n");
#else
  tee_printf("**MHA Mode: Flash Attention**\n\n");
#endif

  int device_id = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  tee_printf("## Environment\n\n");
  tee_printf("| Item | Value |\n");
  tee_printf("|------|-------|\n");
  tee_printf("| GPU | %s |\n", prop.name);
  tee_printf("| VRAM | %.1f GB |\n", prop.totalGlobalMem / 1073741824.0);
  tee_printf("| CUDA Cores | %d |\n", prop.multiProcessorCount);
  tee_printf("| Model | Qwen3 (quant: %s) |\n",
             config.checkpoint_path.find("int4") != std::string::npos ? "INT4" :
             config.checkpoint_path.find("int8") != std::string::npos ? "INT8" : "FP32");
  tee_printf("| Checkpoint | %s |\n", config.checkpoint_path.c_str());
  tee_printf("| Prompt | \"%s\" |\n", config.prompt.c_str());
  tee_printf("| Max Tokens | %d |\n", config.max_tokens);
  tee_printf("| Bench Runs | %d |\n", config.bench_runs);

  double avg_prefill_ms = 0, avg_decode_ms = 0, avg_total_ms = 0;
  double avg_first_token_ms = 0;
  int avg_prompt_tokens = 0, avg_gen_tokens = 0;
  size_t max_peak_mem = 0;

  for (auto& r : results) {
    avg_prefill_ms += r.prefill_ms;
    avg_decode_ms += r.decode_ms;
    avg_total_ms += r.total_ms;
    avg_first_token_ms += r.first_token_ms;
    avg_prompt_tokens += r.prompt_tokens;
    avg_gen_tokens += r.generated_tokens;
    if (r.peak_mem_used_bytes > max_peak_mem) max_peak_mem = r.peak_mem_used_bytes;
  }
  int n = static_cast<int>(results.size());
  avg_prefill_ms /= n;
  avg_decode_ms /= n;
  avg_total_ms /= n;
  avg_first_token_ms /= n;
  avg_prompt_tokens /= n;
  avg_gen_tokens /= n;

  double prefill_tps = avg_prompt_tokens > 0 ? avg_prompt_tokens / (avg_prefill_ms / 1000.0) : 0;
  double decode_tps = avg_gen_tokens > 0 ? avg_gen_tokens / (avg_decode_ms / 1000.0) : 0;

  tee_printf("\n## Performance Summary\n\n");
  tee_printf("| Metric | Value |\n");
  tee_printf("|--------|------:|\n");
  tee_printf("| Prompt Tokens | %d |\n", avg_prompt_tokens);
  tee_printf("| Generated Tokens | %d |\n", avg_gen_tokens);
  tee_printf("| First Token Latency | %.2f ms |\n", avg_first_token_ms);
  tee_printf("| Prefill Throughput | %.1f tok/s |\n", prefill_tps);
  tee_printf("| Decode Throughput | %.1f tok/s |\n", decode_tps);
  tee_printf("| Total Time | %.2f ms |\n", avg_total_ms);
  tee_printf("| Peak GPU Memory | %.1f MB / %.1f MB |\n", max_peak_mem / 1048576.0,
             results[0].total_mem_bytes / 1048576.0);

  tee_printf("\n## Per-Run Details\n\n");
  tee_printf("| Run | Prefill(ms) | Decode(ms) | Total(ms) | Gen Tokens | Decode tok/s |\n");
  tee_printf("|----:|------------:|-----------:|----------:|-----------:|-------------:|\n");
  for (int i = 0; i < n; ++i) {
    auto& r = results[i];
    double dtps = r.generated_tokens > 0 ? r.generated_tokens / (r.decode_ms / 1000.0) : 0;
    tee_printf("| %3d | %11.2f | %10.2f | %9.2f | %10d | %12.1f |\n", i + 1, r.prefill_ms,
               r.decode_ms, r.total_ms, r.generated_tokens, dtps);
  }
}

static void tee_print_profiler_report(const base::OpProfiler& profiler) {
  tee_printf("\n## Operator Profiling Report\n\n");
  tee_printf("%s", profiler.markdown_table().c_str());
  tee_printf("\nTotal operator time: %.2f ms\n", profiler.total_ms());
}

static std::string make_long_prompt(const model::Qwen3Model& model, const std::string& seed,
                                    int target_tokens) {
  auto seed_tokens = model.encode(seed);
  int seed_len = static_cast<int>(seed_tokens.size());
  if (seed_len >= target_tokens) return seed;

  int repeats = (target_tokens / seed_len) + 1;
  std::string long_prompt;
  for (int i = 0; i < repeats; ++i) {
    if (i > 0) long_prompt += " Furthermore, ";
    long_prompt += seed;
  }
  auto final_tokens = model.encode(long_prompt);
  while (static_cast<int>(final_tokens.size()) > target_tokens && long_prompt.size() > 100) {
    long_prompt = long_prompt.substr(0, long_prompt.size() - 50);
    final_tokens = model.encode(long_prompt);
  }
  return long_prompt;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printf(
        "Usage: %s <checkpoint> <tokenizer> [max_tokens] [bench_runs] [quant_bits] "
        "[prompt_tokens] [awq]\n",
        argv[0]);
    printf("  max_tokens    : max tokens to generate (default: 512)\n");
    printf("  bench_runs    : number of benchmark iterations (default: 10)\n");
    printf("  quant_bits    : 0=FP32, 4=INT4, 8=INT8 (default: 0)\n");
    printf("  prompt_tokens : target prompt length in tokens (default: 0=use built-in)\n");
    printf("  awq           : 1=AWQ enabled (default: 0)\n");
    return 1;
  }

  BenchConfig config;
  config.checkpoint_path = argv[1];
  config.tokenizer_path = argv[2];
  if (argc >= 4) config.max_tokens = std::atoi(argv[3]);
  if (argc >= 5) config.bench_runs = std::atoi(argv[4]);
  int quant_bits = 0;
  if (argc >= 6) quant_bits = std::atoi(argv[5]);
  int target_prompt_tokens = 0;
  if (argc >= 7) target_prompt_tokens = std::atoi(argv[6]);
  bool has_awq = false;
  if (argc >= 8) has_awq = (std::atoi(argv[7]) == 1);

#ifdef USE_NAIVE_MHA
  const char* mha_tag = "naive";
#else
  const char* mha_tag = "flash";
#endif

  std::string quant_tag = quant_bits > 0 ? std::string("int") + std::to_string(quant_bits) : "fp32";
  if (has_awq) quant_tag += "_awq";
  std::string full_tag = std::string(mha_tag) + "_" + quant_tag;

  std::string log_path = create_log_file("/home/dzy/KuiperLLama", full_tag.c_str(),
                                          target_prompt_tokens > 0 ? target_prompt_tokens : 0,
                                          config.max_tokens, config.bench_runs);
  if (g_log_file) {
    tee_printf("Log file: %s\n", log_path.c_str());
  }

  bool is_quant = (quant_bits == 4 || quant_bits == 8);
  tee_printf("Loading model (quant_bits=%d, awq=%d)...\n", quant_bits, has_awq ? 1 : 0);
  model::Qwen3Model model(base::TokenizerType::kEncodeBpe, config.tokenizer_path,
                           config.checkpoint_path, is_quant, quant_bits, has_awq);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "Model init failed: " << init_status.get_err_code();
  }

  auto& profiler = base::OpProfiler::instance();

  tee_printf("Model loaded. GPU memory after load: %.1f MB\n\n", get_gpu_memory_used() / 1048576.0);

  if (target_prompt_tokens > 0) {
    config.prompt = make_long_prompt(model, config.prompt, target_prompt_tokens);
    auto check_tokens = model.encode(fill_template(config.prompt));
    tee_printf("Generated long prompt: %d tokens (target: %d)\n\n",
               static_cast<int>(check_tokens.size()), target_prompt_tokens);
  }

  tee_printf("Warming up (%d runs)...\n", config.warmup_runs);
  for (int i = 0; i < config.warmup_runs; ++i) {
    run_benchmark(model, config.prompt, std::min(config.max_tokens, 16), false);
  }

  tee_printf("Running benchmark (%d runs, max_tokens=%d)...\n", config.bench_runs, config.max_tokens);
  std::vector<BenchResult> results;
  for (int i = 0; i < config.bench_runs; ++i) {
    tee_printf("  Run %d/%d...\n", i + 1, config.bench_runs);
    bool do_profile = config.enable_profiling && (i == config.bench_runs - 1);
    auto r = run_benchmark(model, config.prompt, config.max_tokens, do_profile);
    results.push_back(r);
  }

  print_markdown_report(results, config);

  if (config.enable_profiling) {
    tee_print_profiler_report(profiler);
  }

  tee_printf("\nBenchmark complete.\n");

  if (g_log_file) {
    fclose(g_log_file);
    g_log_file = nullptr;
  }
  return 0;
}
