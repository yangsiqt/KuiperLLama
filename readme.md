# KuiperLLama 扩展

在开源项目KuiperLLama的 **Qwen3** 路径上，增量实现 **Flash Attention / Naive MHA 可切换**、**可复现 Benchmark**、以及 **INT4（Q4_0）与 AWQ 权重量化导出 + CUDA 推理**，下文给出环境配置、编译运行命令与benchmark数据。

| 方向 | 分支 | 内容 |
|------|----------|------|
| **Flash + Benchmark** | `feature/flash-attention` | FlashAttention-2 风格 MHA、`bench_qwen3` 分层计时、`OpProfiler` |
| **INT4 + AWQ** | feature/quan` | **Q4_0** 导出、**AWQ** 激活感知缩放、C++ mmap 加载、INT4 GEMM + `awq_input_scales` |

---



## 1. 环境要求与安装配置

### 硬件与操作系统

| 项 | 说明 |
|----|------|
| **GPU** | **NVIDIA GPU + CUDA Toolkit** **12.8** |
| **显存** | 约 **0.6B FP32** 可尝试 6GB 级；**8B FP32** 建议 **16GB+**（如 3090 24GB） |
| **OS** | **Ubuntu2204** 或 **WSL2（Ubuntu2204）** |

### 工具链

| 项 | 要求 |
|----|------|
| **CMake** | ≥ **3.16** |
| **C++** | **GCC 11+**，标准 **C++17** |
| **CUDA** | 与驱动匹配的 Toolkit 12.8；工程默认 `CMAKE_CUDA_COMPILER` 为 `/usr/local/cuda/bin/nvcc`，路径不同时请改 `CMakeLists.txt` |

### C++ 第三方依赖（与课程 `readme.md` 一致）

不开启 **CPM** 时需本机安装并可被 `find_package` 找到：

1. [glog](https://github.com/google/glog)  
2. [gtest](https://github.com/google/googletest)  
3. [sentencepiece](https://github.com/google/sentencepiece)  
4. [Armadillo](https://arma.sourceforge.net/download.html)（常配 **OpenBLAS**）  

开启 **Qwen3**（或 LLaMA3 / Qwen2）时还会用到 **absl、re2**；**`USE_CPM=ON`** 时由 CPM 拉取 **GTest、glog、Armadillo、sentencepiece** 及 **absl、re2、nlohmann_json**（推荐有网络时开启）。

### CMake 常用选项（`CMakeLists.txt`）

| 选项 | 默认 | 含义 |
|------|------|------|
| `USE_CPM` | OFF | 自动拉取上述 C++ 依赖 |
| `QWEN3_SUPPORT` | OFF | **必须 ON** 才能编 `qwen3_infer` / `bench_qwen3` |
| `USE_NAIVE_MHA` | OFF | OFF：**Flash Attention**；ON：Naive MHA |
| `ENABLE_NVTX` | OFF | 供 nsys 打点（需 `libnvToolsExt`） |
| `LLAMA3_SUPPORT` / `QWEN2_SUPPORT` | OFF | 其它模型 |

### CUDA 架构（`cmake/cuda.cmake`）

- 有 GPU 时一般自动检测 **`CMAKE_CUDA_ARCHITECTURES`**。  
- **无 GPU 配置工程**时可手动指定，例如 Ampere：**`-DCMAKE_CUDA_ARCHITECTURES=86`**（3090 / 3060 等）。算力对照见 [NVIDIA 文档](https://developer.nvidia.com/cuda-gpus)。

### 获取模型（Hugging Face）

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 可选：镜像
pip3 install huggingface-cli

# Qwen3-0.6B（小模型，Flash / Naive bench 常用）
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir /home/dzy/models/Qwen3-0.6B --local-dir-use-symlinks False

# Qwen3-8B（INT4 / AWQ / PPL 等）
huggingface-cli download --resume-download Qwen/Qwen3-8B \
  --local-dir /home/dzy/models/Qwen3-8B --local-dir-use-symlinks False
```

`--local-dir` 可按本机改为例如 `~/models/...`。

### Python 依赖（导出 / 校准 / PPL）

| 包 | 用途 |
|----|------|
| `torch` | FP32 链、INT4/AWQ 导出、`eval_awq_ppl.py` |
| `safetensors` | `export_int4.py` 读 HF |
| `transformers` | `load.py`、`write_bin.py`、`export_int4_awq.py` |
| `tqdm` | `load.py` |
| `datasets` | 可选：`wikitext2` 校准与 `eval_awq_ppl.py` |

**AWQ 导出**建议在 GPU 上执行；**8B FP16 校准**常需 **约 16GB+** 显存。



## 2. 编译与运行

### 一键编译（Qwen3 + Flash，默认）

```bash
cd /home/dzy/za/kpllama/KuiperLLama
cmake -B build -S . -DUSE_CPM=ON -DQWEN3_SUPPORT=ON
cmake --build build --target bench_qwen3 qwen3_infer -j$(nproc)
```

无 CPM：先满足上文 C++ 依赖，再 `cmake -B build -S . -DQWEN3_SUPPORT=ON`（不加 `USE_CPM`）。  
**NVTX**：配置时加 `-DENABLE_NVTX=ON`。

### 运行 C++ 推理需要

| 文件                   | 说明                                                         |
| ---------------------- | ------------------------------------------------------------ |
| **单文件权重 `*.bin`** | FP32：`write_bin.py` 产物；INT4：`export_int4.py`；INT4+AWQ：`export_int4_awq.py` |
| **`tokenizer.json`**   | 一般为 **`模型目录/tokenizer.json`**，作第二参数             |

导出脚本通常依赖同目录 **`config.json`** 与 **`*.safetensors`**（或分片权重）。

### 导出推理所需权重文件（导出命令）

**FP32（`tools/export_qwen3`）**

```bash
cd tools/export_qwen3
python load.py --model_name /home/dzy/models/Qwen3-0.6B --output_file qwen3_0.6b_weights.pth
python write_bin.py -p qwen3_0.6b_weights.pth -o qwen0.6.bin2 \
  -n /home/dzy/models/Qwen3-0.6B -d cpu
```

**INT4 朴素 Q4_0（直接读 HF `safetensors`）**

```bash
cd tools/export_qwen3
pip install torch safetensors
python export_int4.py --model_dir /home/dzy/models/Qwen3-8B --group_size 64 \
  --output /home/dzy/models/Qwen3-8B-int4.bin
```

**INT4 + AWQ（GPU 校准）**

```bash
cd tools/export_qwen3
pip install torch safetensors transformers datasets
python export_int4_awq.py --model_dir /home/dzy/models/Qwen3-8B --group_size 64 \
  --device cuda --calibration wikitext2 \
  --outlier_ratio 0.01 \
  --output /home/dzy/models/Qwen3-8B-int4-awq.bin
```

### 实验 A：Naive MHA vs Flash Attention（FP32）

同一 `checkpoint` 与 `tokenizer.json`，仅切换 **`USE_NAIVE_MHA`**；bench 示例：`max_tokens=128`，`bench_runs=6`（FP32 可省略后三个参数）。

**Naive MHA**

```bash
cd /home/dzy/za/kpllama/KuiperLLama
cmake -B build -S . -DUSE_CPM=ON -DQWEN3_SUPPORT=ON -DUSE_NAIVE_MHA=ON
cmake --build build --target bench_qwen3 -j8
./build/demo/bench_qwen3 /home/dzy/za/kpllama/qwen306/qwen3-0.6b/qwen0.6.bin2 \
  /home/dzy/za/kpllama/qwen306/qwen3-0.6b/tokenizer.json 128 6
```

**Flash Attention**

```bash
cd /home/dzy/za/kpllama/KuiperLLama
cmake -B build -S . -DUSE_CPM=ON -DQWEN3_SUPPORT=ON -DUSE_NAIVE_MHA=OFF
cmake --build build --target bench_qwen3 -j8
./build/demo/bench_qwen3 /home/dzy/za/kpllama/qwen306/qwen3-0.6b/qwen0.6.bin2 \
  /home/dzy/za/kpllama/qwen306/qwen3-0.6b/tokenizer.json 128 6
```

若权重与 tokenizer 在 `write_bin.py` 输出目录下，也可使用：`/home/dzy/models/Qwen3-0.6B` 与同目录 `tokenizer.json`。Naive/Flash 频繁切换可用下文 **双 build 目录**。

### 实验 B：INT4 Benchmark（朴素 INT4，`awq=0`）

需已按上文导出 **INT4 `.bin`** 并完成编译。

```bash
cd /home/dzy/za/kpllama/KuiperLLama
./build/demo/bench_qwen3 /home/dzy/models/Qwen3-8B-int4.bin \
  /home/dzy/models/Qwen3-8B/tokenizer.json 128 5 4 128
```

### 实验 C：INT4 + AWQ 推理 / Bench / 困惑度

**交互推理**（最后一参 `1` 表示按 AWQ 格式读头与 scales）：

```bash
./build/demo/qwen3_infer /home/dzy/models/Qwen3-8B-int4-awq.bin \
  /home/dzy/models/Qwen3-8B/tokenizer.json 4 1
```

**Bench**（第七参数 `awq=1`）：

```bash
./build/demo/bench_qwen3 /home/dzy/models/Qwen3-8B-int4-awq.bin \
  /home/dzy/models/Qwen3-8B/tokenizer.json 128 5 4 128 1
```

**困惑度（PyTorch 侧，与 C++ `.bin` 互补）**：

```bash
cd /home/dzy/za/kpllama/KuiperLLama/tools/export_qwen3
python eval_awq_ppl.py \
  --model_dir /home/dzy/models/Qwen3-8B \
  --modes fp16,naive,awq_clip_outlier \
  --max_tokens 8192 \
  --calibration wikitext2 \
  --wikitext_calib_samples 128 \
  --cache_awq /tmp/awq_cache_v2.pt \
  --outlier_ratio 0.01 \
  --device cuda
```

### `bench_qwen3` 可执行文件参数

```text
bench_qwen3 <checkpoint> <tokenizer> [max_tokens] [bench_runs] [quant_bits] [prompt_tokens] [awq]
```

| argv | 含义 | 默认 |
|------|------|------|
| 5 | `quant_bits` | `0`（FP32） |
| 6 | `prompt_tokens` | `0`：内置长 prompt；`>0`：扩展到约该长度 |
| 7 | `awq` | `0`；AWQ 权重用 `1` |

示例：`./build/demo/bench_qwen3 <ckpt> <tok> 512 10 4 0 1`（INT4-AWQ + 内置长 prompt）。



## 3. 性能与精度

下列数据在**固定 checkpoint 与参数**下整理；**换 GPU、驱动或参数会漂移**，以本机 `logs/` 与脚本输出为准。

### 测试口径

| 实验 | 硬件 / 条件 |
|------|-------------|
| Naive vs Flash | **RTX 3060 Laptop**；FP32 小模型；`max_tokens=128`，`bench_runs=20`，仅切换 `USE_NAIVE_MHA` |
| Qwen3-8B INT4 | **RTX 3090 24GB**；`128 20 4 128` |
| PPL | `eval_awq_ppl.py`，WikiText-2，`fp16 / naive / awq_clip_outlier` |

### Naive MHA vs Flash Attention（FP32， Qwen3-0.6B 量级）

| 指标 | Naive MHA | Flash Attention | 变化 |
|------|-----------|-----------------|------|
| Decode tok/s | 82.6 | 93.1 | +12.7% |
| Prefill tok/s | 95.2 | 105.1 | +10.4% |
| 首 Token 延迟 | 273 ms | 247 ms | −9.5% |
| 总时间 | 1836 ms | 1633 ms | −11.0% |
| MHA_Kernel 耗时 | 72.22 ms | 76.73 ms | +6.2% |
| 算子总耗时 | 3459 ms | 3018 ms | −12.8% |
| Peak 显存 | 4123.5 MB | 4123.5 MB | 持平 |

端到端 Flash 更优；单计 **MHA_Kernel** 可能 Flash 略慢，整体算子时间仍下降；峰值显存接近。

### Qwen3-8B INT4量化推理（3090）

| 指标 | 值 |
|------|-----|
| Prefill 吞吐 | 53.5 tok/s |
| Decode 吞吐 | 51.4 tok/s |
| 首 token 延迟 | 2671 ms |
| 峰值显存 | 18413 MB / 24124 MB（约 76%） |
| 单轮总耗时 | 约 5.1 s |

### 困惑度 PPL

| 模式 | NLL | PPL |
|------|-----|-----|
| FP16（基线） | 2.3814 | 10.82 |
| naive（朴素 INT4） | 2.4314 | 11.37 |
| awq_clip_outlier | 2.3999 | 11.02 |

awq_clip_outlier明显优于 naive，PPL降低3.2%，更接近 FP16。

---

