# KuiperLLama 个人改造说明（面向面试官）

本文档概述在课程开源项目 **KuiperLLama** 基础上的工程化、性能与量化改造。实现按**分支**划分，可分别演示：

| 方向 | 典型分支 | 内容 |
|------|----------|------|
| **Flash + Benchmark** | `feature/flash-attention` 等 | FlashAttention-2 风格 MHA、Naive MHA 对比、`bench_qwen3` 分层计时与日志 |
| **INT4 + AWQ** | `int4` / `awq` 等 | Qwen3 **Q4_0** 权重量化导出、**AWQ** 激活感知缩放、CUDA INT4 GEMM + 运行时输入缩放 |

以下 **「五、编译与运行」** 中的命令与当前代码一致（含量化参数）。**先完成「四、环境与安装」** 再编译、导出与推理。

---

## 一、项目与改造范围

| 维度 | 说明 |
|------|------|
| **基线** | 课程框架：C++17 + CUDA，CPU/CUDA 双后端，支持 LLaMA2/3、Qwen2、Qwen3 |
| **Flash + Benchmark** | `OpProfiler`（cudaEvent）、可选 NVTX；`USE_NAIVE_MHA` 切换 Naive / Flash MHA |
| **INT4 + AWQ** | **Weight-only Q4_0**（分组对称、两 int4 打包为 1 byte）；**朴素 INT4**：仅 `packed + per-group FP32 scale`；**AWQ 版**：文件头带 `awq_flag`，每层额外写入 **per-channel `1/s`（awq_input_scales）**，推理时先做 `x * (1/s)` 再 INT4 反量化 MatMul；可选 **outlier 列 FP32**（`awq_flag=2`） |
| **未量化路径** | `quant_bits=0`：FP32 权重与激活（与课程一致） |

---

## 二、工程功能摘要

### 2.1 Benchmark（`bench_qwen3`）

- Prefill / Decode 分离、首 token 延迟、tok/s、峰值显存、`OpProfiler` 算子表、Markdown 日志至 `logs/`。
- 日志文件名包含 MHA 模式（`flash` / `naive`）、量化标签（`fp32` / `int4` / `int4_awq` 等）及 bench 参数摘要。
- **注意**：`create_log_file` 内工程根路径需与本机一致（默认在源码中写死，部署时请改为 `getenv("KUIPER_ROOT")` 或相对路径）。

### 2.2 Flash Attention（CUDA MHA）

- 默认 Flash：`cmake` 未开启 `USE_NAIVE_MHA`。
- 对比基线：`-DUSE_NAIVE_MHA=ON`。

### 2.3 INT4 量化（Qwen3）

- **导出**
  - `tools/export_qwen3/export_int4.py`：**朴素 Q4_0**，输出默认 `Qwen3-<目录名>-int4.bin`（或 `--output`）。
  - `tools/export_qwen3/export_int4_awq.py`：**AWQ + Q4_0**，需 GPU 上做校准与逐层 `alpha` 网格搜索；输出默认 `Qwen3-<目录名>-int4-awq.bin`；`--outlier_ratio>0` 时写入 `awq_flag=2` 及 outlier 权重块。
- **文件布局（量化，与 `Model::read_model_file` 一致）**
  1. `ModelConfig` 8×int32（dim、hidden、layers、heads、kv_heads、vocab、seq、intermediate）
  2. `group_size` int32
  3. **仅 AWQ / AWQ+outlier**：`awq_flag` int32（`1`=AWQ，`2`=AWQ+outlier）；若 `2` 再跟 `outlier_ratio` float32
  4. 各量化层：`packed uint8`（`out_features × in_features / 2`）+ `FP32 scales`（per group）+（若 AWQ）`FP32 awq_input_scales`（长度 `in_features`）+（若 outlier）`k`、`k` 个列索引、`k×out_features` FP32
  5. 其余：Embedding、RMSNorm、Q/K norm 等仍为 **FP32** 裸写
- **推理**
  - `MatmulLayer`：`quant_bits==4` 时走 `get_matmul_kernel_quant4`；若存在 `awq_scales_`，kernel 内与 INT4 反量化路径配合（输入先按 `1/s` 缩放）。
  - **仅支持 CUDA**：`is_quant_model && CPU` 会报错。

### 2.4 精度评估脚本（可选）

- `tools/export_qwen3/eval_awq_ppl.py`：WikiText-2 子集上对比 FP16 / naive Q4_0 / 多种 AWQ 变体的 PPL（与导出同一套 Q4_0 定义）。
- `tools/export_qwen3/compare_calibration.py`：校准数据对 AWQ 的影响对比。

---

## 三、实测性能参考（本机记录，可复现口径）

下列表格为在**固定 checkpoint 与 bench 参数**下的整理结果，便于面试展示；**换 GPU、驱动或参数后数值会漂移**，请以本机 `logs/` 与脚本输出为准。

### 3.1 环境摘要

| 实验 | 典型硬件 | 说明 |
|------|----------|------|
| Flash vs Naive（§5.2） | 与 FP32 小模型 bench 同机 | 同一 `checkpoint`、`max_tokens=128`、`bench_runs=6`，仅切换 `USE_NAIVE_MHA` |
| Qwen3-8B INT4（§5.3） | **NVIDIA RTX 3090（24GB）** | `max_tokens=128`，`bench_runs=5`，`quant_bits=4`，`prompt_tokens=128` |
| 困惑度（§5.4） | CUDA（与 `eval_awq_ppl.py` 一致） | WikiText-2，`fp16 / naive / awq_clip_outlier` 对比 |

### 3.2 Naive MHA vs Flash Attention（FP32，Qwen3-0.6B 量级）

| 指标 | Naive MHA | Flash Attention | 变化 |
|------|-----------|-----------------|------|
| Decode tok/s | 82.6 | 93.1 | +12.7% |
| Prefill tok/s | 95.2 | 105.1 | +10.4% |
| 首 Token 延迟 | 273 ms | 247 ms | −9.5% |
| 总时间 | 1836 ms | 1633 ms | −11.0% |
| MHA_Kernel 耗时 | 72.22 ms | 76.73 ms | +6.2% |
| 算子总耗时 | 3459 ms | 3018 ms | −12.8% |
| Peak 显存 | 4123.5 MB | 4123.5 MB | 持平 |

**简述**：端到端 decode / prefill 与首 token、总时间 Flash 更优；单独统计的 **MHA_Kernel** 时间 Flash 可能略高（归约与调度开销），但整体算子流水线更省；峰值显存几乎相同。

### 3.3 Qwen3-8B INT4（3090，`bench_qwen3` 见 §5.3）

| 指标 | 值 |
|------|-----|
| Prefill 吞吐 | 53.5 tok/s |
| Decode 吞吐 | 51.4 tok/s |
| 首 token 延迟 | 2671 ms |
| 峰值显存 | 18413 MB / 24124 MB（约 76%） |
| 单轮总耗时 | 约 5.1 s |

### 3.4 困惑度 PPL（`eval_awq_ppl.py`，与 §5.4 命令一致）

| 模式 | NLL | PPL |
|------|-----|-----|
| FP16（基线） | 2.3814 | 10.82 |
| naive（朴素 INT4） | 2.4314 | 11.37 |
| awq_clip_outlier | 2.3999 | 11.02 |

**简述**：朴素 INT4 相对 FP16 PPL 上升；**AWQ（含 clip + outlier）** 明显优于 naive，更接近 FP16。

---

## 四、环境与安装（跑通工程）

本节与课程根目录 **`readme.md`** 一致处：**第三方依赖列表、基本 `cmake`/`make` 流程**；与 **`CMakeLists.txt` / `cmake/cuda.cmake`** 一致处：**语言标准、Qwen3 开关、CUDA 架构检测**。个人分支在 Qwen3 上额外支持 **INT4 / AWQ** 导出与推理，见下文 **4.8** 与 **第五节**。

### 4.1 硬件与操作系统

| 项 | 说明 |
|----|------|
| **GPU** | **NVIDIA GPU + 已安装 CUDA Toolkit**；本工程 Qwen3 推理路径以 **CUDA** 为主（`qwen3_infer` / `bench_qwen3` 使用 `kDeviceCUDA`）。 |
| **显存（经验值）** | 小模型（如 0.6B）FP32 可在 **6GB** 级显卡上尝试；**8B FP32** 建议 **16GB+**（如 3090 24GB）；**INT4** 显著降低权重占用，仍以实际 `nvidia-smi` 为准。 |
| **OS** | **Linux** 或 **WSL2（Ubuntu）**（与课程一致）；云主机 3090 推荐原生 Linux。 |

### 4.2 系统级工具链

| 项 | 版本 / 说明 |
|----|-------------|
| **CMake** | **≥ 3.16**（与 `cmake_minimum_required` 一致） |
| **C++ 编译器** | **GCC 11+**（或等价，支持 **C++17**；`CMakeLists.txt` 中 `CMAKE_CXX_STANDARD 17`） |
| **CUDA / nvcc** | 与显卡驱动匹配的 **CUDA Toolkit**；工程中默认 `set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")`，若本机路径不同请在 **`CMakeLists.txt` 或 `cmake -DCMAKE_CUDA_COMPILER=...`** 中改正。 |
| **CUDA 语言标准** | 工程中为 **CUDA C++14**（`CMAKE_CUDA_STANDARD 14`）。 |

### 4.3 第三方依赖（与 `readme.md` 一致）

课程文档列出的依赖如下（**不开启 CPM 时需自行安装开发包**，并能被 `find_package` 找到）：

1. [glog](https://github.com/google/glog)  
2. [gtest](https://github.com/google/googletest)  
3. [sentencepiece](https://github.com/google/sentencepiece)  
4. [Armadillo](https://arma.sourceforge.net/download.html)（通常配合 **OpenBLAS** 等 BLAS）  
5. **CUDA Toolkit**

启用 **Qwen3**（或 LLaMA3 / Qwen2）时，`CMakeLists.txt` 还会链接 **absl、re2**（`find_package`），若使用 **CPM** 则与 **nlohmann_json** 等一并拉取。

**推荐（有网络时）**：与 `readme.md` 相同，开启 **`USE_CPM=ON`**，由 CPM 自动下载 **GTest、glog、Armadillo、sentencepiece**，并在 `QWEN3_SUPPORT` 等打开时拉取 **absl、re2、nlohmann_json**，减少系统包手工对齐成本。

### 4.4 CMake 工程选项（`CMakeLists.txt`）

| 选项 | 默认 | 含义 |
|------|------|------|
| `USE_CPM` | OFF | ON：用 CPM 管理上述 C++ 依赖 |
| `QWEN3_SUPPORT` | OFF | ON：编译 Qwen3 相关代码（**跑 `qwen3_infer` / `bench_qwen3` 必须打开**） |
| `LLAMA3_SUPPORT` / `QWEN2_SUPPORT` | OFF | 课程其它模型路径，与本说明 Qwen3 无冲突、按需开启 |
| `USE_NAIVE_MHA` | OFF | ON：Naive MHA；OFF：**Flash Attention**（见第二节） |
| `ENABLE_NVTX` | OFF | ON：链接 `nvToolsExt`，供 nsys 打点（需库存在） |

### 4.5 CUDA 架构（`cmake/cuda.cmake`）

- 配置阶段若检测到 CUDA，会通过 **`CUDA_DETECT_INSTALLED_GPUS`** 自动设置 **`CMAKE_CUDA_ARCHITECTURES`**（与当前机器上的 GPU 计算能力一致）。
- **无显卡配置工程**（如仅在一台无 GPU 的机器上生成 build 目录）时，自动检测可能不符合预期，可手动指定，例如：  
  `cmake -B build -S . ... -DCMAKE_CUDA_ARCHITECTURES=86`  
  其中 **86** 对应常见 **Ampere（如 RTX 3090 / 3060）**；**89** 对应部分 **Ada**，请以 [NVIDIA 官方 compute capability 表](https://developer.nvidia.com/cuda-gpus) 为准。

### 4.6 Qwen3 模型从哪里来、目录里有什么

课程 **`readme.md`** 中 **Qwen3** 说明：从 **Hugging Face** 将完整模型下载到本地目录。可与 Qwen2.5 类似使用镜像（示例）：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 可选：国内镜像
pip3 install huggingface-cli
huggingface-cli download --resume-download <HF 上的 Qwen3 仓库名> --local-dir ./Qwen3-XXX --local-dir-use-symlinks False
```

（将 `<HF 上的 Qwen3 仓库名>` 换成实际 ID，例如课程或镜像站提供的 **Qwen3-0.6B / 8B** 等；注意权重体积分卷与磁盘空间。其它系列如 **LLaMA / Qwen2.5** 的下载与导出见根目录 **`readme.md`**。）

**运行 C++ 推理至少需要：**

| 文件 / 目录 | 用途 |
|-------------|------|
| **权重文件 `*.bin`** | C++ 可读的单文件 checkpoint：课程流程为 **FP32 `write_bin.py` 产物**；INT4/AWQ 分支为 **`export_int4.py` / `export_int4_awq.py` 产物** |
| **`tokenizer.json`** | Qwen3 BPE 分词器，路径一般为 **`模型目录/tokenizer.json`**，作为 `qwen3_infer` / `bench_qwen3` 的第二个参数 |

导出脚本通常还要求同目录下存在 **`config.json`**（及 HF 的 **`model.safetensors`** 或分片权重，视脚本而定）。

### 4.7 FP32 权重导出（与课程 `readme.md` 两步一致）

课程路径：**`tools/export_qwen3/load.py`** 导出 **`.pth`**，再 **`write_bin.py`** 导出 **`qwen.bin`**。

1. **HF → `.pth`**（在 `tools/export_qwen3` 下，需 **Python：`torch`、`transformers`、`tqdm`** 等）：

```bash
cd tools/export_qwen3
python load.py --model_name /path/to/Qwen3-XXX --output_file qwen3_weights.pth
```

2. **`.pth` → C++ 用 `.bin`**：

```bash
python write_bin.py -p qwen3_weights.pth -o Qwen3-XXX.bin -n /path/to/Qwen3-XXX -d cpu
```

`-n` 指向 **与 HF 一致的模型目录**（用于 `config` / tokenizer 等元数据与脚本内逻辑）。得到的 **`Qwen3-XXX.bin`** + **`/path/to/Qwen3-XXX/tokenizer.json`** 即可用于：

```bash
./build/demo/qwen3_infer Qwen3-XXX.bin /path/to/Qwen3-XXX/tokenizer.json
```

### 4.8 Python 环境（INT4 / AWQ 与可选评估）

| 依赖 | 用途 |
|------|------|
| `torch` | FP32 导出链、`export_int4.py` / `export_int4_awq.py`、校准 |
| `safetensors` | `export_int4.py` 从 HF 读权重 |
| `transformers` | `load.py` / `write_bin.py` 链、`export_int4_awq.py` 加载 FP16 模型与激活标定 |
| `tqdm` | `load.py` 进度条 |
| `datasets` | 可选：`eval_awq_ppl.py`、`export_int4_awq.py` 中 `wikitext2` 校准 |

**AWQ 导出**建议在 **单卡 GPU** 上执行（`--device cuda`），显存需容纳 **FP16 校准**用模型（如 8B 常需 **16GB+**）。

### 4.9 一键编译摘要（Qwen3 + CPM）

与 `readme.md` 的「编译方法」一致，仅多 **`QWEN3_SUPPORT`**：

```bash
cd /path/to/KuiperLLama
cmake -B build -S . -DUSE_CPM=ON -DQWEN3_SUPPORT=ON
cmake --build build --target bench_qwen3 qwen3_infer -j$(nproc)
```

无 CPM 时：先按 **4.3** 安装系统级 glog、gtest、Armadillo、sentencepiece，再 `cmake -B build -S . -DQWEN3_SUPPORT=ON`（不开 `USE_CPM`）。

---

## 五、编译与运行

### 5.1 CMake 基础配置

```bash
cd /path/to/KuiperLLama
# Flash Attention（默认，不定义 USE_NAIVE_MHA）
cmake -B build -S . -DUSE_CPM=ON -DQWEN3_SUPPORT=ON
cmake --build build --target bench_qwen3 qwen3_infer -j$(nproc)
```

- **Naive MHA**：配置时追加 `-DUSE_NAIVE_MHA=ON`（见 §5.2）。
- **NVTX**：追加 `-DENABLE_NVTX=ON`（需系统存在 `libnvToolsExt`）。

### 5.2 实验 A：Naive vs Flash MHA（FP32，与 §3.2 对应）

同一权重与 tokenizer，仅切换 CMake 中的 **`USE_NAIVE_MHA`**，bench 参数示例：`max_tokens=128`，`bench_runs=6`（后两位为 FP32 默认，可省略 `quant_bits` / `prompt_tokens` / `awq`）。

**1）Naive MHA（无 Flash）**

```bash
cd /path/to/KuiperLLama
cmake -B build -S . -DUSE_CPM=ON -DQWEN3_SUPPORT=ON -DUSE_NAIVE_MHA=ON
cmake --build build --target bench_qwen3 -j8
./build/demo/bench_qwen3 <checkpoint_fp32.bin> <tokenizer.json> 128 6
```

**2）Flash Attention**

```bash
cd /path/to/KuiperLLama
cmake -B build -S . -DUSE_CPM=ON -DQWEN3_SUPPORT=ON -DUSE_NAIVE_MHA=OFF
cmake --build build --target bench_qwen3 -j8
./build/demo/bench_qwen3 <checkpoint_fp32.bin> <tokenizer.json> 128 6
```

作者曾用路径示例（请替换为本机）：`.../qwen3-0.6b/qwen0.6.bin2` 与同级 `tokenizer.json`。若 Naive / Flash 交替编译较慢，可使用 **§5.7** 的双目录方案。

### 5.3 实验 B：INT4 导出与 bench（与 §3.3 对应）

**导出**：`tools/export_qwen3/export_int4.py` 从 HuggingFace 目录读取 **`*.safetensors`** + **`config.json`**，写出 **Q4_0** 单文件 `.bin`（无 `awq_flag` 段）。

```bash
cd /path/to/KuiperLLama/tools/export_qwen3
pip install torch safetensors
python export_int4.py --model_dir /path/to/Qwen3-8B --group_size 64 \
  --output /path/to/Qwen3-8B-int4.bin
```

省略 `--output` 时，默认在当前工作目录生成 `Qwen3-<model_dir 目录名>-int4.bin`。

**Bench（朴素 INT4，`awq` 为 0，可省略末位）**：参数顺序为  
`<checkpoint> <tokenizer> <max_tokens> <bench_runs> <quant_bits> <prompt_tokens> [awq]`。

```bash
cd /path/to/KuiperLLama
./build/demo/bench_qwen3 /path/to/Qwen3-8B-int4.bin /path/to/Qwen3-8B/tokenizer.json \
  128 5 4 128
```

作者环境示例：`./build/demo/bench_qwen3 .../Qwen3-8B-int4.bin .../Qwen3-8B/tokenizer.json 128 5 4 128`。

推理时 **`awq` 必须为 0**（与 `export_int4.py` 产物一致）。

### 5.4 实验 C：AWQ 导出、C++ 运行与困惑度（与 §3.4 对应）

**导出 INT4 + AWQ**（需 GPU、`transformers` 等；与 `awq_clip_outlier` 评估对齐时可加 **`--outlier_ratio 0.01`**，对应 C++ 侧 `awq_flag=2`）：

```bash
cd /path/to/KuiperLLama/tools/export_qwen3
pip install torch safetensors transformers datasets
python export_int4_awq.py --model_dir /path/to/Qwen3-8B --group_size 64 \
  --device cuda --calibration wikitext2 \
  --outlier_ratio 0.01 \
  --output /path/to/Qwen3-8B-int4-awq.bin
```

纯 AWQ、不要 outlier 时设 `--outlier_ratio 0` 或省略该参数（以脚本默认为准）。

**C++ 交互推理（INT4 + AWQ，最后一参 `1`）**：

```bash
./build/demo/qwen3_infer /path/to/Qwen3-8B-int4-awq.bin /path/to/Qwen3-8B/tokenizer.json 4 1
```

**C++ bench（INT4 + AWQ）**：在 §5.3 命令末尾再加 **`1`**（第七个参数）。

```bash
./build/demo/bench_qwen3 /path/to/Qwen3-8B-int4-awq.bin /path/to/Qwen3-8B/tokenizer.json \
  128 5 4 128 1
```

**困惑度（PyTorch 侧复现各模式，与 C++ `.bin` 互补）**：

```bash
cd /path/to/KuiperLLama/tools/export_qwen3
python eval_awq_ppl.py \
  --model_dir /path/to/Qwen3-8B \
  --modes fp16,naive,awq_clip_outlier \
  --max_tokens 8192 \
  --calibration wikitext2 \
  --wikitext_calib_samples 128 \
  --cache_awq /tmp/awq_cache_v2.pt \
  --outlier_ratio 0.01 \
  --device cuda
```

### 5.5 交互推理 `qwen3_infer`（参数小结）

```bash
# FP32
./build/demo/qwen3_infer <checkpoint> <tokenizer.json>

# INT4 朴素（export_int4.py）
./build/demo/qwen3_infer <checkpoint-int4.bin> <tokenizer.json> 4

# INT4 + AWQ（export_int4_awq.py）
./build/demo/qwen3_infer <checkpoint-int4-awq.bin> <tokenizer.json> 4 1
```

`quant_bits`：`0`=FP32，`4`=INT4；末位 `awq`：`1` 仅当权重文件含 AWQ 头与 `awq_input_scales`（与导出一致）。

### 5.6 Benchmark `bench_qwen3`（参数小结）

```text
bench_qwen3 <checkpoint> <tokenizer> [max_tokens] [bench_runs] [quant_bits] [prompt_tokens] [awq]
```

| 位置 | 含义 | 默认 |
|------|------|------|
| argv[5] | `quant_bits` | `0`（FP32） |
| argv[6] | `prompt_tokens` | `0`：内置长 prompt；`>0`：扩展到约该 token 数 |
| argv[7] | `awq` | `0`；AWQ `.bin` 用 `1` |

其它示例：长跑 bench 可用  
`./build/demo/bench_qwen3 <ckpt> <tok> 512 10 4 0 1`（INT4-AWQ，内置长 prompt）。

### 5.7 双 build 目录（减少 Naive / Flash 切换重编译）

```bash
cmake -B build_flash -S . -DUSE_CPM=ON -DQWEN3_SUPPORT=ON -DUSE_NAIVE_MHA=OFF
cmake -B build_naive -S . -DUSE_CPM=ON -DQWEN3_SUPPORT=ON -DUSE_NAIVE_MHA=ON
```

---

## 六、仓库与分支说明

- **课程原仓库**：见根目录 `readme.md`。
- **个人改造**：可按面试需要切换分支演示 **Flash+Benchmark** 与 **INT4+AWQ**；二者均基于同一 Qwen3 推理骨架，量化分支额外包含 `model.cpp` 头解析、`qwen3.cpp` 量化层加载、`matmul` + CUDA **INT4 kernel** 与 **AWQ scales** 路径。

---

## 七、核心文件索引（Code Review）

| 内容 | 路径 |
|------|------|
| Flash / Naive MHA | `kuiper/source/op/kernels/cuda/mha_kernel.cu` |
| INT4 MatMul + AWQ | `kuiper/source/op/matmul.cpp`、`kuiper/source/op/kernels/cuda/matmul_kernel.cu`、`kuiper/source/op/kernels/kernels_interfaces.cpp` |
| 量化权重布局 / mmap | `kuiper/source/model/model.cpp`（`read_model_file`）、`kuiper/source/model/qwen3.cpp`（`create_param_quant_layers`） |
| Layer 权重 INT4 / AWQ | `kuiper/source/op/layer.cpp`（`set_weight` INT4、`set_awq_scales`） |
| CMake | `CMakeLists.txt`（`USE_NAIVE_MHA`、`ENABLE_NVTX`、`QWEN3_SUPPORT`） |
| Profiler | `kuiper/include/base/profiler.h` |
| Benchmark | `demo/bench_qwen3.cpp` |
| 推理入口 | `demo/main_qwen3.cpp` |
| Qwen3 FP32：`HF → .pth` | `tools/export_qwen3/load.py` |
| Qwen3 FP32：`.pth → .bin` | `tools/export_qwen3/write_bin.py` |
| 导出 INT4 | `tools/export_qwen3/export_int4.py` |
| 导出 INT4+AWQ | `tools/export_qwen3/export_int4_awq.py` |
| PPL 对比 | `tools/export_qwen3/eval_awq_ppl.py` |

---

*文档版本：覆盖 Flash/Benchmark 与 INT4/AWQ 分支；性能与显存以本机 `logs/`、`nvidia-smi` 及 bench 输出为准。*
