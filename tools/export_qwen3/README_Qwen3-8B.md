# Qwen3-8B 导出流程

模型已下载到 `/home/dzy/models/Qwen3-8B`，按以下步骤导出并运行。

## 步骤 1：将 HuggingFace 模型转为 .pth

```bash
cd /home/dzy/KuiperLLama/tools/export_qwen3
python load.py --model_name /home/dzy/models/Qwen3-8B --output_file qwen3_8b_weights.pth
```

（若显存不足可加 `--device cpu`，会较慢）

## 步骤 2：将 .pth 转为 C++ 推理用的 .bin

```bash
python write_bin.py -p qwen3_8b_weights.pth -o Qwen3-8B.bin -n /home/dzy/models/Qwen3-8B
```

`config.json` 会自动读取，无需手动改 `config.py`。

## 步骤 3：运行推理

```bash
cd /home/dzy/KuiperLLama
./build/demo/qwen3_infer Qwen3-8B.bin /home/dzy/models/Qwen3-8B/tokenizer.json
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `-p, --checkpoint` | 输入的 .pth 文件 |
| `-o, --output` | 输出的 .bin 文件 |
| `-n, --model_name` | 模型目录（含 config.json、tokenizer.json） |
| `-d, --device` | 加载设备（cpu/cuda） |

## 其他规模模型

换用其他 Qwen3 模型时，只需修改 `--model_name` 和 `-n` 为对应目录，`config.json` 会自动解析。
