# Fast_Infer 项目工作总结

## 一、项目概况

Fast_Infer 是一个从零构建的、基于 Triton 的 LLM 推理加速框架，目标是让 **Llama 3.2 3B Instruct** 模型在 **RTX 5060 8GB** 消费级 GPU 上高效运行。权重从 ModelScope（魔搭社区）下载，通过 INT4 分组量化压缩至 8GB 显存以内。

**最终成果：**

| 方法 | 显存 | 速度 (256 tokens) |
|------|------|-------------------|
| HuggingFace Transformers FP16 | 6.00 GB | 39.3 t/s |
| **Fast_Infer INT4** | **2.55 GB** | **41.8 t/s** |
| Fast_Infer FP16 | 6.44 GB | ~31 t/s |

INT4 模式在速度上反超 HF 基准约 6%，同时显存占用不到一半。

---

## 二、框架设计思路

### 2.1 总体架构

```
fast_infer/
├── config.py          # LlamaConfig 数据类（模型超参 + 推理默认值）
├── download.py        # ModelScope snapshot_download
├── loader.py          # Safetensors 加载 + INT4 分组量化
├── kernels/
│   ├── rms_norm.py    # RMSNorm Triton 内核
│   ├── rope.py        # RoPE（PyTorch 实现，内存带宽受限）
│   ├── matmul.py      # 融合 INT4 反量化+矩阵乘法 + FP16 Triton 矩阵乘法
│   ├── mlp.py         # SwiGLU MLP（融合 SiLU Triton 内核）
│   └── attention.py   # FlashAttention v2（预填充）+ PagedAttention（解码）
├── kv_cache.py        # 分块 KV 缓存（PagedAttention 风格，block_size=64）
├── model.py           # LlamaForCausalLM（28 层，无 nn.Module）
├── sampler.py         # Temperature + Top-p 核心采样 + Repetition Penalty
├── engine.py          # InferenceEngine：加载模型、分词、生成
└── main.py            # 命令行交互式对话
```

### 2.2 核心设计决策

#### 不使用 `torch.nn.Module`

模型是纯 Python 类，将权重作为 dict 管理，直接调用 Triton 内核。这样做的原因：
- 避免框架开销（`nn.Module` 的参数注册、钩子系统等）
- 对 INT4 打包权重（`(packed_int8, scales)` 元组）的存储更加灵活
- 用显式的 `dict` 结构而非嵌套 `nn.Module`，数据流更加透明

#### 双模式策略

两种模式共享同一份代码路径，在初始化时通过 `memory_efficient` 参数切换：

```
memory_efficient=False (速度模式):
  加载 → INT4 量化 → 立即反量化为 FP16 → 释放 INT4 内存
  推理时: FP16 Triton 矩阵乘法

memory_efficient=True (INT4 模式):
  加载 → INT4 量化 → 保留打包格式
  推理时: 融合 INT4 反量化+矩阵乘法 Triton 内核
```

设计理念：INT4 不仅仅是为了节省显存——通过减少权重的内存带宽占用（读取 int8 打包数据 + fp16 scale，而非完整 fp16 权重），反量化矩阵乘法在 decode 中比 FP16 矩阵乘法更快。

#### 融合投影权重以减少内核启动次数

从 QKV 和 Gate+Up 的融合中获得了一些关键的优化成果：

```
每层内核启动次数:
  融合前: Q, K, V, O, gate, up, down = 7 次矩阵乘法
  融合后: QKV, O, gate_up, down = 4 次矩阵乘法
  节省: 每层 3 次 × 28 层 = 84 次内核启动
```

对于 INT4 模式，这在张量保持为 `(packed_int8, scales)` 元组时实现——分别在 dim=0 上拼接 packed 和 scales 即可产生一个融合后的权重元组。

#### RoPE 频率表

为了避免满布 131072 个位置（`max_position_embeddings`），改为仅预计算 `max_seq_len`（4096）个位置。节省了约 65 MB 显存且不影响正确性。

将 RoPE 改为 Triton 内核的方案也经过了尝试和调试。遇到了一个微妙的坑：Q 和 K 张量在模型中是来自拼接后的 QKV 矩阵乘法输出的非 contiguous 视图（由于 `.split()` + `.view()`）。修复方法是对 Triton 内核改用正确的步长（stride），或者直接调用 `.contiguous()`。最终重新采用 PyTorch 实现，因为该操作受内存带宽限制，Triton 实现并没有优势。

### 2.3 数据流

**预填充阶段：**
```
embed[input_ids] → 28× 解码层 (
    RMSNorm → QKV 投影 → RoPE → FlashAttention v2 → O 投影
    → RMSNorm → SwiGLU MLP
) → 最终 RMSNorm → lm_head → logits
```
KV 缓存在此阶段被填充。

**解码阶段：**
```
单 token → 相同流程，但使用 PagedAttention 从缓存中读取 KV 块
```

### 2.4 内核设计

| 内核 | 实现方式 | 关键技术 |
|------|----------|----------|
| **FlashAttention v2**（预填充） | Triton | 在线 softmax，因果掩码，分块 Q/K/V |
| **PagedAttention**（解码） | Triton | 在线 softmax，逻辑到物理块映射，GQA 路由（无 materialized expansion） |
| **INT4 反量化+矩阵乘法** | Triton | 即时半字节解包，分组 scale 加载，FP32 累加 |
| **FP16 矩阵乘法** | Triton | 分块 GEMM，L2 缓存优化（GROUP_M=8） |
| **RMSNorm** | Triton | FP32 累加，单次数据遍历 |
| **SwiGLU** | Triton | 融合 SiLU + multiply |

---

## 三、测试结构

### 3.1 基准测试（`benchmark.py`）

对比两种模式的端到端性能测试：

```bash
python benchmark.py                    # HF + 两种 Fast_Infer 模式对比
python benchmark.py --skip-hf          # 仅 Fast_Infer（避免 HF OOM）
python benchmark.py --max-tokens 128   # 更长的生成
```

指标追踪：tokens/s、峰值 VRAM、输出一致性。

### 3.2 正确性测试（手动）

**确定性 Greedy 解码测试：**
- 固定 prompt：`"The capital of France is"`
- Temperature=0（greedy），比较输出与参考值
- 期望输出：`"The capital of France is Paris..."`
- 在 FP16 和 INT4 两种模式下均需完全匹配

**逐层差异测试：**
- 将 PyTorch RoPE 输出与 Triton RoPE 输出在每一层中进行对比
- FP16 精度下每元素最大差异 < 0.01 即为通过

**内核单元测试：**
- RMSNorm：对比 `torch.nn.functional.rms_norm`
- RoPE：对比纯 PyTorch 实现（`x*cos + rotate_half(x)*sin`）
- 矩阵乘法：`x @ w.T` 结果对比

### 3.3 当前测试覆盖的局限性

1. **没有自动化测试套件。** 所有测试均为手动执行。回归测试完全依赖开发者纪律。
2. **没有与 HF 输出的数值对比测试。** 验证是通过检查输出是否连贯（而非精确对比 token ID）完成的。
3. **没有压力测试。** 没有长序列、大批量、边界条件（如 prompt 为空、最大长度 token 等）。
4. **内核正确性仅做了点级对比。** 没有进行注意力模式的对比、KV 缓存一致性的对比，也没有在 28 层全部执行完毕得到的最终 logits 层面进行对比。

### 3.4 推荐的测试补充

1. 添加与 HF greedy 解码输出的确定性对比（固定种子，逐层对比 logits）
2. 将测试集设为 3-5 个 prompt（短、中、长），避免某一种情况过度拟合
3. 针对 INT4 量化误差进行统计（与 FP16 参考值之间的 MSE）
4. 在正式版推理运行之前加入轻量级冒烟测试

---

## 四、性能优化历程

| 阶段 | INT4 速度 | 变更说明 |
|------|-----------|----------|
| 初始 | 3.1 t/s | 基础 INT4 内核，无融合 |
| QKV 融合 | 43.0 t/s | INT4 模式下 Q/K/V 拼接为单一投影（+1288%） |
| Gate+Up 融合 | ~41 t/s | MLP 门控+上投影融合（边际提升，减少内核启动） |
| RoPE 频率表 | ~42 t/s | 预计算从 131K 缩减至 4K 位置（节省 65 MB 显存） |

**主要收益来自 QKV 融合**——减少内核启动是最大的杠杆。

### 4.1 未成功的尝试

**Triton RoPE 内核：** 编写完成、调试完成、在单元级别验证正确。但在集成时发现 Q/K 来自拼接后的 QKV 输出，因此是非 contiguous 的，需要额外的内存拷贝。由于 RoPE 受内存带宽限制，Triton 实现没有任何优势。已恢复为 PyTorch 版本。

**`torch.compile`（CUDA 图）：** `mode='reduce-overhead'` 因可变 KV 缓存状态导致"tensor output overwritten"错误而失败。`mode='default'` 在 FP16 模式下有效，但在 INT4 模式下会触发严重的重新编译（速度从 42 t/s 跌至 3 t/s），因为 dynamo 无法很好地追踪自定义 Triton 内核。

---

## 五、模型配置

```
hidden_size=3072, num_hidden_layers=28, num_attention_heads=24
num_key_value_heads=8, head_dim=128, intermediate_size=8192
vocab_size=128256, rope_theta=500000, max_position_embeddings=131072
```

分组查询注意力（GQA）：24 个 Q 头 / 8 个 KV 头 = 每组 3 个查询头。

---

## 六、环境与运行

```bash
# 环境
source fast_infer_env/bin/activate   # Python 3.12, requirements.txt

# 下载模型
python -m fast_infer.download

# 推理
python -m fast_infer.main --model-dir <path-to-model-dir>
python -m fast_infer.main --temperature 0.6 --top-p 0.9 --max-tokens 256
```
