# DiT 差分加速探索（对照 Cambricon-D）

这个目录是一个**独立的测试**，跟主项目（mini-wam）解耦，用来验证
Cambricon-D 论文里"差分计算（differential computing）"的思路，能否加速我们
的 DiT 推理。

## 背景

Cambricon-D 的核心洞察：扩散/流匹配模型跨时间步迭代时，**输入只发生微小变化**，
因此不必每步重算整个网络，而是算**差值 Δ** 并让它传播：

- 线性层（matmul）：`ΔY = W·ΔX`（精确，因为 matmul 是线性的）
- 非线性层（ReLU/LayerNorm/softmax/GELU/SiLU）：会阻断 Δ 传播，论文用
  **sign-mask 近似** `ΔY' = ΔY · sgn(Y_{t-1})`（正确率 ~99.6%）

**关键**：论文的加速来自**硬件**（低比特 Δ 乘加阵列 + sign-mask 特殊单元 +
近存处理 NDP），不是纯软件算法。

## 本目录做什么

1. `dit_block.py` —— 从主项目复制的 DiTBlock（对照组）。
2. `differential_dit.py` —— 精确差分版 DiTBlock（线性层走 Δ 路径，非线性层 merge）。
3. `benchmark.py` —— 验证一致性 + 测时间 + 测 Δ 稀疏度。

**首要目标**：差分版输出必须与对照组一致（bit-exact），然后才测时间差。
