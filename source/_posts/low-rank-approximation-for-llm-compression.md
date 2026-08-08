---
title: "Low-Rank Approximation for LLM Compression: Mathematical Foundations"
date: 2026-08-08
mathjax: true
tags: [math, LLM, compression, SVD, ASVD]
---

# 低秩逼近用于 LLM 压缩：数学推导

## 1. 问题设定

一个 transformer 的线性层做的事情是矩阵乘法：

$$y = Wx, \quad W \in \mathbb{R}^{m \times n}$$

比如 `gate_proj` 的 $W$ 是 $12288 \times 4096$，把 4096 维的隐状态映射到 12288 维的中间表示。

**目标**：用更少的参数近似 $W$。如果能找到 $A \in \mathbb{R}^{m \times r}$，$B \in \mathbb{R}^{r \times n}$（$r \ll \min(m,n)$），使得 $AB \approx W$，那参数量从 $mn$ 降到 $(m+n)r$。

例如 gate_proj：$12288 \times 4096 = 50.3M$ 参数 → rank-384 时 $(12288+4096)\times 384 = 6.29M$，压缩 8 倍。

## 2. Plain SVD：最基础的低秩逼近

### 2.1 目标函数

最朴素的想法：让 $AB$ 在 Frobenius 范数下尽可能接近 $W$：

$$\min_{\text{rank}(\hat W) \le r} \|W - \hat W\|_F^2 = \min_{\text{rank}(\hat W) \le r} \sum_{j,i} (W_{ji} - \hat W_{ji})^2$$

### 2.2 SVD 分解

对 $W$ 做奇异值分解：

$$W = U \Sigma V^T = \sum_{k=1}^{\min(m,n)} \sigma_k u_k v_k^T$$

其中 $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ 是奇异值，$u_k$ 和 $v_k$ 分别是左/右奇异向量。

### 2.3 Eckart-Young 定理

**最优 rank-$r$ 逼近是截断 SVD**：

$$\hat W = \sum_{k=1}^{r} \sigma_k u_k v_k^T = U_r \Sigma_r V_r^T$$

最小误差为：

$$\|W - \hat W\|_F^2 = \sum_{k=r+1}^{\min(m,n)} \sigma_k^2$$

即被丢掉的奇异值的平方和。这是**所有** rank-$r$ 矩阵中的最优解——不存在更好的 rank-$r$ 逼近。

### 2.4 分解为 A、B

取：

$$A = U_r \sqrt{\Sigma_r} \in \mathbb{R}^{m \times r}, \quad B = \sqrt{\Sigma_r} V_r^T \in \mathbb{R}^{r \times n}$$

则 $AB = U_r \Sigma_r V_r^T = \hat W$。推理时计算 $y = ABx = A(Bx)$：先做 $n \to r$，再做 $r \to m$，两次小矩阵乘法。

### 2.5 问题

Plain SVD 最小化的是**权重误差** $\|W - AB\|_F$——对 $W$ 所有元素等权。但实际使用时，我们关心的是**输出误差** $\|Wx - ABx\|$，而输入 $x$ 不是均匀分布的。

## 3. ASVD（Activation-aware SVD）：考虑输入分布

### 3.1 真正要最小化的

我们实际关心的是输出误差在数据分布上的期望：

$$\min_{A,B} \mathbb{E}_{x \sim \mu}\big[\|Wx - ABx\|^2\big]$$

展开（利用迹的循环性和期望的线性性）：

$$\mathbb{E}\|(W-AB)x\|^2 = \mathbb{E}\big[\text{tr}\big((W-AB)xx^T(W-AB)^T\big)\big] = \text{tr}\big((W-AB)\,\Sigma\,(W-AB)^T\big)$$

其中 $\Sigma = \mathbb{E}[xx^T] \in \mathbb{R}^{n \times n}$ 是输入的（未中心化的）协方差矩阵。

### 3.2 变量代换

设 $\Sigma$ 有分解 $\Sigma = S S^T$（比如 $S = \Sigma^{1/2}$ 或 Cholesky 分解）。则：

$$\text{tr}\big((W-AB)\,\Sigma\,(W-AB)^T\big) = \|(W-AB)S\|_F^2$$

所以原问题等价于：

$$\min_{\text{rank}(\hat W) \le r} \|(W - \hat W)S\|_F^2$$

再做代换 $\tilde W = WS$：

$$= \min_{\text{rank}(\hat W) \le r} \|\tilde W - \hat W S\|_F^2$$

注意 $\hat W S$ 的秩仍然 $\le r$（因为 $\hat W$ 的秩 $\le r$），所以这等价于对 $\tilde W = WS$ 做标准的 rank-$r$ 最优逼近，由 Eckart-Young 定理解出。

### 3.3 完整求解过程

1. 计算 $\tilde W = WS$
2. 对 $\tilde W$ 做 SVD：$\tilde W = U \Sigma_{\tilde W} V^T$
3. 截断：$\tilde W_r = U_r \Sigma_r V_r^T$
4. 还原：$\hat W = \tilde W_r S^{-1}$，即 $A = U_r \sqrt{\Sigma_r}$，$B = \sqrt{\Sigma_r} V_r^T S^{-1}$

验证：$AB = U_r \Sigma_r V_r^T S^{-1}$，所以 $(W - AB)S = WS - U_r \Sigma_r V_r^T = \tilde W - \tilde W_r$，正是 SVD 截断的残差，Eckart-Young 最优。

### 3.4 两种 $S$ 的选择

#### Whiten ASVD（精确解）

取 $S = \Sigma^{1/2}$（或 $S = \text{chol}(\Sigma)$），精确保留完整协方差结构。

- **优点**：精确最小化 $\mathbb{E}\|Wx - ABx\|^2$，是理论最优
- **缺点**：需要存/计算完整的 $n \times n$ 协方差矩阵（4096×4096 = 64MB/层），加上 Cholesky 分解

#### Diag ASVD（对角近似）

取 $S = \text{diag}(s_1, \ldots, s_n)$，其中 $s_i = (\mathbb{E}[|x_i|])^\alpha$。

这是把 $\Sigma$ 近似为对角矩阵——**只保留各通道的边际统计量，忽略通道间的相关性**。

- **优点**：只需要一个 $n$ 维向量（16KB/层），计算极快
- **缺点**：忽略了协方差的非对角部分
- **实测**：diag（α=1.0）给出 val loss 8.91，whiten 给出 8.64——差距仅 0.27，说明对角近似已经足够好

## 4. Diag ASVD 的详细推导

### 4.1 动机

设输入 $x$ 的各通道方差差异很大（LLM 中确实如此，某些通道激活值比其他通道大几万倍）。$s_i = \mathbb{E}[|x_i|]^\alpha$ 作为 $\sqrt{\text{Var}(x_i)}$ 的近似（对对称分布，$\mathbb{E}[|x|] \propto \sqrt{\text{Var}(x)}$）。

### 4.2 加权矩阵

$$\tilde W = W \cdot \text{diag}(s) \quad \Rightarrow \quad \tilde W_{ji} = W_{ji} \cdot s_i$$

物理含义：$W$ 的第 $i$ 列乘以 $s_i$。如果通道 $i$ 的激活量大（$s_i$ 大），这一列被放大 → SVD 会优先保真；如果 $s_i$ 小，这一列被缩小 → 可以牺牲。

### 4.3 SVD + 还原

对 $\tilde W$ 做 SVD 并截断到 rank $r$：

$$\tilde W = U\Sigma V^T, \quad \tilde W_r = U_r \Sigma_r V_r^T$$

还原（除以 $s$）：

$$A = U_r \sqrt{\Sigma_r}, \quad B = \sqrt{\Sigma_r} V_r^T \cdot \text{diag}(s)^{-1}$$

验证输出：$(AB)x = U_r \Sigma_r V_r^T \text{diag}(s)^{-1} x$

### 4.4 $\alpha$ 的作用

$s_i = (\mathbb{E}[|x_i|])^\alpha$。$\alpha$ 控制加权的强度：

- $\alpha = 0$：不加权，退化为 plain SVD
- $\alpha = 0.5$：ASVD 论文默认值
- $\alpha = 1.0$：我们实测最优——直接用激活量均值
- $\alpha > 1$：过度加权高激活通道，可能忽略其他方向

实测 Qwen3-8B（rank=384，uniform）：

| $\alpha$ | 初始 val loss |
|---|---|
| 0.25 | 18.69（≈ plain SVD） |
| 0.5 | 10.83 |
| 0.75 | 9.35 |
| **1.0** | **8.91** |
| 1.25 | 9.11 |
| 1.5 | 10.09 |
| 2.0 | 11.79 |

$\alpha$ 有一个阈值效应：0.25 几乎等于不加权（差距只有 0.04），但从 0.5 开始急剧改善。最优在 1.0。

## 5. 三者关系的直观理解

|   | 目标 | 等价于 |
|---|---|---|
| Plain SVD | $\min \|W - AB\|_F^2$ | 假设 $x \sim \mathcal{N}(0, I)$（各向同性） |
| Diag ASVD | $\min \|(W-AB)\text{diag}(s)\|_F^2$ | 假设 $x$ 各分量独立但方差不同 |
| Whiten ASVD | $\min \|(W-AB)\Sigma^{1/2}\|_F^2$ | 精确假设 $x \sim \mathcal{N}(0, \Sigma)$ |

从 plain 到 diag 到 whiten，逐步引入更多关于数据分布的信息：

- Plain：什么都不知道，平等对待所有方向
- Diag：知道每个通道有多重要（边际统计量）
- Whiten：知道通道之间的完整相关结构

## 6. 逼近的理论极限：Kolmogorov 宽度

给定任何 $S$，ASVD 给出的是**该加权范数下的最优** rank-$r$ 逼近。最优误差为：

$$\inf_{\text{rank}(\hat W) \le r} \|(W - \hat W)S\|_F^2 = \sum_{k=r+1}^{\min(m,n)} \sigma_k^2(WS)$$

这是 **Kolmogorov $r$-宽度**（对矩阵集合在加权 Frobenius 范数下的版本）。它是**与算法无关的下界**——不可能存在任何方法做得比 SVD 截断更好。

如果 $\sigma_k(WS)$ 衰减快（前几个奇异值占据大部分能量），低秩逼近很好；如果衰减慢，再好的算法也无能为力。

我们实测 Qwen3-8B 的情况：

| 矩阵 | rank=384 加权能量保留 | 含义 |
|---|---|---|
| L35 down_proj | **98.5%** | 几乎满秩可恢复——因为 massive activations 让少数方向主导 |
| L0 gate_proj | **32.5%** | 丢掉 2/3 的信息——因为浅层激活分布均匀 |
| L0 up_proj | **25.1%** | 丢掉 3/4——最难压缩 |

## 7. 为什么单矩阵最优不等于端到端最优

ASVD 保证每个矩阵在其加权范数下最优，但模型的 val loss 取决于**所有层的复合映射** $F = f_L \circ \cdots \circ f_1$。

设每层误差为 $\varepsilon_\ell = f_\ell - \tilde f_\ell$，则端到端误差：

$$F - \tilde F = \sum_\ell (\partial f_{>\ell})\,\varepsilon_\ell + O(\varepsilon^2)$$

这里 $\partial f_{>\ell}$ 是下游层对误差的放大/缩小。如果各层误差的方向恰好互相抵消（$\langle \partial f_{>\ell}\varepsilon_\ell,\, \partial f_{>k}\varepsilon_k \rangle < 0$），那端到端误差比各层误差之和小——我们实测这个亚线性因子为 **1.64×**。

**逐矩阵优化不能利用这个相消结构**。而端到端蒸馏（联合优化所有 $A_\ell, B_\ell$）可以——这就是为什么它能把 8.50 降到 3.79。

## 8. 总结

| 方法 | 数学保证 | 实测 val loss | 局限 |
|---|---|---|---|
| Plain SVD | $\|W - AB\|_F$ 最优 | 18.65 | 完全忽略数据分布 |
| Diag ASVD (α=1) | $\|(W-AB)\text{diag}(s)\|_F$ 最优 | 8.91 | 忽略通道相关性 |
| Whiten ASVD | $\mathbb{E}\|Wx-ABx\|^2$ 最优 | 8.64 | 仅限单矩阵层面 |
| + 非均匀 rank 分配 | 同上 + 预算分配最优 | 8.50 | 仍是单矩阵层面 |
| 端到端蒸馏 | 直接优化复合映射 | **3.79** | 需要训练 |

从 18.65 到 8.50 是"在单矩阵层面做到极致"；从 8.50 到 3.79 是"承认单矩阵不够，改成优化复合映射"。后者的改进（4.71 nat）远大于前者（10.15 nat 看似更大，但大部分是从"完全不对"到"基本对"的门槛效应）。
