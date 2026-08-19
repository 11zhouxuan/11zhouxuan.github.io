---
title: "Trajectory-Correcting Linear Distillation: Breaking the Closed-Form Frontier in Low-Rank LLM Compression"
date: 2026-08-19
mathjax: true
tags: [math, linear-algebra, LLM, compression, distillation, low-rank, ridge-regression]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 轨迹矫正线性蒸馏：突破低秩压缩的闭式方法边界

### 1. 问题：闭式方法卡在一堵"坍缩墙"上

[上一篇](/2026/08/17/representation-collapse-in-low-rank-compression/)里我们把 Qwen3-8B 的每个线性层替换为 rank-384 的 $AB$（85% 压缩率），发现所有闭式（无训练）方法卡在一条边界上：

| 方法 | val loss | 实质 |
|---|---|---|
| Plain SVD | 18.65 | 噪声压倒信号 |
| ASVD（激活加权） | 10.83 | 部分退化 |
| 坍缩 ASVD | 8.50 | **常数函数**（98.6% 预测逗号） |
| 常数预测器理论下界 | 7.51 | $H(\text{unigram})$ |

8.50 看似最好，实为假象：它是"放弃预测、输出接近 unigram 分布"的退化策略，离常数预测器的信息论下界只有 1 nat。所有试图打破坍缩的手段（正交约束、rank 重分配、给关键组件 70% 更多参数）都让 loss 变得更差。

但有一个关键事实说明这堵墙不是低秩本身的极限：**从 8.50 出发做端到端训练能到 3.79——而训练出的模型就是同样的 rank-384 结构**。也就是说，rank-384 参数空间里存在好得多的点，只是"逐层逼近 $W$"这类局部目标找不到它。

### 2. 洞察：错的不是低秩，是逐层目标

所有失败的闭式方法都在解同一类问题——用某种范数逼近教师权重：

$$\min_{A,B} \lVert (W - AB)S \rVert_F^2$$

这个目标有一个隐含假设：这一层在推理时会收到**教师轨迹上的干净输入**。但压缩后的模型里，第 $i$ 层收到的是**已经被前 $i-1$ 层的压缩误差污染的输入**。误差逐层累积，36 层后不是坍缩就是爆炸。

三个"考虑误差传播"的目标，只差一个符号位置，结果天壤之别：

| 逐层目标 | 含义 | val loss |
|---|---|---|
| 逼近 $W$，按漂移分布加权 | 追逐腐化的分布 | 19.40 |
| 匹配 $W x_{s}$ | 教师权重作用在脏输入上——**顺从漂移** | 12.22 |
| 匹配 $W x_{t}$ | **把激活拉回教师轨迹** | **6.72** |

只有第三种在**矫正**漂移。每一层不再模仿 $W$，而是成为一个矫正器：接住漂移的输入，输出教师轨迹上本该有的结果。

### 3. 方法：逐层仿射岭回归

从 Block 0 到 35 顺序处理。处理第 $i$ 层时（前面各层已压缩）：

**第 1 步**：教师和学生对同样的数据前向，在该层入口配对采集 $x_t$（教师轨迹输入）和 $x_s$（学生轨迹输入），累积互协方差 $\Sigma_{ts} = \sum x_t x_s^T$ 和自协方差 $\Sigma_{ss} = \sum x_s x_s^T$（中心化版本）。

**第 2 步**：解仿射岭回归

$$\min_{M,b}\; \mathbb{E}\lVert W x_t - M x_s - b \rVert^2 \quad\Longrightarrow\quad M^* = W\Sigma_{ts}(\Sigma_{ss} + \lambda I)^{-1}$$

直觉：$\Sigma_{ts}\Sigma_{ss}^{-1}$ 是"从脏输入线性还原干净输入"的最优算子，再与 $W$ 复合——还原与变换合成在一个矩阵里。

**第 3 步**：白化截断到 rank $r$。取 $L = \mathrm{chol}(\Sigma_{ss}+\lambda I)$，对 $M^*L$ 做 SVD 截断——这在 $x_s$ 的真实分布下是可证最优的截断（白化坐标里的 Eckart–Young）。截断后重算 $b = W\bar{x}_t - AB\bar{x}_s$，让 bias 顺带吸收截断在均值处的误差。

**第 4 步**：替换该层，处理下一层。下一层采集到的 $x_s$ 自动包含新换上的层的误差，于是被下一层矫正。

### 4. 结果

| 配置 | held-out val loss | 预测多样性 |
|---|---|---|
| v1（8 个统计 batch，无 bias） | 6.72 | 773 个 unique token |
| **v2（32 个 train batch + bias）** | **5.59** | 1264 个 unique，top-1 为 " the"，位置间 KL=5.4 |

完整版图：

$$18.65 \to 10.83 \to \underbrace{8.50}_{\text{坍缩假象}} \to \underbrace{7.51}_{\text{常数下界}} \to \mathbf{5.59} \to \underbrace{3.79}_{\text{训练}} \to \underbrace{2.11}_{\text{教师}}$$

5.59 **低于常数预测器的信息论下界**——模型真正携带 context→token 的互信息，这是所有闭式方法中的第一次。预测行为完全健康：top-1 是正确的高频词 " the"（教师也是），位置间分布 KL=5.4（坍缩模型是 0.007）。

### 5. 方法的平台：三个负结果

进一步的优化尝试全部失败，方法在 ~5.6 收敛：

1. **不动点迭代 → 5.94（变差）**。用第一遍的学生重新采集统计、重解所有层，反而破坏了第一遍矫正链的自洽性——每层的解适配了上游的特定误差模式，重解任何一层都会让下游已学到的补偿失配。**单遍 sequential 就是甜点。**
2. **谱驱动的 per-layer rank 分配 → 5.72（略差）**。均匀 rank 已是甜点；从均匀运行算出的分配无法迁移到新运行（漂移模式随分配改变）。
3. **λ 不敏感**：$10^{-3}$ 与 $10^{-4}$ 差 0.01。

### 6. 剩下的差距在哪：R² 诊断

逐层测量"教师输入能从学生输入线性恢复的比例"（回归的 $R^2$）：

| 位置 | $R^2$ |
|---|---|
| 前几层 | 0.65~0.80 |
| 中段（q/k/v 输入） | 0.52~0.60 |
| 中段（down\_proj 输入，即 SwiGLU 乘积） | **0.25~0.40** |
| 尾段 | 回升至 ~0.6 |

中段网络的漂移有一半以上是**非线性**的，其中 SwiGLU 的 gate×up 乘积处最严重——乘法把上游的线性误差二次化，产生了线性算子无法还原的成分。这就是 5.6 平台的成因：**逐层线性矫正已经榨干了漂移中的线性可恢复部分，剩余 1.8 nat 的差距属于非线性漂移**，原理上需要非线性矫正器或全局优化（训练）才能跨越。

### 7. 结论

1. **低秩压缩的瓶颈不在表达能力，在优化目标**。rank-384 空间中存在 3.79 的点；"逐层逼近 W"找不到它，"逐层矫正轨迹"能走到 5.59。
2. **闭式方法的正确姿势是回归而不是分解**：输入取自学生的真实（漂移）分布，目标取自教师的理想轨迹——每层既是压缩，也是对上游误差的一次线性纠错。
3. **逐层线性方法的天花板约为 5.6**（85% 压缩率下），由漂移的非线性成分决定，SwiGLU 乘法是主要来源。
4. 更正上一篇的结论："闭式方法无法同时打破坍缩又降低 loss"是错的——错的是当时测试的所有方法共享的"逼近 W"目标，而不是闭式本身。

</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Trajectory-Correcting Linear Distillation: Breaking the Closed-Form Frontier in Low-Rank LLM Compression

### 1. The Problem: A "Collapse Wall" for Closed-Form Methods

In the [previous post](/2026/08/17/representation-collapse-in-low-rank-compression/) we replaced every linear layer of Qwen3-8B with a rank-384 factorization $AB$ (85% compression) and found all closed-form (training-free) methods stuck at a frontier:

| Method | val loss | Reality |
|---|---|---|
| Plain SVD | 18.65 | Noise overwhelms signal |
| ASVD (activation-weighted) | 10.83 | Partially degenerate |
| Collapsed ASVD | 8.50 | **Constant function** (98.6% commas) |
| Constant-predictor floor | 7.51 | $H(\text{unigram})$ |

The 8.50 is an illusion: a "give up and emit the unigram distribution" strategy sitting 1 nat above the information-theoretic floor for constant predictors. Every attempt to break the collapse (orthogonality constraints, rank reallocation, 70% more parameters for key components) made loss worse.

One fact showed the wall is not intrinsic to low rank: **end-to-end training from 8.50 reaches 3.79 — and the trained model has exactly the same rank-384 structure**. A far better point exists in the same parameter space; layerwise "approximate $W$" objectives simply cannot find it.

### 2. The Insight: Low Rank Isn't Wrong — the Layerwise Objective Is

Every failed closed-form method solves some version of

$$\min_{A,B} \lVert (W - AB)S \rVert_F^2$$

with a hidden assumption: at inference this layer will receive **clean inputs from the teacher's trajectory**. In the compressed model, layer $i$ actually receives inputs **corrupted by the accumulated error of layers $0..i-1$**. Over 36 layers the error either collapses or explodes.

Three "error-propagation-aware" objectives differ only in where one symbol sits — with wildly different outcomes:

| Layerwise objective | Meaning | val loss |
|---|---|---|
| Approximate $W$, weighted by drifted stats | Chases the corrupted distribution | 19.40 |
| Match $W x_s$ | Teacher weight on corrupted input — **follows the drift** | 12.22 |
| Match $W x_t$ | **Pulls activations back to the teacher trajectory** | **6.72** |

Only the third CORRECTS drift. Each layer stops imitating $W$ and becomes a corrector: take the drifted input, emit what the teacher's trajectory would have produced.

### 3. The Method: Layerwise Affine Ridge Regression

Process blocks 0→35 sequentially. For layer $i$ (upstream already compressed):

**Step 1**: Run teacher and student on the same data; capture paired inputs $x_t$ (teacher trajectory) and $x_s$ (student trajectory) at the layer's entrance; accumulate the cross-covariance $\Sigma_{ts} = \sum x_t x_s^T$ and self-covariance $\Sigma_{ss} = \sum x_s x_s^T$ (centered).

**Step 2**: Solve the affine ridge regression

$$\min_{M,b}\; \mathbb{E}\lVert W x_t - M x_s - b \rVert^2 \quad\Longrightarrow\quad M^* = W\Sigma_{ts}(\Sigma_{ss} + \lambda I)^{-1}$$

Intuition: $\Sigma_{ts}\Sigma_{ss}^{-1}$ is the optimal linear operator recovering the clean input from the corrupted one, composed with $W$ — restoration and transformation fused into one matrix.

**Step 3**: Whitened rank-$r$ truncation. With $L = \mathrm{chol}(\Sigma_{ss}+\lambda I)$, SVD-truncate $M^*L$ — provably optimal under the student's true input distribution (Eckart–Young in whitened coordinates). Recompute $b = W\bar{x}_t - AB\bar{x}_s$ afterwards so the bias also absorbs the truncation error at the mean.

**Step 4**: Replace the layer, move on. The next layer's $x_s$ automatically includes the newly introduced error, which the next regression corrects.

### 4. Results

| Configuration | held-out val loss | Prediction diversity |
|---|---|---|
| v1 (8 stat batches, no bias) | 6.72 | 773 unique tokens |
| **v2 (32 train batches + bias)** | **5.59** | 1264 unique, top-1 " the", cross-position KL 5.4 |

The full landscape:

$$18.65 \to 10.83 \to \underbrace{8.50}_{\text{collapse illusion}} \to \underbrace{7.51}_{\text{constant floor}} \to \mathbf{5.59} \to \underbrace{3.79}_{\text{trained}} \to \underbrace{2.11}_{\text{teacher}}$$

5.59 is **below the constant-predictor floor** — the model genuinely carries context→token mutual information, a first among all closed-form methods. Its behavior is healthy: top-1 is the correct high-frequency word " the" (same as the teacher), cross-position KL 5.4 (the collapsed model: 0.007).

### 5. The Plateau: Three Negative Results

Further optimization attempts all failed; the method converges at ~5.6:

1. **Fixed-point iteration → 5.94 (worse)**. Re-collecting stats with the pass-1 student and re-solving destroys the self-consistency of the pass-1 corrector chain — each layer's solution is adapted to its upstream's specific error pattern; re-solving any layer invalidates downstream compensations. **A single sequential sweep is the sweet spot.**
2. **Spectrum-driven per-layer rank allocation → 5.72 (slightly worse)**. Uniform ranks are already the sweet spot; an allocation computed from a uniform run does not transfer.
3. **λ-insensitive**: $10^{-3}$ vs $10^{-4}$ differ by 0.01.

### 6. Where the Remaining Gap Lives: the R² Diagnostic

Per layer, measure the fraction of the teacher input linearly recoverable from the student input (the regression $R^2$):

| Location | $R^2$ |
|---|---|
| Early blocks | 0.65–0.80 |
| Mid-network (q/k/v inputs) | 0.52–0.60 |
| Mid-network (down\_proj input = SwiGLU product) | **0.25–0.40** |
| Late blocks | recovers to ~0.6 |

More than half of the mid-network drift is **nonlinear**, worst at the SwiGLU gate×up product — the multiplication squares upstream linear errors into components no linear operator can undo. This explains the 5.6 plateau: **layerwise linear correction has extracted all the linearly recoverable drift; the remaining 1.8-nat gap is nonlinear drift**, requiring nonlinear correctors or global optimization (training) to cross.

### 7. Conclusions

1. **The bottleneck of low-rank compression is not expressiveness but the optimization objective.** A 3.79 point exists in the rank-384 space; "approximate $W$ per layer" cannot find it, while "correct the trajectory per layer" reaches 5.59.
2. **The right closed-form primitive is regression, not factorization**: inputs from the student's real (drifted) distribution, targets from the teacher's ideal trajectory — each layer is simultaneously compression and one step of linear error correction.
3. **The ceiling for layerwise-linear methods is ≈5.6** (at 85% compression), set by the nonlinear component of the drift, primarily the SwiGLU multiplication.
4. A correction to the previous post: "closed-form methods cannot simultaneously break collapse and lower loss" was wrong — what was broken was the shared "approximate $W$" objective of every method tested then, not closed-form itself.

</div>

<script>
function switchLang(lang) {
  document.querySelectorAll('.lang-content').forEach(function(el) {
    el.style.display = 'none';
  });
  document.querySelectorAll('.lang-btn').forEach(function(el) {
    el.classList.remove('active');
  });
  document.querySelector('.lang-' + lang).style.display = 'block';
  document.getElementById('btn-' + lang).classList.add('active');
}
</script>
