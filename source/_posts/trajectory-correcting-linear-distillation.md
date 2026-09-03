---
title: "Low-Rank Compression Series (2): Trajectory-Correcting Linear Distillation — Breaking the Closed-Form Frontier"
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

## 低秩压缩系列（二）：轨迹矫正线性蒸馏——突破闭式方法的边界

> 📖 如果你不熟悉语言模型的基本词汇（loss、残差流、SVD、蒸馏……），建议先读[预备知识篇](/2026/08/30/lord-compression-primer/)，10 分钟即可补齐全部背景。

### 1. 问题：闭式方法卡在一堵"坍缩墙"上

[上一篇](/2026/08/17/representation-collapse-in-low-rank-compression/)里我们把 Qwen3-8B 的每个线性层替换为 rank-384 的 $AB$（85% 压缩率），发现所有闭式（无训练）方法卡在一条边界上：

| 方法 | val loss | 实质 |
|---|---|---|
| Plain SVD | 18.65 | 噪声压倒信号 |
| ASVD（$\alpha=0.5$，均匀 rank） | 10.83 | 高度退化，但未坍缩 |
| ASVD（$\alpha=1.0$，逐矩阵分配 rank） | 8.50 | **完全坍缩成常数函数**（98.6% 预测逗号） |
| 常数预测器理论下界 | 7.51 | $H(\text{unigram})$ |

两行 ASVD 是同一方法的不同配置：把加权指数从 0.5 提到 1.0、均匀 rank 换成逐矩阵分配（最小的矩阵被饿到 rank 32），模型就从"高度退化"滑进"完全坍缩"。最后一行的**常数预测器**指完全无视上下文、每个位置输出同一个概率分布的模型——它能达到的最低 loss 是语料词频分布的熵 $H(\text{unigram}) = 7.51$（推导见第一篇第 2 节）。这条线是试金石：**loss 低于 7.51 才说明模型真的在利用上下文**。

8.50 看似最好，实为假象：它是"放弃预测、输出接近词频分布"的退化策略，离常数预测器的下界只有 1 nat。所有试图打破坍缩的手段（正交约束、rank 重分配、给关键组件 70% 更多参数）都让 loss 变得更差。

但有一个关键事实说明这堵墙不是低秩本身的极限：**从 8.50 出发做端到端训练能到 3.79——而训练出的模型就是同样的 rank-384 结构**。也就是说，rank-384 参数空间里存在好得多的点，只是"逐层逼近 $W$"这类局部目标找不到它。

### 2. 洞察：错的不是低秩，是逐层目标

所有失败的闭式方法都在解同一类问题——用某种范数逼近教师权重：

$$\min\_{A,B} \lVert (W - AB)S \rVert\_F^2$$

这个目标有一个隐含假设：这一层在推理时会收到**教师轨迹上的干净输入**。但压缩后的模型里，第 $i$ 层收到的是**已经被前 $i-1$ 层的压缩误差污染的输入**。误差逐层累积，36 层后不是坍缩就是爆炸。

记 $x\_t$ 为这一层在教师模型里本来会收到的干净输入，$x\_s$ 为压缩后的学生模型实际送进来的、已经漂移的输入。三个"考虑误差传播"的目标，只差在**输入用谁、目标用谁**，结果天壤之别：

| 逐层目标 | 含义 | val loss |
|---|---|---|
| 逼近 $W$，按漂移分布加权 | 追逐腐化的分布 | 19.40 |
| 匹配 $W x\_{s}$ | 教师权重作用在脏输入上——**顺从漂移** | 12.22 |
| 匹配 $W x\_{t}$ | **把激活拉回教师轨迹** | **6.72** |

只有第三种在**矫正**漂移。每一层不再模仿 $W$，而是成为一个矫正器：接住漂移的输入，输出教师轨迹上本该有的结果。

### 3. 方法：逐层仿射岭回归

先定义**校准数据** $D$：从训练集（FineWeb-Edu）里取出的一小批文本，本篇用 32 个 batch、每 batch 8 段 × 8192 token，合计约 210 万个 token 位置。它的唯一用途是**估计统计量**——把教师和学生在同样文本上跑一遍前向，收集每个权重矩阵入口处输入的均值与协方差，供下面的回归求解使用。全程没有任何梯度更新，所以这类方法叫"闭式"（closed-form，解方程直接得到答案）。$D$ 与用来报告 loss 的验证数据完全无重叠。（校准数据的用量和多样性本身对结果影响很大，这是[第五篇](/2026/08/30/closed-form-moving-ceiling/)的主题之一。）

完整算法只有一个循环：

> **算法 1：轨迹矫正线性蒸馏**
>
> **输入**：教师模型的全部权重矩阵 $W\_1, \dots, W\_N$，按前向顺序编号（$N = 252$：模型共 36 层，每层含 7 个）、校准数据 $D$、目标秩 $r$、正则强度 $\lambda$
>
> **对 $\ell = 1, 2, \dots, N$ 依次执行**（循环体内的 $x\_t, x\_s, \Sigma, M^\*, A, B, b$ 都属于当前的 $W\_\ell$，为简洁省略编号）：
>
> 1. 教师与当前学生（前 $\ell-1$ 个矩阵已替换）在 $D$ 上配对前向，在 $W\_\ell$ 的入口采集教师输入 $x\_t$ 与学生输入 $x\_s$，累积均值 $\bar{x}\_t, \bar{x}\_s$ 和中心化协方差 $\Sigma\_{ts} = \sum x\_t x\_s^T$、$\Sigma\_{ss} = \sum x\_s x\_s^T$
> 2. 岭回归：$M^\* = W\_\ell \Sigma\_{ts} (\Sigma\_{ss} + \lambda I)^{-1}$
> 3. 白化截断：$L = \mathrm{chol}(\Sigma\_{ss} + \lambda I)$，对 $M^\* L$ 做 SVD 保留前 $r$ 项得 $U\_r \Sigma\_r V\_r^T$，令 $A = U\_r \Sigma\_r$、$B = V\_r^T L^{-1}$
> 4. 偏置：$b = W\_\ell \bar{x}\_t - AB\bar{x}\_s$；把 $W\_\ell$ 替换为 $x \mapsto ABx + b$
>
> **输出**：全部 $N$ 个权重矩阵替换完毕的学生模型

其中第 2 步的 $M^\*$ 是下面这个仿射岭回归的解，**本篇的全部改进都在这一个式子里**：

$$(M^\*, b^\*) = \arg\min\_{M, b}\ \mathbb{E}\big\lVert W\_\ell x\_t - M x\_s - b \big\rVert^2$$

目标 $W\_\ell x\_t$ 是教师轨迹的干净输出，输入 $x\_s$ 是学生的漂移输入，期望取自学生的真实运行分布。对比第一篇的代理目标 $\min \lVert (W - AB) S \rVert\_F^2$——那里输入分布和逼近目标都活在教师的世界里——这一处换目标贡献了 8.50 → 5.60 的全部差距。

每一步的理由：

- **顺序处理（循环体）**：$W\_\ell$ 入口采集到的 $x\_s$ 自动携带前面所有已替换矩阵的误差，本次回归顺带矫正它——矫正链因此环环自洽。
- **第 2 步（岭回归）**：带 $\lambda I$ 正则项的最小二乘，求逆稳定、抑制对采样噪声的过拟合。直觉：$\Sigma\_{ts}\Sigma\_{ss}^{-1}$ 是"从脏输入线性还原干净输入"的最优算子，与 $W\_\ell$ 复合成一个矩阵。
- **第 3 步（白化截断）**：直接截断 $M^\*$ 隐含"输入各方向同等重要"的假设，而真实输入分布并非如此；先用 $L$ 变换到协方差为单位阵的白化坐标再截断，是 $x\_s$ 真实分布下可证最优的截断（白化坐标里的 Eckart–Young 定理，推导见[预备篇](/2026/08/30/lord-compression-primer/)）。
- **第 4 步（偏置）**：$b$ 顺带吸收截断在均值处的误差。

### 4. 结果

| 配置 | val loss | 预测多样性 |
|---|---|---|
| v1（8 个统计 batch，无 bias） | 6.72 | 773 个 unique token |
| **v2（32 个 train batch + bias）** | **5.60** | 1264 个 unique，top-1 为 " the"，位置间 KL=5.4 |

（本系列的 loss 后来统一按严格协议重测：800 段 × 8192 token 的验证数据、8 折，折间波动约 ±0.02。本文轨迹矫正系列的数字均为重测值；第 1 节坍缩时代的数字仍是早期窗口的测量，只作定性对照。）

完整版图：

$$18.65 \to 10.83 \to \underbrace{8.50}\_{\text{坍缩假象}} \to \underbrace{7.51}\_{\text{常数下界}} \to \mathbf{5.60} \to \underbrace{3.79}\_{\text{训练}} \to \underbrace{2.11}\_{\text{教师}}$$

5.60 **低于常数预测器的信息论下界 7.51**——模型真正携带了从上下文到下一个 token 的互信息，这是所有闭式方法中的第一次。预测行为完全健康：top-1 是正确的高频词 " the"（教师也是），不同位置输出分布之间的 KL=5.4，说明预测确实随上下文变化（坍缩模型是 0.007，即每个位置都输出同一个分布）。

### 5. 方法的平台：三个负结果

进一步的优化尝试全部失败，方法在 ~5.6 收敛：

1. **不动点迭代 → 5.95（变差）**。想法是"压缩后的学生漂移变了，那就用它重新采集统计、把所有层再解一遍，迭代到自洽"。实测反而破坏了第一遍矫正链的自洽性——每层的解适配了上游的特定误差模式，重解任何一层都会让下游已学到的补偿失配。**单遍顺序处理就是最优做法。**
2. **按奇异值谱分配 per-layer rank → 5.73（略差）**。给谱衰减慢（更难压）的层多分 rank、衰减快的少分。均匀 rank 已接近最优；从均匀运行算出的分配也无法迁移到新运行（漂移模式随分配改变）。
3. **λ 不敏感**：$10^{-3}$ 与 $10^{-4}$ 差 0.01。

### 6. 剩下的差距在哪：R² 诊断

逐层测量"教师输入能从学生输入线性恢复的比例"（回归的 $R^2$）：

| 位置 | $R^2$ |
|---|---|
| 前几层 | 0.65~0.80 |
| 中段（q/k/v 输入） | 0.52~0.60 |
| 中段（down\_proj 输入，即 SwiGLU 乘积） | **0.25~0.40** |
| 尾段 | 回升至 ~0.6 |

中段网络的漂移有一半以上是**非线性**的，其中 SwiGLU 的 gate×up 乘积处最严重——两个带误差的量相乘，误差项会出现平方与交叉项，这是任何线性算子都无法还原的成分。这就是 5.6 平台的成因：**逐层线性矫正已经榨干了漂移中的线性可恢复部分，剩余 1.8 nat 的差距（5.60 到训练的 3.79）属于非线性漂移**，原理上需要非线性矫正器或全局优化（训练）才能跨越。

### 7. 结论

1. **低秩压缩的瓶颈不在表达能力，在优化目标**。rank-384 空间中存在 3.79 的点；"逐层逼近 W"找不到它，"逐层矫正轨迹"能走到 5.60。
2. **闭式方法的正确姿势是回归而不是分解**：输入取自学生的真实（漂移）分布，目标取自教师的理想轨迹——每层既是压缩，也是对上游误差的一次线性纠错。
3. **逐层线性矫正在 ~5.6 收敛**：漂移中线性可恢复的部分已经榨干，剩余差距是非线性的。这个"天花板"是否真的到头，是[下一篇](/2026/08/22/closed-form-ceiling/)的主题。
4. 更正上一篇的结论："闭式方法无法同时打破坍缩又降低 loss"是错的——错的是当时测试的所有方法共享的"逼近 W"目标，而不是闭式本身。

</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Low-Rank Compression Series (2): Trajectory-Correcting Linear Distillation — Breaking the Closed-Form Frontier

> 📖 New to language-model vocabulary (loss, residual stream, SVD, distillation...)? Read [the primer](/2026/08/30/lord-compression-primer/) first — ten minutes covers all the background.

### 1. The Problem: A "Collapse Wall" for Closed-Form Methods

In the [previous post](/2026/08/17/representation-collapse-in-low-rank-compression/) we replaced every linear layer of Qwen3-8B with a rank-384 factorization $AB$ (85% compression) and found all closed-form (training-free) methods stuck at a frontier:

| Method | val loss | Reality |
|---|---|---|
| Plain SVD | 18.65 | Noise overwhelms signal |
| ASVD ($\alpha=0.5$, uniform ranks) | 10.83 | Severely degenerate, but not collapsed |
| ASVD ($\alpha=1.0$, per-matrix ranks) | 8.50 | **Fully collapsed to a constant function** (98.6% commas) |
| Constant-predictor floor | 7.51 | $H(\text{unigram})$ |

The two ASVD rows are the same method under different configurations: raising the weighting exponent from 0.5 to 1.0 and swapping uniform ranks for per-matrix allocation (the smallest matrix starved to rank 32) slides the model from "severely degenerate" into "fully collapsed." The last row's **constant predictor** is a model that ignores context entirely and emits the same distribution at every position — the lowest loss it can reach is the entropy of the corpus word-frequency distribution, $H(\text{unigram}) = 7.51$ (derived in part 1, Section 2). That line is the litmus test: **only a loss below 7.51 proves the model actually uses context**.

The 8.50 is an illusion: a "give up and emit the word-frequency distribution" strategy sitting 1 nat above the constant-predictor floor. Every attempt to break the collapse (orthogonality constraints, rank reallocation, 70% more parameters for key components) made loss worse.

One fact showed the wall is not intrinsic to low rank: **end-to-end training from 8.50 reaches 3.79 — and the trained model has exactly the same rank-384 structure**. A far better point exists in the same parameter space; layerwise "approximate $W$" objectives simply cannot find it.

### 2. The Insight: Low Rank Isn't Wrong — the Layerwise Objective Is

Every failed closed-form method solves some version of

$$\min\_{A,B} \lVert (W - AB)S \rVert\_F^2$$

with a hidden assumption: at inference this layer will receive **clean inputs from the teacher's trajectory**. In the compressed model, layer $i$ actually receives inputs **corrupted by the accumulated error of layers $0..i-1$**. Over 36 layers the error either collapses or explodes.

Write $x\_t$ for the clean input this layer would have received inside the teacher, and $x\_s$ for the drifted input the compressed student actually feeds it. Three "error-propagation-aware" objectives differ only in **which input and which target they use** — with wildly different outcomes:

| Layerwise objective | Meaning | val loss |
|---|---|---|
| Approximate $W$, weighted by drifted stats | Chases the corrupted distribution | 19.40 |
| Match $W x\_s$ | Teacher weight on corrupted input — **follows the drift** | 12.22 |
| Match $W x\_t$ | **Pulls activations back to the teacher trajectory** | **6.72** |

Only the third CORRECTS drift. Each layer stops imitating $W$ and becomes a corrector: take the drifted input, emit what the teacher's trajectory would have produced.

### 3. The Method: Layerwise Affine Ridge Regression

First, what **calibration data** $D$ means: a small batch of text drawn from the training set (FineWeb-Edu) — here 32 batches of 8 passages × 8192 tokens, about 2.1M token positions in total. Its only purpose is **estimating statistics**: run the teacher and the student over the same text and collect the means and covariances of the inputs at each weight matrix's entrance, to be consumed by the regressions below. No gradient update happens anywhere, which is why these methods are called closed-form (solve equations, get the answer directly). $D$ has no overlap with the validation data used to report losses. (How much calibration data, and how diverse, turns out to matter a great deal — one of [part 5](/2026/08/30/closed-form-moving-ceiling/)'s subjects.)

The complete algorithm is a single loop:

> **Algorithm 1: Trajectory-Correcting Linear Distillation**
>
> **Input**: all of the teacher's weight matrices $W\_1, \dots, W\_N$ in forward order ($N = 252$: 36 layers with 7 matrices each), calibration data $D$, target rank $r$, regularization strength $\lambda$
>
> **For $\ell = 1, 2, \dots, N$** (inside the loop, $x\_t, x\_s, \Sigma, M^\*, A, B, b$ all belong to the current $W\_\ell$; the index is dropped for brevity):
>
> 1. Run teacher and the current student (matrices $1..\ell-1$ already replaced) on $D$ in paired forwards; at $W\_\ell$'s entrance collect the teacher input $x\_t$ and student input $x\_s$; accumulate the means $\bar{x}\_t, \bar{x}\_s$ and centered covariances $\Sigma\_{ts} = \sum x\_t x\_s^T$, $\Sigma\_{ss} = \sum x\_s x\_s^T$
> 2. Ridge regression: $M^\* = W\_\ell \Sigma\_{ts} (\Sigma\_{ss} + \lambda I)^{-1}$
> 3. Whitened truncation: $L = \mathrm{chol}(\Sigma\_{ss} + \lambda I)$; take the top-$r$ SVD of $M^\* L$ as $U\_r \Sigma\_r V\_r^T$; set $A = U\_r \Sigma\_r$, $B = V\_r^T L^{-1}$
> 4. Bias: $b = W\_\ell \bar{x}\_t - AB\bar{x}\_s$; replace $W\_\ell$ with $x \mapsto ABx + b$
>
> **Output**: the student model with all $N$ weight matrices replaced

The $M^\*$ of step 2 solves the affine ridge regression below — **all of this post's improvement is inside this one equation**:

$$(M^\*, b^\*) = \arg\min\_{M, b}\ \mathbb{E}\big\lVert W\_\ell x\_t - M x\_s - b \big\rVert^2$$

The target $W\_\ell x\_t$ is the teacher trajectory's clean output; the input $x\_s$ is the student's drifted input; the expectation runs over the student's real running distribution. Contrast with part 1's proxy $\min \lVert (W - AB) S \rVert\_F^2$ — there, both the input distribution and the target live in the teacher's world — and this single change of objective accounts for all of 8.50 → 5.60.

Why each step:

- **Sequential processing (the loop)**: the $x\_s$ collected at $W\_\ell$'s entrance automatically carries the errors of all previously replaced matrices, so this regression corrects them in passing — the corrector chain stays self-consistent link by link.
- **Step 2 (ridge)**: least squares with a $\lambda I$ regularizer — stable inversion, damped overfit to sampling noise. Intuition: $\Sigma\_{ts}\Sigma\_{ss}^{-1}$ is the optimal linear operator recovering the clean input from the corrupted one, fused with $W\_\ell$ into one matrix.
- **Step 3 (whitened truncation)**: truncating $M^\*$ directly assumes all input directions matter equally, which the real input distribution violates; transforming with $L$ into whitened coordinates (identity covariance) first makes the truncation provably optimal under $x\_s$'s true distribution (Eckart–Young in whitened coordinates; derivation in [the primer](/2026/08/30/lord-compression-primer/)).
- **Step 4 (bias)**: $b$ absorbs the truncation error at the mean.

### 4. Results

| Configuration | val loss | Prediction diversity |
|---|---|---|
| v1 (8 stat batches, no bias) | 6.72 | 773 unique tokens |
| **v2 (32 train batches + bias)** | **5.60** | 1264 unique, top-1 " the", cross-position KL 5.4 |

(The series' losses were later re-measured under one rigorous protocol: 800 validation passages × 8192 tokens, 8 folds, fold-to-fold spread about ±0.02. The trajectory-correction numbers in this post are the re-measured values; the collapse-era numbers in Section 1 remain early-window measurements, kept for qualitative contrast only.)

The full landscape:

$$18.65 \to 10.83 \to \underbrace{8.50}\_{\text{collapse illusion}} \to \underbrace{7.51}\_{\text{constant floor}} \to \mathbf{5.60} \to \underbrace{3.79}\_{\text{trained}} \to \underbrace{2.11}\_{\text{teacher}}$$

5.60 is **below the constant-predictor floor of 7.51** — the model genuinely carries mutual information from context to next token, a first among all closed-form methods. Its behavior is healthy: top-1 is the correct high-frequency word " the" (same as the teacher), and the KL divergence between output distributions at different positions is 5.4, i.e. predictions really do vary with context (the collapsed model: 0.007, the same distribution everywhere).

### 5. The Plateau: Three Negative Results

Further optimization attempts all failed; the method converges at ~5.6:

1. **Fixed-point iteration → 5.95 (worse)**. The idea: the compressed student's drift has changed, so re-collect statistics with it and re-solve every layer, iterating toward self-consistency. In practice this destroys the self-consistency of the pass-1 corrector chain — each layer's solution is adapted to its upstream's specific error pattern; re-solving any layer invalidates downstream compensations. **A single sequential sweep is the best approach.**
2. **Singular-spectrum-driven per-layer rank allocation → 5.73 (slightly worse)**. Give more rank to layers whose spectrum decays slowly (harder to compress), less to the rest. Uniform ranks are already near-optimal, and an allocation computed from a uniform run does not transfer to a new run (the drift pattern changes with the allocation).
3. **λ-insensitive**: $10^{-3}$ vs $10^{-4}$ differ by 0.01.

### 6. Where the Remaining Gap Lives: the R² Diagnostic

Per layer, measure the fraction of the teacher input linearly recoverable from the student input (the regression $R^2$):

| Location | $R^2$ |
|---|---|
| Early blocks | 0.65–0.80 |
| Mid-network (q/k/v inputs) | 0.52–0.60 |
| Mid-network (down\_proj input = SwiGLU product) | **0.25–0.40** |
| Late blocks | recovers to ~0.6 |

More than half of the mid-network drift is **nonlinear**, worst at the SwiGLU gate×up product — multiplying two quantities that each carry error produces squared and cross terms, components no linear operator can undo. This explains the 5.6 plateau: **layerwise linear correction has extracted all the linearly recoverable drift; the remaining 1.8-nat gap (from 5.60 to the trained 3.79) is nonlinear drift**, requiring nonlinear correctors or global optimization (training) to cross.

### 7. Conclusions

1. **The bottleneck of low-rank compression is not expressiveness but the optimization objective.** A 3.79 point exists in the rank-384 space; "approximate $W$ per layer" cannot find it, while "correct the trajectory per layer" reaches 5.60.
2. **The right closed-form primitive is regression, not factorization**: inputs from the student's real (drifted) distribution, targets from the teacher's ideal trajectory — each layer is simultaneously compression and one step of linear error correction.
3. **Layerwise linear correction converges at ~5.6**: the linearly recoverable part of the drift is exhausted; what remains is nonlinear. Whether this ceiling is truly final is the subject of [the next post](/2026/08/22/closed-form-ceiling/).
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
  var postTitles = {zh: '低秩压缩系列（二）：轨迹矫正线性蒸馏——突破闭式方法的边界', en: 'Low-Rank Compression Series (2): Trajectory-Correcting Linear Distillation — Breaking the Closed-Form Frontier'};
  var titleEl = document.querySelector('.post-title');
  if (titleEl) titleEl.textContent = postTitles[lang];
}
switchLang('zh');
</script>
