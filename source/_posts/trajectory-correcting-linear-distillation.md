---
title: "Trajectory-Correcting Linear Distillation: Breaking the Closed-Form Frontier in Low-Rank LLM Compression (Part 2)"
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

## 轨迹矫正线性蒸馏：突破低秩压缩的闭式方法边界（二）

> 📖 如果你不熟悉语言模型的基本词汇（loss、残差流、SVD、蒸馏……），建议先读[预备知识篇](/2026/08/30/lord-compression-primer/)，10 分钟即可补齐全部背景。

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

从 Block 0 到 35 顺序处理。处理第 $i$ 层时（前面各层已压缩）：

**第 1 步**：教师和学生对同样的数据前向，在该层入口配对采集 $x\_t$（教师轨迹输入）和 $x\_s$（学生轨迹输入），累积互协方差 $\Sigma\_{ts} = \sum x\_t x\_s^T$ 和自协方差 $\Sigma\_{ss} = \sum x\_s x\_s^T$（中心化版本）。

**第 2 步**：解仿射岭回归（岭回归 = 带 $\lambda I$ 正则项的最小二乘，保证矩阵可逆、抑制对采样噪声的过拟合）：

$$\min\_{M,b}\; \mathbb{E}\lVert W x\_t - M x\_s - b \rVert^2 \quad\Longrightarrow\quad M^\* = W\Sigma\_{ts}(\Sigma\_{ss} + \lambda I)^{-1}$$

直觉：$\Sigma\_{ts}\Sigma\_{ss}^{-1}$ 是"从脏输入线性还原干净输入"的最优算子，再与 $W$ 复合——还原与变换合成在一个矩阵里。

**第 3 步**：白化截断到 rank $r$。直接对 $M^\*$ 做 SVD 截断隐含"输入各方向同等重要"的假设，而真实输入分布并非如此；先取 $L = \mathrm{chol}(\Sigma\_{ss}+\lambda I)$ 把输入坐标变换到协方差为单位阵的"白化"坐标，再对 $M^\*L$ 做 SVD 截断，就是在 $x\_s$ 的真实分布下可证最优的截断（白化坐标里的 Eckart–Young 定理，推导见[预备篇](/2026/08/30/lord-compression-primer/)）。截断后重算 $b = W\bar{x}\_t - AB\bar{x}\_s$，让 bias 顺带吸收截断在均值处的误差。

**第 4 步**：替换该层，处理下一层。下一层采集到的 $x\_s$ 自动包含新换上的层的误差，于是被下一层矫正。

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
3. ~~逐层线性方法的天花板约为 5.6。~~ **【后续更新】** 5.6 平台随后被「免税矫正器」（lm_head 矫正 + 残差流全秩矫正器，均为零/低参数成本）推进到 **5.10**，并且我们用 oracle 实验测出了整个逐层范式的硬地板（≈4.0）。详见[完结篇《闭式压缩的天花板》](/2026/08/22/closed-form-ceiling/)。
4. 更正上一篇的结论："闭式方法无法同时打破坍缩又降低 loss"是错的——错的是当时测试的所有方法共享的"逼近 W"目标，而不是闭式本身。

</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Trajectory-Correcting Linear Distillation: Breaking the Closed-Form Frontier in Low-Rank LLM Compression (Part 2)

> 📖 New to language-model vocabulary (loss, residual stream, SVD, distillation...)? Read [the primer](/2026/08/30/lord-compression-primer/) first — ten minutes covers all the background.

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

Process blocks 0→35 sequentially. For layer $i$ (upstream already compressed):

**Step 1**: Run teacher and student on the same data; capture paired inputs $x\_t$ (teacher trajectory) and $x\_s$ (student trajectory) at the layer's entrance; accumulate the cross-covariance $\Sigma\_{ts} = \sum x\_t x\_s^T$ and self-covariance $\Sigma\_{ss} = \sum x\_s x\_s^T$ (centered).

**Step 2**: Solve the affine ridge regression (ridge = least squares with a $\lambda I$ regularizer, keeping the matrix invertible and damping overfit to sampling noise):

$$\min\_{M,b}\; \mathbb{E}\lVert W x\_t - M x\_s - b \rVert^2 \quad\Longrightarrow\quad M^\* = W\Sigma\_{ts}(\Sigma\_{ss} + \lambda I)^{-1}$$

Intuition: $\Sigma\_{ts}\Sigma\_{ss}^{-1}$ is the optimal linear operator recovering the clean input from the corrupted one, composed with $W$ — restoration and transformation fused into one matrix.

**Step 3**: Whitened rank-$r$ truncation. Truncating the SVD of $M^\*$ directly would implicitly assume all input directions matter equally, which the real input distribution violates; instead take $L = \mathrm{chol}(\Sigma\_{ss}+\lambda I)$ to transform the input into "whitened" coordinates where the covariance is the identity, then SVD-truncate $M^\*L$ — provably optimal under the student's true input distribution (Eckart–Young in whitened coordinates; derivation in [the primer](/2026/08/30/lord-compression-primer/)). Recompute $b = W\bar{x}\_t - AB\bar{x}\_s$ afterwards so the bias also absorbs the truncation error at the mean.

**Step 4**: Replace the layer, move on. The next layer's $x\_s$ automatically includes the newly introduced error, which the next regression corrects.

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
3. ~~The ceiling for layerwise-linear methods is ≈5.6.~~ **[Later update]** The 5.6 plateau was subsequently pushed to **5.10** by 'tax-free correctors' (the lm_head fix plus full-rank residual-stream correctors, at zero/low parameter cost), and an oracle experiment established the hard floor of the entire layerwise paradigm (≈4.0). See [the finale: The Closed-Form Ceiling](/2026/08/22/closed-form-ceiling/).
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
  var postTitles = {zh: '轨迹矫正线性蒸馏：突破低秩压缩的闭式方法边界（二）', en: 'Trajectory-Correcting Linear Distillation: Breaking the Closed-Form Frontier in Low-Rank LLM Compression (Part 2)'};
  var titleEl = document.querySelector('.post-title');
  if (titleEl) titleEl.textContent = postTitles[lang];
}
switchLang('zh');
</script>
