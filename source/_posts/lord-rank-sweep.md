---
title: "Low-Rank Compression Series (6): The Rank Sweep — How Quality Degrades from Rank 384 down to 24"
date: 2026-09-04
mathjax: true
sticky: 5
tags: [math, linear-algebra, LLM, compression, distillation, low-rank, rank-sweep]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 低秩压缩系列（六）：秩扫描——rank 从 384 一路砍到 24，质量如何退化

> 📖 如果你不熟悉语言模型的基本词汇（loss、残差流、SVD、蒸馏……），建议先读[预备知识篇](/2026/08/30/lord-compression-primer/)，10 分钟即可补齐全部背景。

前五篇把一件事做到了头：在 **2.29B 等预算**（全网 rank 384）这一个点上，闭式压缩从坍缩假象的 8.50 推到了 4.59。但整个系列只在这一个预算点上活动。[第五篇](/2026/08/30/closed-form-moving-ceiling/)结尾提出的下一个问题是：这套方法离开 2.29B 还站得住吗？**把 rank 一路砍半——384、192、96、48、24——质量怎么退化？有没有某个 rank 以下突然崩坏？** 本篇交这份答卷。

（本文的 loss 均为严格协议测量值：800 段 × 8192 token 的验证数据、8 折，折间波动约 ±0.02。教师 Qwen3-8B 为 2.11。）

### 1. 实验设置：配方不动，只缩放 rank

被扫描的对象是第五篇的终点配方（逐矩阵轨迹矫正回归 + 稀疏残差 + 梯度加权度量 + 残差流矫正器 + lm\_head 矫正），**所有算法成分保持不变，只把和 rank 相关的预算等比缩小**。记名义 rank 为 $R \in \lbrace 384, 192, 96, 48, 24 \rbrace$：

- **逐矩阵 rank 分配**：第四篇的 loss 敏感度分配给了每个矩阵一个基准秩 $\bar{r}\_\ell$（均值 251、范围 [116, 478]，其余预算在稀疏项和矫正器手里）。扫描时按比例缩放并设下限：

$$r\_\ell(R) = \max\big(8, \mathrm{round}(\bar{r}\_\ell \cdot R / 384)\big)$$

- **稀疏残差**：每个矩阵的非零元预算同样等比缩放，$\mathrm{nnz}\_\ell(R) = R\_S \cdot (m\_\ell + n\_\ell)/1.5$，其中 $R\_S = \max(4, \mathrm{round}(66 \cdot R/384))$（66 是 $R=384$ 时稀疏项占预算 21% 的换算值，1.5 是稀疏索引的存储开销折算）。

- **不随 rank 缩放的部分**：11 个残差流矫正器和 lm\_head 矫正是全秩回归，形状只依赖 hidden 维度，原样保留；校准数据（512 batch × 16 shard）、正则强度、矫正间隔 $K=3$ 也全部不变。

这个设计让曲线只回答一个问题：**transformer 核心的低秩容量变小时，质量怎么变**——而不是把配方调参的差异混进来。（每个 rank 点的派生配置数值见附录 A。）

### 2. 结果：一条很平的曲线

五个点全部按同一严格协议评测：

| 名义 rank $R$ | 总参数 | val loss | 相邻减半的代价 |
|---|---|---|---|
| 384 | 2.29B | **4.585** | — |
| 192 | 1.86B | **5.072** | +0.49 |
| 96 | 1.65B | **5.480** | +0.41 |
| 48 | 1.54B | **5.755** | +0.27 |
| 24 | 1.49B | **6.037** | +0.28 |

两个观察：

**第一，没有崩坏点。** 系列开头最担心的事情——某个 rank 以下模型突然坍缩成常数预测器——从未发生。哪怕在 $R=24$（逐矩阵 rank 均值只有 16、最小 8），模型依然健康。作为参照：第一篇里 rank **384** 的坍缩 ASVD 是 8.50、朴素 SVD 是 10.83——本篇 rank 只有它们 1/16 的模型（6.04）远好于它们。当年的"坍缩"确实是方法问题而不是容量问题，这条曲线是这个论点最完整的证据。

**第二，减半的代价在递减**：0.49 → 0.41 → 0.27 → 0.28 nat。低秩容量每砍一半，loss 涨幅反而变小，曲线在低秩端趋平。部分原因是下一节的固定地板——但即使只看 transformer 核心，从 0.86B 砍到 0.06B（−93%）总共只付了 1.45 nat，退化极其平缓。

### 3. 参数量的诚实账本：固定地板

上表有一个容易误读的地方：rank 砍了 16 倍，总参数只从 2.29B 降到 1.49B（−35%）。原因是学生模型里有一大块**完全不随 rank 变的固定开销**：

| 组成 | 参数量 | 备注 |
|---|---|---|
| embed + lm\_head | 1.245B | 词表 151936 × 4096 × 2（输入输出不共享） |
| 矫正器（11 个全秩 + head 矫正 bias） | 0.185B | 形状只依赖 hidden 维度 |
| **固定开销合计** | **1.43B** | 不随 $R$ 变 |
| 低秩因子 + 稀疏残差（$R=384$） | 0.86B | 随 $R$ 近似线性缩放 |

于是固定开销的占比从 $R=384$ 的 62% 一路涨到 $R=24$ 的 **96%**——最小的那个学生本质上是"两张巨大的词表矩阵，外加一点点 transformer"。

这带来两个提醒。其一，**画 loss–参数量曲线时要小心**：横轴若用总参数，低秩端的曲线形状几乎完全由词表矩阵决定，与被研究的压缩方法无关；报告时应单独标出固定开销，或直接用 non-embedding 参数做横轴（Chinchilla 之后 scaling law 文献的标准做法）。其二，如果目标是真正的小模型，**下一个该动的不是 rank 而是那 1.245B**：共享输入输出 embedding（Qwen3 的小尺寸型号本来就这样做）可立省 0.62B，lm\_head 本身的低秩化则要先单独测敏感度——第三篇的结论提醒我们它是全网收益最大的矫正位置，动它须谨慎。

### 4. 结论

1. **轨迹矫正配方对 rank 极其鲁棒**：384 → 24（逐矩阵均值 251 → 16），loss 退化全程平缓（每减半 +0.27~0.49 nat），无任何坍缩迹象。"低秩压缩在高压缩率下必然崩坏"对这套方法不成立。

2. **减半代价递减**（0.49 → 0.28），说明轨迹矫正回归在极低秩下仍能把有限容量用在刀刃上——这正是白化截断"按真实输入分布分配精度"的设计初衷。

3. **低秩端的参数账本由词表矩阵主导**（固定开销占比 62% → 96%）。秩扫描真正测量的是 transformer 核心的容量-质量关系；总参数口径下的任何结论都要先扣掉这 1.43B 的地板。

下一步回到[预备篇](/2026/08/30/lord-compression-primer/)第 0 节的初心：这些闭式 init 的真正用途是给小模型预训练造起点。"低秩 init vs 随机 init"的 scaling law 对照实验（多个尺寸、同数据同超参）正在准备，那将是下一篇的主题。

---

## 附录

### 附录 A：每个 rank 点的派生配置

按第 1 节的缩放规则，各 rank 点实际使用的配置：

| 名义 $R$ | rank map 均值 | rank map 范围 | 稀疏 $R\_S$ | 低秩因子+bias | 稀疏 nnz |
|---|---|---|---|---|---|
| 384 | 251 | [116, 478] | 66 | 0.684B | 120.0M |
| 192 | 125 | [58, 239] | 33 | 0.342B | 60.0M |
| 96 | 63 | [29, 120] | 17 | 0.172B | 29.1M |
| 48 | 31 | [14, 60] | 8 | 0.087B | 14.5M |
| 24 | 16 | [8, 30] | 4 | 0.045B | 7.3M |

（rank map 均值低于名义 $R$，是因为总预算里稀疏项和矫正器占走了一部分；$R=24$ 时下限 8 开始明显起作用，范围被压到 [8, 30]。参数量为从 artifact 文件实测的数值，稀疏项按 nnz × 1.5 计入索引开销。）

### 附录 B：小 rank 下的一个数值坑

rank 48 的蒸馏在第 12 个 block 处崩过一次：白化截断需要对统计矩阵做 Cholesky 分解（要求矩阵正定），而小 rank 下某些统计矩阵的最小特征值贴近浮点误差，分解报"not positive-definite"。修复是标准的两步：先把矩阵对称化（$\Sigma \leftarrow (\Sigma + \Sigma^T)/2$，消掉累积浮点不对称），失败则把对角正则项每次放大 10 倍重试（最多 7 次）。加固后 48 和 24 一次通过。教训：**在名义正定矩阵上做分解的代码，正则强度要能自适应地升级**——固定的 $\lambda$ 在设计工况下够用，在扫描的极端点上不一定。

</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Low-Rank Compression Series (6): The Rank Sweep — How Quality Degrades from Rank 384 down to 24

> 📖 New to language-model vocabulary (loss, residual stream, SVD, distillation...)? Read [the primer](/2026/08/30/lord-compression-primer/) first — ten minutes covers all the background.

The first five posts pushed one point to its limit: at the **equal 2.29B budget** (rank 384 everywhere), closed-form compression went from the collapse illusion's 8.50 down to 4.59. But the whole series lived at that single budget point. The question [part 5](/2026/08/30/closed-form-moving-ceiling/) left open: does the method survive away from 2.29B? **Keep halving the rank — 384, 192, 96, 48, 24 — how does quality degrade, and is there a rank below which things suddenly break?** This post delivers the answer.

(All losses are measured under the rigorous protocol: 800 validation passages × 8192 tokens, 8 folds, fold-to-fold spread about ±0.02. The teacher Qwen3-8B sits at 2.11.)

### 1. Setup: Freeze the Recipe, Scale Only the Rank

The object under the sweep is part 5's endpoint recipe (per-matrix trajectory-correcting regression + sparse residuals + gradient-weighted metric + residual-stream correctors + lm\_head fix). **Every algorithmic ingredient stays fixed; only rank-related budgets scale proportionally.** With nominal rank $R \in \lbrace 384, 192, 96, 48, 24 \rbrace$:

- **Per-matrix rank allocation**: part 4's loss-sensitivity allocation assigned each matrix a base rank $\bar{r}\_\ell$ (mean 251, range [116, 478]; the rest of the budget lives in the sparse terms and correctors). The sweep scales it with a floor:

$$r\_\ell(R) = \max\big(8, \mathrm{round}(\bar{r}\_\ell \cdot R / 384)\big)$$

- **Sparse residuals**: each matrix's nonzero budget scales the same way, $\mathrm{nnz}\_\ell(R) = R\_S \cdot (m\_\ell + n\_\ell)/1.5$ with $R\_S = \max(4, \mathrm{round}(66 \cdot R/384))$ (66 corresponds to the sparse share of 21% at $R=384$; the 1.5 discounts index storage overhead).

- **What does not scale**: the 11 residual-stream correctors and the lm\_head fix are full-rank regressions whose shapes depend only on the hidden width — kept as is; calibration data (512 batches × 16 shards), regularization, and the corrector interval $K=3$ are also unchanged.

The design makes the curve answer exactly one question — **how quality responds as the transformer core's low-rank capacity shrinks** — without mixing in recipe re-tuning. (Derived per-rank configurations are in Appendix A.)

### 2. Results: a Remarkably Flat Curve

All five points, same rigorous protocol:

| Nominal rank $R$ | Total params | val loss | Cost of each halving |
|---|---|---|---|
| 384 | 2.29B | **4.585** | — |
| 192 | 1.86B | **5.072** | +0.49 |
| 96 | 1.65B | **5.480** | +0.41 |
| 48 | 1.54B | **5.755** | +0.27 |
| 24 | 1.49B | **6.037** | +0.28 |

Two observations:

**First, there is no breaking point.** The scenario the series began by fearing — below some rank, the model suddenly collapses into a constant predictor — never happens. Even at $R=24$ (per-matrix mean rank 16, minimum 8) the model stays healthy. For reference: part 1's collapsed ASVD at rank **384** scored 8.50 and plain SVD 10.83 — this post's model with **1/16 of their rank** (6.04) beats both by a wide margin. The old "collapse" really was a method problem, not a capacity problem, and this curve is the most complete evidence for that thesis.

**Second, the cost of halving shrinks**: 0.49 → 0.41 → 0.27 → 0.28 nats. Each halving of low-rank capacity costs less than the previous one; the curve flattens at the low end. Part of this is the fixed floor of the next section — but even counting only the transformer core, cutting it from 0.86B to 0.06B (−93%) costs a total of 1.45 nats. The degradation is remarkably graceful.

### 3. An Honest Parameter Ledger: the Fixed Floor

The table above invites one misreading: rank dropped 16×, yet total parameters only fell from 2.29B to 1.49B (−35%). The reason is a large block of the student that **does not depend on rank at all**:

| Component | Params | Note |
|---|---|---|
| embed + lm\_head | 1.245B | vocab 151936 × 4096 × 2 (input/output untied) |
| Correctors (11 full-rank + head-fix bias) | 0.185B | shapes depend only on the hidden width |
| **Fixed overhead, total** | **1.43B** | independent of $R$ |
| Low-rank factors + sparse residuals ($R=384$) | 0.86B | scales roughly linearly with $R$ |

So the fixed overhead's share climbs from 62% at $R=384$ to **96%** at $R=24$ — the smallest student is essentially "two enormous vocabulary matrices plus a sliver of transformer."

Two takeaways. First, **be careful when plotting loss against parameter count**: with total parameters on the x-axis, the low-rank end's shape is dictated almost entirely by the vocabulary matrices, not by the compression method under study; report the fixed overhead separately, or use non-embedding parameters as the axis (standard practice in the scaling-law literature since Chinchilla). Second, if the goal is a genuinely small model, **the next thing to cut is not rank but that 1.245B**: tying the input and output embeddings (which Qwen3's small variants already do) saves 0.62B outright, while low-ranking the lm\_head itself needs a sensitivity test first — part 3 found it to be the single most valuable correction site in the network, so touch it with care.

### 4. Conclusions

1. **The trajectory-correcting recipe is extremely robust to rank**: 384 → 24 (per-matrix mean 251 → 16) degrades smoothly the whole way (+0.27 to +0.49 nats per halving), with no sign of collapse. "Low-rank compression must break down at high compression" does not hold for this method.

2. **The halving cost shrinks** (0.49 → 0.28): even at very low rank, the trajectory-correcting regression keeps spending its limited capacity where it matters — precisely what the whitened truncation's "allocate precision by the real input distribution" design was for.

3. **At the low-rank end the parameter ledger is dominated by the vocabulary matrices** (fixed share 62% → 96%). What the sweep truly measures is the capacity–quality relation of the transformer core; any conclusion phrased in total parameters must first subtract the 1.43B floor.

The next step returns to the founding motivation in [the primer](/2026/08/30/lord-compression-primer/), Section 0: these closed-form inits exist to give small-model pretraining a head start. A "low-rank init vs. random init" scaling-law comparison (multiple sizes, same data and hyperparameters) is in preparation — that will be the next post.

---

## Appendix

### Appendix A: Derived Configuration at Each Rank

Applying Section 1's scaling rules, the actual configuration at each point:

| Nominal $R$ | rank-map mean | rank-map range | sparse $R\_S$ | low-rank factors+bias | sparse nnz |
|---|---|---|---|---|---|
| 384 | 251 | [116, 478] | 66 | 0.684B | 120.0M |
| 192 | 125 | [58, 239] | 33 | 0.342B | 60.0M |
| 96 | 63 | [29, 120] | 17 | 0.172B | 29.1M |
| 48 | 31 | [14, 60] | 8 | 0.087B | 14.5M |
| 24 | 16 | [8, 30] | 4 | 0.045B | 7.3M |

(The rank-map mean sits below the nominal $R$ because part of the total budget lives in the sparse terms and correctors; at $R=24$ the floor of 8 visibly kicks in, squeezing the range to [8, 30]. Parameter counts are measured from the artifact files; sparse terms are counted as nnz × 1.5 for index overhead.)

### Appendix B: A Numerical Pitfall at Small Ranks

The rank-48 distillation crashed once at block 12: the whitened truncation Cholesky-factorizes a statistics matrix (which must be positive-definite), and at small ranks some of these matrices have smallest eigenvalues at the level of floating-point error, so the factorization reports "not positive-definite." The fix is the standard two-step: symmetrize first ($\Sigma \leftarrow (\Sigma + \Sigma^T)/2$, removing accumulated floating-point asymmetry), and on failure retry with the diagonal regularizer inflated 10× each time (up to 7 attempts). With the hardening in place, ranks 48 and 24 ran through on the first try. Lesson: **code that factorizes nominally positive-definite matrices needs a regularizer that can escalate** — a fixed $\lambda$ suffices in the designed operating range, not necessarily at a sweep's extremes.

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
  var postTitles = {zh: '低秩压缩系列（六）：秩扫描——rank 从 384 一路砍到 24，质量如何退化', en: 'Low-Rank Compression Series (6): The Rank Sweep — How Quality Degrades from Rank 384 down to 24'};
  var titleEl = document.querySelector('.post-title');
  if (titleEl) titleEl.textContent = postTitles[lang];
}
switchLang('zh');
</script>
