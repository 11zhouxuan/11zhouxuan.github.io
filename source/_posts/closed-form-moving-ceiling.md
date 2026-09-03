---
title: "Low-Rank Compression Series (5): The Ceiling That Kept Moving — Sparse Residuals, a Better Metric, and Calibration Data"
date: 2026-08-30
mathjax: true
sticky: 10
tags: [math, linear-algebra, LLM, compression, distillation, low-rank, sparse, calibration]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 低秩压缩系列（五）：会移动的天花板——稀疏残差、更好的度量与校准数据

> 📖 如果你不熟悉语言模型的基本词汇（loss、残差流、SVD、蒸馏……），建议先读[预备知识篇](/2026/08/30/lord-compression-primer/)，10 分钟即可补齐全部背景。

[第四篇](/2026/08/25/closed-form-anatomy/)用三个组成部分的解剖宣布闭式赛道收官于 5.05。然后它在一周内被推到了 **4.59**——又砍掉近 0.5 nat，距离一个端到端蒸馏训练出来的模型（3.79）只剩 0.80。这篇讲天花板是怎么两次移动的。诚实地说，两次都源于同一件事：**合作者对"已经到头了"这个结论的不服气**。第一次的不服（"低秩逼近还有很大空间"）指向了度量修正，第二次（对统计地基的持续加码）打开了校准数据工程。"工具箱已榨干"的结论永远带着一个隐藏脚注：*在当前的度量、当前的表达形式、当前的统计地基下*。

背景（一句话）：Qwen3-8B（2.11）→ 每个 linear 换 rank-r 低秩因子（2.29B 等预算，删 72% 参数），方法 = 逐层轨迹矫正回归 + 免税矫正器 + loss 感知分配，第四篇终点 5.05。

（本文 loss 均为严格协议重测值：800 段 × 8192 token、8 折，折间波动约 ±0.02。第 6 节的 oracle 诊断保持当时的测量口径。）

### 0. 先看最终配方：四处改动写进同一个式子

第二篇确立的逐层目标是：解出 $M^\*$ 后在白化度量下截断，$\min\_{A,B} \lVert (M^\* - AB) L \rVert\_F^2$。本篇终点的逐层目标：

$$\min\_{A, B, S}\ \Big\lVert \underbrace{\mathrm{diag}(w)}\_{\text{改动②：loss 敏感度度量}} \big(M^\* - AB - \underbrace{S}\_{\text{改动①：稀疏残差}}\big) L \Big\rVert\_F^2 \qquad \text{s.t.}\quad \mathrm{rank}(AB) \le r, \quad \mathrm{nnz}(S) \le k$$

其中 $y\_j$ 是该层输出向量的第 $j$ 个分量，$w\_j^2 = \mathbb{E}\big[(\partial \mathcal{L}/\partial y\_j)^2\big]$（loss 对它的梯度平方的均值）；交替求解——AB 步是加权白化截断，S 步是取加权残差的 top-$k$ 元素。**改动③**不在式子里而在式子的原料里：所有统计量（$\Sigma\_{ss}, \Sigma\_{ts}, L, w$）改由 512 批 × 16 个数据分片的校准数据估计，本篇会证明这一条的贡献不小于任何算法改动。矫正器侧还有**改动④**（rms-lift：把矫正器输入从 $h$ 提升为 $[h;\ h(\bar{r}/\mathrm{rms}(h) - 1)]$，见第 5 节）。

下面按发生顺序逐个讲这四处改动。

### 1. 稀疏残差：W ≈ AB + S，第一次凿穿截断税

截断税（~2 nat，第四篇的组成部分①）的架构根源是内容通路的超位置存储：误差集中在少数几个又大又互不相关的矩阵元素上，低秩在数学上表达不了这种结构，但**稀疏项可以**——它天生就是"少数任意位置的大元素"。这是文献中证据最一致的换范式方向（OATS/HASSLE-free/LoSparse：等预算下 S+LR 一致优于纯 LR，且差距随压缩率放大）。

做法：在每层的截断步里做交替求解（3 轮）——

$$AB \leftarrow \text{whiten-SVD}(M^\*-S,\ r), \qquad S \leftarrow \text{top-}k\big(|M^\*-AB| \cdot \sigma\_{col}\big)$$

AB 步就是现成的白化截断，S 步按白化度量给残差打分取 top-nnz。预算严格配平：S 的每个非零值按 1.5 个参数计（含 index 开销；严格 CSR-int16 应为 2.0，勘误后纪录依然成立），从 rank 里扣。

**S/LR 配比的剂量曲线**（S 占总预算的份额）：

| S 份额 | val loss |
|---|---|
| 0%（纯低秩） | 5.05 |
| **21%** | **5.01** |
| 41% | 5.04 |

21% 即最优——**文献的"sparse 为主（70/30）"结论不迁移**，因为我们的低秩部分是轨迹矫正回归（远强于他们的朴素 SVD），S 只需要补真正的重尾。这也是一个方法论提醒：移植文献结论时，配比参数依附于基线强度。

### 2. 换度量：输出敏感度加权截断

第四篇曾逐个位置测量截断损失的分布（下称“损失分布图”），其中留了一个反常没解释：qkv@late 在白化能量下"几乎无损"（保留 94-98%），却是最大的税项（+0.56）。反常的谜底是：**我们的截断在错误的目标函数下最优**。白化 SVD 最小化的是该层输出的 L2 误差（Eckart-Young），但输出误差到最终 loss 的映射高度各向异性——q/k 的某些方向误差被 softmax 与下游放大数十倍，down 的很多大能量方向却被残差流稀释。

修法仍是闭式的。设 $y\_j$ 为该层输出向量的第 $j$ 个分量，测 $w\_j^2 = \mathbb{E}[(\partial L/\partial y\_j)^2]$（一次 backprop pass，全参数冻结、只让 embedding 输出带梯度），截断目标改为

$$\min\_{AB}\ \lVert \mathrm{diag}(w) (M^\*-AB) L \rVert\_F^2$$

SVD 前乘 $\mathrm{diag}(w)$、解出后除回即可。**单独 −0.07（4.98），零参数成本**——lm_head 矫正以来最大的免费午餐。两个工程细节决定成败：$w$ 必须做尺度归一 + clamp（[0.01, 100]）；且 $w$ 在一个"邻近"的 artifact 上测一次即可（见附录 A 的迭代失败）。

### 3. 超线性叠加：修正"小改进不叠加"

第四篇提出过"小改进不叠加"的规律（0.02 级的改进合并时互相侵蚀）。这次的组合实验修正了它：

| 配置（等预算） | val loss |
|---|---|
| 基线 | 5.05 |
| 只加稀疏残差 | 5.01（−0.04） |
| 只换度量 | 4.98（−0.07） |
| **两者组合** | **4.88（−0.17）** |

−0.04 和 −0.07 叠出了 −0.17，比线性相加的 −0.11 还多赚 0.06——**超线性**。机制：梯度度量同时升级了 S 的选择（打分从"能量残差"变成"loss 关键残差"，$|R|\cdot\sigma\_{col}\cdot w\_{row}$），而度量加权的 SVD 留下的残差恰好更适合 S 捡走。修正后的规律：**同一度量下的小改进不叠加；换度量的改进会放大其他所有杠杆**——因为度量作用于每一个有 rank 约束的决策点。

（一个理论上干净的推论顺带得证：对角输出加权对**全秩**回归无影响——逐行独立求解时权重不改变解。所以残差矫正器和 lm_head 矫正都吃不到这份收益，度量的收益只存在于 rank 约束处。这也回头解释了为什么当年 lm_head 矫正不需要任何加权就很强。）

### 4. 校准数据工程：被文献整体忽略的维度

闭式方法的一切统计量（$\Sigma\_{ss}, \Sigma\_{ts}$、梯度二阶矩）都来自校准数据。文献的标配是 128-256 条短样本（SVD-LLM 用 256×C4）；没有人把校准数据当成一个需要工程的对象。我们把它当成剂量-响应实验做了一遍：

| 统计配置 | val loss | 备注 |
|---|---|---|
| 32 批 × 1 shard | 4.88 | 组合配方基线 |
| 128 × 1 | 4.73 | 数量 ×4：−0.15 |
| 128 × 4 shards | 4.68 | **同数量、多样性 ×4**：再 −0.05 |
| 256 × 8 | 4.64 | 数量与多样性再翻倍 |
| 256 × 16 | 4.63 | 多样性饱和于 8 shards（与 ×8 打平） |
| **512 × 16** | **4.61** | 数量到 512 仍有小幅收益 |

从 4.88 到 4.61——**校准数据工程独立贡献了 ~0.27 nat**，比任何单个算法改进都大。两个可迁移的定量结论：

1. **数量与多样性是两个独立的轴。** 数量的第一档收益最大（32→128 批：−0.15），它压的是采样方差；数量吃饱之后，批数不变、把 1 个 shard 换成 4 个仍有 −0.05——这是偏差-方差分解里的**偏差项**：同一个 shard 里加再多数据，分布的偏色一分不动，只有换数据源才能压掉。
2. 两轴各自饱和：多样性 8 shards（256×8 与 256×16 打平）、数量 ~512 批（26 万→105 万 token 位置）仍有小幅收益。饱和点大约在"统计噪声降到单个算法改进量级（±0.01）"处，符合直觉。

与四个胜利并排的还有四个干净的失败：每当我们把一个有效的简单机制"做精"（迭代重测梯度、完整协方差、更大稀疏份额、按损失分布图定向分配），结果都没有变好——**在校准数据有限的闭式世界里，参数量少的估计器赢**。清单和幅度见附录 A。

### 5. 换度量的最后一笔收益：rms-lift 的平反

第四篇里，rms-lift 非线性矫正器在等预算下与直接加 rank 打平，被判"机制证实、收益归零"。换上新度量和加固后的统计地基重试，它贡献了最后的 −0.02：**4.61 → 4.59，当前的闭式最终纪录**。方向再次支持"换度量会重新激活此前打平的杠杆"，但幅度必须诚实标注——这 0.02 只有约一倍折间波动，是锦上添花而不是新机制。

### 6. 新范式的 oracle：地板低于一个训练过的模型

给新配方重测 oracle 分解（换掉全部子层输入为教师干净值）：

$$\text{oracle}\_{\text{新范式}} = 3.75 \quad (<\ 3.79\_{\text{端到端蒸馏训练}})$$

**闭式构造的层已经好到：只要输入干净，就能追平一个真训练过的模型。** 与旧范式对照：oracle 地板从 4.15 降到 3.75（稀疏+度量削掉了 0.4 的税本身）；可矫正漂移余量从 0.98 扩到 1.20 nat——层越好，漂移的相对代价越大，**当前纪录与训练之间的差距大头已经从"税"换成了"漂移"**。而漂移矫正器是全秩的、吃不到度量收益（第 3 节的推论），这 1.2 nat 仍锁在非线性后面：部分清洁实验（early/cliff）的伤害从旧范式的 −0.4 恶化到 −1.2，自洽链更紧了。

### 7. 结论

**最终版图**（85% 线性层压缩、等预算 2.29B、零训练）：

$$8.50 \to 5.60 \to 5.10 \to 5.05 \to \underbrace{4.88}\_{\text{稀疏×度量}} \to \underbrace{4.61}\_{\text{+校准工程}} \to \underbrace{\mathbf{4.59}}\_{\text{+rms-lift}} \to \underbrace{3.75}\_{\text{新 oracle}} \to \underbrace{3.79}\_{\text{端到端蒸馏}} \to 2.11\_{\text{教师}}$$

本篇新增的三条可迁移定律：

1. **度量是一等公民。** 逼近算法的"最优性"永远是相对某个度量的；换成 loss 感知的度量后，此前"榨干"的每个杠杆都重新有了改进空间。检查任何压缩管线时，先问它的截断在什么度量下最优。
2. **校准数据是一个被忽略的工程对象。** 数量和多样性各自独立贡献、各自饱和，两个轴都值得用剂量实验便宜地测一遍。~0.27 nat 的免费收益在文献的标准配置里全部躺着没人捡。
3. **参数少的估计器赢**（闭式版奥卡姆剃刀）：对角 > 完整协方差，单遍 > 迭代，阻尼 > 全量执行。校准数据的信息量是硬约束，精巧化只是把它重新分配给噪声。

天花板还会不会动？诚实的答案：本篇之后，配方内的所有维度（表达形式、度量、分配、矫正器、统计）都各自测到了饱和点，剩余的结构性候选只剩逐 head 的 (V,O) 联合分解一项。但第四篇也曾这样宣布过收官——这大概是这个系列最诚实的一课：**"到头了"是一个关于当前假设集的陈述，不是关于问题本身的**。

这个系列的下一步也不在"把 2.29B 的 loss 再降 0.02"上了。回到[预备篇](/2026/08/30/lord-compression-primer/)第 0 节的初心：低秩 init 的真正用途是给**任意大小**的小模型预训练造起点。rank 384/192/96/48/24 的秩扫描和"低秩 init vs 随机 init"的 scaling law 对照实验正在跑，那将是下一篇的主题。




---

## 附录

### 附录 A：失败清单——简单鲁棒版永远赢

与四个胜利并排的是四个干净的失败，它们共享一个模式——**每当我们把一个有效的简单机制"做精"，结果都更差**：

| 精巧化尝试 | 结果 | 简单版 |
|---|---|---|
| 迭代 G（在新模型上重测梯度再重建） | 4.89 | 一次测量 4.88 |
| 完整 G（k=64 特征子空间 + 对角） | 4.74 | 对角 G 4.73 |
| S 份额加大到 32%（度量加持下"S 更聪明了"） | 4.90 | 21% 的 4.88 仍最优 |
| 按损失分布图定向的矩阵类型×深度分配 | 5.08 | Fisher 阻尼图 5.05 |

诚实标注幅度：严格协议下这些"变差"多在 0.01~0.03，部分已在折间波动之内。所以稳健的结论不是"精巧化有害"，而是**四个更贵、更复杂的版本没有一个带来收益**。机制各不相同（迭代破坏自洽链、特征子空间对少量校准 batch 过拟合、oracle 收益不等于真实边际价值），但工程结论一致：**在校准数据有限的闭式世界里，参数量少的估计器赢**——对角赢完整、单遍赢迭代、阻尼赢全量。每个"更准的模型"都要用同样的几十到几百个 batch 去喂，多出来的自由度全变成噪声。


</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Low-Rank Compression Series (5): The Ceiling That Kept Moving — Sparse Residuals, a Better Metric, and Calibration Data

> 📖 New to language-model vocabulary (loss, residual stream, SVD, distillation...)? Read [the primer](/2026/08/30/lord-compression-primer/) first — ten minutes covers all the background.

[Part 4](/2026/08/25/closed-form-anatomy/) closed the closed-form track at 5.05 with a three-component anatomy of the gap. It was then pushed to **4.59** within a week — nearly another 0.5 nat, leaving just 0.80 to an end-to-end-distilled, actually-trained model (3.79). This post is about how the ceiling moved twice. Honestly, both moves came from the same source: **a collaborator refusing to accept "it's over"**. The first refusal ("low-rank approximation still has plenty of room") led to the metric fix; the second (insisting on the statistical foundations) opened up calibration-data engineering. "The toolbox is exhausted" always carries a hidden footnote: *under the current metric, the current representation, and the current statistics*.

Background in one line: Qwen3-8B (2.11) → every linear replaced by rank-r factors (2.29B equal budget, 72% removed); method = layerwise trajectory-correcting regression + tax-free correctors + loss-aware allocation; part 4 endpoint 5.05.

(All losses in this post are re-measured under the rigorous protocol: 800 validation passages × 8192 tokens, 8 folds, fold-to-fold spread about ±0.02. The oracle diagnostics in Section 6 keep their original measurement window.)

### 0. The Final Recipe First: Four Changes in One Formula

Part 2 established the layerwise objective: solve for $M^\*$, then truncate in the whitened metric, $\min\_{A,B} \lVert (M^\* - AB) L \rVert\_F^2$. This post's endpoint objective per layer:

$$\min\_{A, B, S}\ \Big\lVert \underbrace{\mathrm{diag}(w)}\_{\text{change ②: loss-sensitivity metric}} \big(M^\* - AB - \underbrace{S}\_{\text{change ①: sparse residual}}\big) L \Big\rVert\_F^2 \qquad \text{s.t.}\quad \mathrm{rank}(AB) \le r, \quad \mathrm{nnz}(S) \le k$$

where $y\_j$ is the $j$-th component of the layer's output vector and $w\_j^2 = \mathbb{E}\big[(\partial \mathcal{L}/\partial y\_j)^2\big]$ (the mean squared gradient of the loss with respect to it); solved by alternation — the AB step is a weighted whitened truncation, the S step keeps the top-$k$ entries of the weighted residual. **Change ③** is not in the formula but in its raw material: all statistics ($\Sigma\_{ss}, \Sigma\_{ts}, L, w$) are now estimated from 512 batches × 16 data shards of calibration data — this post will show its contribution is no smaller than any algorithmic change. On the corrector side there is **change ④** (rms-lift: lift the corrector input from $h$ to $[h;\ h(\bar{r}/\mathrm{rms}(h) - 1)]$, Section 5).

The sections below introduce these four changes in the order they happened.

### 1. Sparse Residuals: W ≈ AB + S, First Breach of the Truncation Tax

The truncation tax (~2 nats, component ① in part 4) is rooted in superposition storage on the content pathway: the error concentrates in a few large, mutually unrelated matrix entries — structure low-rank factors mathematically cannot express. A **sparse term can**: it is by nature "a few large entries at arbitrary positions." It is also the literature's most consistently supported paradigm change (OATS/HASSLE-free/LoSparse: S+LR beats pure LR at equal budget, with the gap widening at higher compression).

Method: alternate inside each layer's truncation step (3 rounds) —

$$AB \leftarrow \text{whiten-SVD}(M^\*-S,\ r), \qquad S \leftarrow \text{top-}k\big(|M^\*-AB| \cdot \sigma\_{col}\big)$$

The AB step is the existing whitened truncation; the S step scores the residual in the whitened metric and keeps the top-nnz. Strictly budgeted: each nonzero of S is charged 1.5 parameters (index overhead; strict CSR-int16 would be 2.0 — the record survives the erratum), paid out of rank.

**The S/LR dose-response** (S's share of the total budget):

| S share | val loss |
|---|---|
| 0% (pure low-rank) | 5.05 |
| **21%** | **5.01** |
| 41% | 5.04 |

21% is the optimum — **the literature's sparse-heavy (70/30) recipe does not transfer**, because our low-rank part is trajectory-correcting regression (far stronger than their plain SVD), so S only needs to catch the genuine heavy tails. A methodological reminder: ratio hyperparameters are attached to baseline strength.

### 2. Changing the Metric: Output-Sensitivity-Weighted Truncation

Part 4's loss-distribution map left one anomaly unexplained: qkv@late looks "nearly lossless" in whitened energy (94-98% kept) yet is the largest tax item (+0.56). The resolution: **our truncation was optimal under the wrong objective**. Whitened SVD minimizes the layer output's L2 error (Eckart-Young), but the map from output error to final loss is sharply anisotropic — some q/k directions get amplified tens of times by softmax and downstream, while many high-energy down directions get diluted in the residual stream.

The fix stays closed-form. Let $y\_j$ be the $j$-th component of the layer's output vector and measure $w\_j^2 = \mathbb{E}[(\partial L/\partial y\_j)^2]$ (one backprop pass; freeze all parameters and let only the embedding output require grad), and truncate under

$$\min\_{AB}\ \lVert \mathrm{diag}(w) (M^\*-AB) L \rVert\_F^2$$

— multiply by $\mathrm{diag}(w)$ before the SVD, divide after. **−0.07 on its own (4.98) at zero parameter cost**, the biggest free lunch since the lm_head fix. Two engineering details decide success: $w$ must be scale-normalized and clamped ([0.01, 100]); and one measurement on a "nearby" artifact suffices (see the iteration failure in Appendix A).

### 3. Super-Additive Stacking: Amending the No-Stacking Rule

Part 4 proposed a rule that small improvements don't stack (0.02-level gains erode each other on combination). The combination experiment this time amended it:

| Configuration (equal budget) | val loss |
|---|---|
| baseline | 5.05 |
| sparse residual only | 5.01 (−0.04) |
| metric change only | 4.98 (−0.07) |
| **both combined** | **4.88 (−0.17)** |

−0.04 and −0.07 stacked into −0.17, beating the plain sum of −0.11 by another 0.06 — **super-additive**. Mechanism: the gradient metric upgrades S's selection (its score becomes $|R|\cdot\sigma\_{col}\cdot w\_{row}$, targeting loss-critical rather than energy-heavy residuals), while the metric-weighted SVD leaves behind exactly the residue S is best at catching. The amended rule: **improvements under the same metric don't stack; changing the metric amplifies every other lever** — because the metric acts at every rank-constrained decision point.

(A theoretically clean corollary came for free: diagonal output weighting cannot affect a **full-rank** regression — rows solve independently, so weights don't change the solution. Hence residual correctors and the lm_head fix cannot collect this gain; the metric's gain exists only where a rank constraint does. This retroactively explains why the lm_head fix was strong without any weighting.)

### 4. Calibration-Data Engineering: the Dimension the Literature Skips

Everything a closed-form method knows comes from calibration data ($\Sigma\_{ss}, \Sigma\_{ts}$, gradient moments). The literature's standard is 128-256 short samples (SVD-LLM: 256×C4); nobody treats calibration data as an object of engineering. We ran it as a dose-response study:

| Statistics | val loss | Note |
|---|---|---|
| 32 batches × 1 shard | 4.88 | combo-recipe baseline |
| 128 × 1 | 4.73 | quantity ×4: −0.15 |
| 128 × 4 shards | 4.68 | **same quantity, diversity ×4**: another −0.05 |
| 256 × 8 | 4.64 | double both again |
| 256 × 16 | 4.63 | diversity saturates at 8 shards (tie with ×8) |
| **512 × 16** | **4.61** | quantity still pays mildly at 512 |

From 4.88 to 4.61 — **calibration-data engineering alone contributed ~0.27 nat**, more than any single algorithmic idea. Two transferable quantitative findings:

1. **Quantity and diversity are two independent axes.** Quantity's first installment is the largest (32 → 128 batches: −0.15) — it shrinks sampling variance. Once quantity is fed, holding 128 batches fixed and going 1 shard → 4 shards still buys −0.05 — the **bias** term of the bias-variance split: more data from the same shard leaves the distributional tint untouched; only changing the data source removes it.
2. Each axis saturates on its own: diversity at 8 shards (256×8 ties 256×16), quantity still pays mildly at ~512 batches (0.26M → 1.05M token positions). Saturation arrives roughly when statistical noise drops to the size of one algorithmic improvement (±0.01), as intuition suggests.

Alongside the four wins sit four clean failures: every attempt to refine a working simple mechanism (re-measured gradients, full covariance, a larger sparse share, loss-map-guided allocation) failed to improve anything — **in the calibration-limited closed-form world, the estimator with fewer parameters wins**. The list and the margins are in Appendix A.

### 5. The Metric's Last Installment: rms-Lift Redeemed

In part 4 the rms-lift nonlinear corrector tied with simply buying rank at equal budget and was shelved ("mechanism confirmed, gain zero"). Retried under the new metric and the reinforced statistics, it contributed the final −0.02: **4.61 → 4.59, the closed-form track's final record**. The direction again supports "changing the metric re-energizes levers that had washed out" — but honesty about the size: this 0.02 is about one fold-to-fold spread, a finishing touch rather than a new mechanism.

### 6. The New Paradigm's Oracle: a Floor Below a Trained Model

Re-running the oracle decomposition (swap all sublayer inputs with the teacher's clean values) on the new recipe:

$$\text{oracle}\_{\text{new}} = 3.75 \quad (<\ 3.79\_{\text{end-to-end distilled}})$$

**The closed-form layers are now good enough that, given clean inputs, they would match an actually-trained model.** Against the old paradigm: the oracle floor dropped 4.15 → 3.75 (sparse+metric shaved 0.4 off the tax itself), while the correctable-drift headroom grew 0.98 → 1.20 nats — the better the layers, the relatively costlier the drift. **The gap between the current record and training is now mostly drift, not tax.** And drift correctors are full-rank, hence unable to collect the metric's gain (the corollary in Section 3); that 1.2 nats stays locked behind the nonlinearities — partial-cleaning experiments (early/cliff) now HURT by −1.2 (vs −0.4 in the old paradigm): the self-consistent chain got tighter.

### 7. Conclusions

**The final landscape** (85% linear-layer compression, equal 2.29B budget, zero training):

$$8.50 \to 5.60 \to 5.10 \to 5.05 \to \underbrace{4.88}\_{\text{sparse×metric}} \to \underbrace{4.61}\_{\text{+calibration eng.}} \to \underbrace{\mathbf{4.59}}\_{\text{+rms-lift}} \to \underbrace{3.75}\_{\text{new oracle}} \to \underbrace{3.79}\_{\text{e2e distilled}} \to 2.11\_{\text{teacher}}$$

Three new transferable lessons from this round:

1. **The metric is a first-class citizen.** An approximation algorithm's "optimality" is always relative to a metric; switching to a loss-aware one reopened every previously "exhausted" lever. When auditing any compression pipeline, first ask under which metric its truncation is optimal.
2. **Calibration data is a neglected engineering object.** Quantity and diversity contribute independently and saturate independently; both axes are cheap to measure with dose-response runs. ~0.27 nat of free gains lie untouched in the literature's standard setup.
3. **Fewer-parameter estimators win** (the closed-form Occam's razor): diagonal > full covariance, single-pass > iterated, damped > full-strength. The information in the calibration set is the hard constraint; sophistication merely reallocates it to noise.

Will the ceiling move again? The honest answer: after this post, every dimension of the recipe (representation, metric, allocation, correctors, statistics) has been measured to its own saturation point, and the one remaining structural candidate is a per-head joint (V,O) decomposition. But part 4 declared closure once, too — which is perhaps this series' most honest lesson: **"it's over" is a statement about the current hypothesis set, not about the problem.**

The series' next step is also no longer about shaving another 0.02 off a 2.29B model. Back to Section 0 of [the primer](/2026/08/30/lord-compression-primer/): the real use of low-rank initialization is manufacturing pretraining starting points for small models of **any size**. The rank sweep (384/192/96/48/24) and the "low-rank init vs random init" scaling-law comparison are running now — that will be the next post.




---

## Appendix

### Appendix A: The Failure List — Simple-Robust Always Wins

Alongside four wins sit four clean failures, sharing one pattern — **every time we refined a working simple mechanism, the result got worse**:

| Refinement attempt | Result | The simple version |
|---|---|---|
| Iterated G (remeasure gradients on the new model, rebuild) | 4.89 | one measurement: 4.88 |
| Full G (k=64 eigen-subspace + diagonal) | 4.74 | diagonal G: 4.73 |
| S share raised to 32% ("S is smarter now") | 4.90 | 21% at 4.88 remains optimal |
| Loss-map-guided family×depth allocation | 5.08 | damped Fisher map: 5.05 |

Honesty about the sizes: under the rigorous protocol these "regressions" are mostly 0.01-0.03, some within fold noise. The robust conclusion is therefore not "refinement hurts" but that **none of the four more expensive, more complex versions delivered any gain**. The mechanisms differ (iteration breaks the self-consistent chain; the eigen-subspace overfits the small calibration set; oracle gains ≠ real marginal value), but the engineering conclusion is uniform: **in the calibration-limited closed-form world, the estimator with fewer parameters wins** — diagonal over full, single-pass over iterated, damped over full-strength. Every "more accurate model" must be fed by the same few-dozen-to-few-hundred batches; the extra degrees of freedom all turn into noise.


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
  var postTitles = {zh: '低秩压缩系列（五）：会移动的天花板——稀疏残差、更好的度量与校准数据', en: 'Low-Rank Compression Series (5): The Ceiling That Kept Moving — Sparse Residuals, a Better Metric, and Calibration Data'};
  var titleEl = document.querySelector('.post-title');
  if (titleEl) titleEl.textContent = postTitles[lang];
}
switchLang('zh');
</script>
