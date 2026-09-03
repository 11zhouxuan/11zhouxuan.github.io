---
title: "Low-Rank Compression Series (3): The Closed-Form Ceiling — Tax-Free Correctors, an Oracle Bound, and a Nonlinear Amplifier"
date: 2026-08-22
mathjax: true
sticky: 30
tags: [math, linear-algebra, LLM, compression, distillation, low-rank, oracle-bound]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 低秩压缩系列（三）：闭式压缩的天花板——免税矫正器、oracle 上界与非线性放大器

> 📖 如果你不熟悉语言模型的基本词汇（loss、残差流、SVD、蒸馏……），建议先读[预备知识篇](/2026/08/30/lord-compression-primer/)，10 分钟即可补齐全部背景。

这是低秩压缩三部曲的完结篇（[第一篇：表示坍缩](/2026/08/17/representation-collapse-in-low-rank-compression/)，[第二篇：轨迹矫正线性蒸馏](/2026/08/19/trajectory-correcting-linear-distillation/)）。上一篇结束时，闭式（无训练）方法停在 val loss 5.60。本篇回答三个问题：还能推到哪？极限在哪、为什么？以及这一切对"闭式 init + 训练"的完整路线意味着什么。

（本文的 loss 均为严格协议重测值：800 段 × 8192 token 的验证数据、8 折，折间波动约 ±0.02。第 4 节训练过程中的 loss 来自训练时的快速评估，口径略有不同，只用于组内对比。）

### 1. 消融的意外：便宜的赢了贵的

上一篇的 R² 诊断指出剩余差距主要是 SwiGLU 乘法处的非线性漂移。我们据此设计了 v3：给 down\_proj 的回归输入扩展为 $[gu;\ \mathrm{silu}(g);\ u]$（36864 维）——乘法处的误差交叉项 $g\delta\_u + u\delta\_g$（$\delta$ 是各自的漂移误差）在原空间里是非线性的，但在这个扩展空间里重新变成了线性可表，理论上非常对症。同时顺手加了一个"免费"组件：lm\_head 矫正。

消融实验（逐个拆掉组件、分别测量每个组件的贡献）的结果完全出乎预料（同为 rank-384 基础）：

| 组件 | 贡献 | 代价 |
|---|---|---|
| 扩展特征 | −0.06 nat | **+340M 参数** |
| **lm\_head 矫正** | **−0.29 nat** | **≈0 参数** |

精心设计的特征工程收益微薄（纯误差二阶项 $\delta\_g\delta\_u$ 在扩展空间中依然不可表），而"免费"的那个贡献了近五倍的改善。两者基本可加：$5.60 - 0.29 - 0.06 = 5.26$，实测组合 5.23——比简单相加还略好一点，差距在折间波动的边缘。

### 2. 免税矫正器原理

先解释本篇反复出现的一个词：把一个矩阵砍到 rank 384 必然损失一部分逼近精度，这份不可避免的损失我们称为**截断税**。全网 252 个低秩层，每层都在交税。

lm\_head 矫正为什么这么值？出口流水线是：36 个 block → final RMSNorm → lm\_head（4096→151936，未压缩）→ logits。所有压缩方法只矫正被压缩的层，而 lm\_head "没被压缩所以没人碰"——但它收到的**输入**已经漂移了。解一个 4096×4096 的全秩回归 $P$（学生 final hidden → 教师 final hidden），吸收进头部权重：

$$W\_{lm}' = W\_{lm}P, \qquad b = W\_{lm}(\bar{x}\_t - P\bar{x}\_s)$$

形状不变、计算量不变、零新增参数（除一个 0.15M 的 bias）。它强在四点：final norm 处的漂移 74% 线性可恢复（$R^2=0.742$）；矫正是**全秩**的——这是全网唯一不付截断税的位置；它是最后一跳，矫正直通 logits；每个 token 都经过它。

这提炼出一个通用原理：**把预算花在免截断税的位置**。全网还有一类这样的位置——block 之间的残差流。每 6 个 block 插一个满秩矫正 $h \leftarrow Ph + b$（拟合学生残差 → 教师残差，$W=I$ 无需截断，16.8M 参数/个），block rank 从 384 降到 353 配平预算：

| 同预算配置（2.29B） | val loss |
|---|---|
| v2 | 5.60 |
| + lm\_head 矫正 | 5.31 |
| + 5 个残差矫正器（K=6，rank 353） | 5.11 |
| + 11 个残差矫正器（K=3，rank 316） | **5.10** |
| 对照：v2 rank=768（**2 倍参数**） | 5.12 |

同预算的免税矫正器追平甚至略胜两倍参数的暴力加 rank。K=6→3 只再赚 0.01，已在折间波动量级——收益在此饱和。**闭式同预算纪录：5.10。**

本篇的完整方法是在第二篇算法 1 的基础上做两处扩展：

> **算法 2：免税矫正器（对算法 1 的两处扩展）**
>
> **扩展 1——残差流矫正器**：在算法 1 的循环中，每当一个 block 的 7 个矩阵全部替换完毕、且该 block 编号是 $K$ 的倍数（本文 $K=3$，全网共 11 个矫正器），追加执行：
>
> 1. 教师与当前学生在 $D$ 上配对前向，在该 block 出口采集教师残差流 $h\_t$ 与学生残差流 $h\_s$，累积均值与协方差
> 2. 解全秩回归 $(P, b) = \arg\min\_{P, b}\ \mathbb{E}\lVert h\_t - P h\_s - b \rVert^2$（$P \in \mathbb{R}^{4096 \times 4096}$，闭式解与算法 1 第 2 步同构）
> 3. 在该 block 后插入矫正步 $h \mapsto Ph + b$；此后的统计采集自动包含它的效果，矫正链保持自洽
>
> **扩展 2——lm_head 矫正**：循环结束后，在 final RMSNorm 出口采集 $x\_t, x\_s$，同样解全秩回归 $P$，把它吸收进未压缩的输出头：$W\_{lm}' = W\_{lm}P$、$b\_{lm} = W\_{lm}(\bar{x}\_t - P\bar{x}\_s)$——形状与计算量不变，零新增参数（除 0.15M 的 bias）
>
> **预算配平**：11 个矫正器共约 185M 参数，由全网 rank 从 384 降到 316 支付，总参数量保持 2.29B
>
> **输出**：学生模型 = 低秩层 + 11 个残差流矫正器 + 矫正后的 lm_head（val loss = **5.10**）

与算法 1 逐矩阵回归的关键区别一目了然：那里的 $M^\*$ 必须截断到 rank $r$（付截断税），这里的 $P$ 保持满秩——**没有 rank 约束、最优解不损失精度，这正是"免税"的形式化含义**。

### 3. oracle 上界：逐层范式的地板在 ~4.0

5.10 距离训练可达的 3.3 还有多远的闭式空间？我们直接测量"矫正器的理论极限"。做法是一个 **oracle 实验**——允许作弊、专门用来测理论上限的实验：运行时把每个子层的输入张量替换成教师的干净值（配对前向，替换 q/k/v 和 gate/up 的输入），于是每层只贡献自己的局部截断误差，误差在残差流中只加性累积、不再穿过非线性复合。

这个作弊模型的 loss 是"rank-r 线性层 + 任意输入侧矫正器"整个家族的**地板**：无论矫正器是线性、分段还是任意非线性，它最多做到把输入还原干净——也就是最多做到 oracle 的水平，不可能更低。实测：

$$\text{oracle} = 4.0 \sim 4.3$$

即使漂移矫正做到神级，rank-384 局部截断误差之和也要付 4 个 nat。想到 2.5？对这个范式**严格不可达**。

但同时出现了一个更深的事实：**训练（step 500 时 3.20）已经打穿了 oracle 界（4.0）**。逻辑上完全自洽：oracle 只约束"每层模仿教师对应层"的范式，而梯度训练**重组了各层的分工**——训练出的层不再是教师各层的逼近，而是一组全新的因子化方案，其复合效果超越逐层模仿的极限。这是第二篇"瓶颈在优化目标不在表达能力"论点的最强版本。

剩余非线性漂移的来源也定位到了：几乎全部来自**单独一层**（block 16）——它是教师自带的非线性放大器，容量、分配、线性矫正三类手段全部无效，闭式极限因此锁定在 ~5.1（调查细节见附录 A）。

### 4. 训练判决：好的闭式 init 值 200+ 步训练

双臂对照（同 seed、同数据序、同 global batch），只差初始化：

| step | 坍缩 init（8.49） | lindist init（5.62） |
|---|---|---|
| 50 | 7.72 | 4.31 |
| 200 | 6.35 | 3.64 |
| 500 | —（已停） | **3.20** |

lindist 臂 step 150 就超过了端到端蒸馏的 3.79；坍缩臂跑了 200 步还不如 lindist 臂的起点。**好的闭式初始化至少值 200 步（4 亿 token）的训练量，且优势在观察窗口内持续存活。**

（各个 loss 水平的模型实际生成的文本什么样——从词汤、复读机到通顺叙事——见附录 B 的采样对照，它给这些数字提供了直观刻度。）

### 5. 结论

1. **闭式赛道的最终格局**（85% 压缩率，2.29B 同预算）：

$$8.50\_{\text{坍缩假象}} \to \mathbf{5.10}\_{\text{闭式冠军}} \to \underbrace{\approx 4.0}\_{\text{逐层范式 oracle 地板}} \to 3.20\_{\text{训练@500}} \to 2.11\_{\text{教师}}$$

2. **三条可迁移的原理**：闭式压缩的正确原语是回归而不是分解（第二篇）；预算应优先花在**免截断税**的位置（lm\_head、残差流）；动手设计精巧特征之前，先**审计所有"没被压缩所以没人管"的环节**——收益最大的一击往往在盲区里。

3. **闭式极限的成因链**：实际 5.10 → 理论 4.0 之间是 block-16 型非线性放大器锁死的空间（教师原生功能对输入漂移的放大，非压缩之过）；4.0 以下必须打破逐层模仿范式——目前只有全局优化（训练）做得到，而且它确实做到了（3.20 < 4.0）。

4. **工程结论**：压缩-恢复的最优路线 = 闭式轨迹矫正 init（一次 90 分钟的 GPU 计算，5.10）+ 继续预训练。闭式研究的全部价值在于把训练起点从 8.5 拉到 5.1、把"可用模型"的到达时间提前数百步。

实际的 5.10 和 oracle 地板 4.0 之间还隔着 1.1 nat。这段空间的结构是什么、还能挖出多少，是[第四篇](/2026/08/25/closed-form-anatomy/)的主题。


---

## 附录

### 附录 A：悬崖调查——block 16 是非线性放大器

逐 block 测量残差流的线性可恢复度，发现剩余非线性漂移几乎全部来自**单独一层**：

$$R^2:\ \underbrace{0.815}\_{\text{block 16 attention 后}} \xrightarrow{\ \text{block 16 的 MLP}\ } \underbrace{0.272}\_{\text{一层砍掉 0.543}}$$

其余 35 个 block 的子层变化都在 ±0.02 量级。block 16 在教师中本就特殊：MLP 输出幅度是邻居的 2 倍，override ratio（MLP 输出相对残差流的幅度之比）0.60 vs 邻居的 0.25——这是文献中 massive activation 层（少数激活幅度异常大的层）的典型特征。

两个定点打击实验都失败了：block 16 MLP 全秩（连矫正都满血）→ 5.40，更差；前载 rank（0-16 用 480、17-35 用 298，减少流入悬崖的漂移）→ 5.38，也更差。结论：**悬崖不是压缩造成的，是教师自己的功能特性**——入口处 18% 的非线性残余误差被它的原生大幅度非线性计算放大成出口处 73% 的不可恢复纠缠。容量、分配、线性矫正三类武器全部无效；这 18% 对可负担的 rank 提升也不敏感。闭式极限就此锁定在 ~5.1。

### 附录 B：不同 loss 水平的模型在生成什么

文本采样给出了质变的直观刻度（temperature 0.8）：

| loss | 实际观感 |
|---|---|
| 8.5（坍缩） | 词汤：". and. f, to. i,.,. as the.." |
| 5.6（lindist init） | 短语碎片 + 数字循环："the 1400-1400- 1922-1131..." |
| 4.0（训练 100 步） | 语法流畅但复读机："...explorations to explore the history of mathematics that form the history of mathematics..." |
| 3.2（训练 500 步） | **复读消失**，叙事结构出现，剩余问题是事实幻觉（"Apollo 11 在月球一段 7.5 英里的区域"） |
| 2.1（教师） | 流畅且事实正确 |

复读机的消退符合"能力增长使复制策略失去 loss 优势"的预测——退化策略（坍缩、复读）是能力不足时交叉熵的理性选择，能力恢复到哪一层，对应的退化就消失到哪一层。


</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Low-Rank Compression Series (3): The Closed-Form Ceiling — Tax-Free Correctors, an Oracle Bound, and a Nonlinear Amplifier

> 📖 New to language-model vocabulary (loss, residual stream, SVD, distillation...)? Read [the primer](/2026/08/30/lord-compression-primer/) first — ten minutes covers all the background.

This concludes the low-rank compression trilogy ([part 1: representation collapse](/2026/08/17/representation-collapse-in-low-rank-compression/), [part 2: trajectory-correcting linear distillation](/2026/08/19/trajectory-correcting-linear-distillation/)). Part 2 ended with closed-form (training-free) methods at val loss 5.60. This post answers: how much further can they go, where is the hard ceiling and why, and what it all means for the "closed-form init + training" pipeline.

(All losses in this post are re-measured under the rigorous protocol: 800 validation passages × 8192 tokens, 8 folds, fold-to-fold spread about ±0.02. The training-curve losses in Section 4 come from the quick in-training evaluation, a slightly different measurement used only for within-group comparison.)

### 1. An Ablation Surprise: the Cheap Component Beats the Expensive One

Part 2's R² diagnostic blamed the remaining gap on nonlinear drift at the SwiGLU multiplication. We designed v3 accordingly: widen down\_proj's regression input to $[gu;\ \mathrm{silu}(g);\ u]$ (36864-dim) — the error cross terms $g\delta\_u + u\delta\_g$ (where $\delta$ is each factor's drift) are nonlinear in the original space but become linearly representable in the extended one; theoretically well-aimed. We also added a "free" component along the way: an lm\_head correction.

The ablation (removing components one at a time to measure each one's contribution) was a complete surprise (rank-384 base):

| Component | Gain | Cost |
|---|---|---|
| Extended features | −0.06 nat | **+340M params** |
| **lm\_head correction** | **−0.29 nat** | **≈0 params** |

The carefully engineered features underdelivered (the pure second-order term $\delta\_g\delta\_u$ remains unrepresentable even in the extended space), while the free component contributed nearly 5× more. The two are essentially additive: $5.60 - 0.29 - 0.06 = 5.26$, versus 5.23 measured for the combination — slightly better than the plain sum, a gap at the edge of the fold noise.

### 2. The Tax-Free Corrector Principle

First, a word used throughout this post: cutting a matrix to rank 384 inevitably loses some approximation accuracy; we call this unavoidable loss the **truncation tax**. All 252 low-rank layers in the network pay it.

Why is the lm\_head fix so valuable? The exit pipeline is: 36 blocks → final RMSNorm → lm\_head (4096→151936, uncompressed) → logits. Every compression method corrects only compressed layers, and lm\_head is "uncompressed, so nobody touches it" — yet its **input** has drifted. Solve a full-rank 4096×4096 regression $P$ (student final hidden → teacher final hidden) and absorb it:

$$W\_{lm}' = W\_{lm}P, \qquad b = W\_{lm}(\bar{x}\_t - P\bar{x}\_s)$$

Same shape, same FLOPs, zero new parameters (besides a 0.15M bias). Four multipliers: final-norm drift is 74% linearly recoverable ($R^2=0.742$); the correction is **full-rank** — the only spot in the network that pays no truncation tax; it is the last hop, feeding straight into logits; every token passes through it.

This generalizes into a principle: **spend budget where corrections are truncation-tax-free**. One more family of such spots exists — the residual stream between blocks. Insert a full-rank corrector $h \leftarrow Ph + b$ every 6 blocks (student residual → teacher residual; $W=I$, no truncation needed; 16.8M params each), shrinking block ranks 384→353 to stay on budget:

| Equal-budget configuration (2.29B) | val loss |
|---|---|
| v2 | 5.60 |
| + lm\_head fix | 5.31 |
| + 5 residual correctors (K=6, rank 353) | 5.11 |
| + 11 residual correctors (K=3, rank 316) | **5.10** |
| Reference: v2 rank=768 (**2× params**) | 5.12 |

Equal-budget tax-free correctors match and slightly beat brute-force rank at twice the parameters. K=6→3 buys only 0.01 more, already at the fold-noise level — the gains saturate here. **Closed-form equal-budget record: 5.10.**

The complete method of this post is Algorithm 1 from part 2 plus two extensions:

> **Algorithm 2: Tax-Free Correctors (two extensions of Algorithm 1)**
>
> **Extension 1 — residual-stream correctors**: inside Algorithm 1's loop, whenever a block's 7 matrices have all been replaced and the block index is a multiple of $K$ ($K=3$ here, 11 correctors in total), additionally:
>
> 1. Run teacher and the current student on $D$ in paired forwards; at that block's exit collect the teacher residual stream $h\_t$ and student residual stream $h\_s$; accumulate means and covariances
> 2. Solve the full-rank regression $(P, b) = \arg\min\_{P, b}\ \mathbb{E}\lVert h\_t - P h\_s - b \rVert^2$ ($P \in \mathbb{R}^{4096 \times 4096}$; the closed-form solution mirrors Algorithm 1, step 2)
> 3. Insert the correction $h \mapsto Ph + b$ after that block; later statistics automatically include its effect, keeping the corrector chain self-consistent
>
> **Extension 2 — lm_head fix**: after the loop, collect $x\_t, x\_s$ at the final RMSNorm's exit, solve the same full-rank regression $P$, and absorb it into the uncompressed output head: $W\_{lm}' = W\_{lm}P$, $b\_{lm} = W\_{lm}(\bar{x}\_t - P\bar{x}\_s)$ — same shape, same FLOPs, zero new parameters (besides a 0.15M bias)
>
> **Budget balance**: the 11 correctors cost about 185M parameters, paid for by lowering all ranks from 384 to 316; the total stays at 2.29B
>
> **Output**: student model = low-rank layers + 11 residual-stream correctors + corrected lm_head (val loss = **5.10**)

The contrast with Algorithm 1's per-matrix regression is immediate: there, $M^\*$ must be truncated to rank $r$ (pays the truncation tax); here, $P$ stays full-rank — **no rank constraint, no accuracy lost at the optimum, which is exactly what "tax-free" formalizes**.

### 3. The Oracle Bound: the Layerwise Paradigm's Floor Is ~4.0

How much closed-form room remains between 5.10 and the 3.3 that training reaches? We measured the theoretical limit of correctors directly, with an **oracle experiment** — an experiment that is allowed to cheat, used purely to establish a theoretical bound: at runtime, swap every sublayer's input tensor for the teacher's clean value (paired forwards, swapping at the q/k/v and gate/up entrances), so each layer contributes only its own local truncation error, and errors accumulate additively in the residual stream without compounding through nonlinearities.

This cheating model's loss is the **floor** for the entire family "rank-r linear layers + arbitrary input-side correctors": no corrector, however nonlinear, can do better than restoring the inputs to clean — that is, no better than the oracle. Measured:

$$\text{oracle} = 4.0 \sim 4.3$$

Even with god-tier drift correction, the summed local truncation errors of rank 384 cost 4 nats. A target of 2.5 is **provably unreachable** for this paradigm.

Simultaneously, a deeper fact emerged: **training (3.20 at step 500) has already broken through the oracle bound (4.0)**. This is logically consistent and profound: the oracle only binds the "each layer imitates its teacher counterpart" paradigm, whereas gradient training **reorganizes the division of labor across layers** — the trained layers are no longer approximations of individual teacher layers but a new factorized solution whose composition beats what layerwise imitation permits. This is the strongest form of part 2's thesis that the bottleneck is the optimization objective, not expressiveness.

The source of the remaining nonlinear drift is also located: nearly all of it comes from a **single layer** (block 16) — a nonlinear amplifier native to the teacher, immune to capacity, allocation and linear correction alike, which locks the closed-form limit at ~5.1 (investigation in Appendix A).

### 4. The Training Verdict: a Good Closed-Form Init Is Worth 200+ Steps

A two-arm controlled comparison (same seed, data order, global batch), differing only in initialization:

| step | collapsed init (8.49) | lindist init (5.62) |
|---|---|---|
| 50 | 7.72 | 4.31 |
| 200 | 6.35 | 3.64 |
| 500 | — (stopped) | **3.20** |

The lindist arm passed end-to-end distillation's 3.79 by step 150; the collapsed arm after 200 steps was still worse than the lindist arm's starting point. **A good closed-form init is worth at least 200 steps (~0.4B tokens) of training, and the advantage persists throughout the observation window.**

(What models at each loss level actually generate — from word salad through broken-record loops to fluent narrative — is shown in Appendix B, which gives these numbers a tangible scale.)

### 5. Conclusions

1. **The final closed-form landscape** (85% compression, equal 2.29B budget):

$$8.50\_{\text{collapse illusion}} \to \mathbf{5.10}\_{\text{closed-form champion}} \to \underbrace{\approx 4.0}\_{\text{layerwise oracle floor}} \to 3.20\_{\text{trained@500}} \to 2.11\_{\text{teacher}}$$

2. **Three transferable principles**: the right closed-form primitive is regression, not factorization (part 2); spend budget at **truncation-tax-free** spots (lm\_head, residual stream); before engineering clever features, **audit every "uncompressed, so unmanaged" station** — the biggest win tends to hide in the blind spot.

3. **The causal chain of the ceiling**: the space between practical 5.10 and theoretical 4.0 is locked by block-16-style nonlinear amplification (the teacher's native sensitivity to input drift — not compression's fault); going below 4.0 requires breaking the layerwise-imitation paradigm, which only global optimization does — and demonstrably did (3.20 < 4.0).

4. **The engineering takeaway**: the optimal compress-and-recover pipeline = closed-form trajectory-correcting init (one 90-minute GPU computation, 5.10) + continued pretraining. The entire value of the closed-form program is moving the training start from 8.5 to 5.1 and pulling the arrival of a usable model forward by hundreds of steps.

Between the practical 5.10 and the oracle floor of 4.0 lie 1.1 nats. What that space is made of, and how much of it can still be mined, is the subject of [part 4](/2026/08/25/closed-form-anatomy/).


---

## Appendix

### Appendix A: The Cliff — Block 16 Is a Nonlinear Amplifier

Measuring residual-stream linear recoverability block by block revealed that the remaining nonlinear drift comes almost entirely from a **single layer**:

$$R^2:\ \underbrace{0.815}\_{\text{after block-16 attention}} \xrightarrow{\ \text{block-16 MLP}\ } \underbrace{0.272}\_{\text{one layer destroys 0.543}}$$

Every other sublayer in all 36 blocks moves R² by ±0.02. Block 16 is special in the teacher itself: MLP output 2× its neighbors' magnitude, override ratio (the MLP output's magnitude relative to the residual stream) 0.60 vs their 0.25 — the signature of a massive-activation layer (one of the few layers with abnormally large activations, a documented phenomenon).

Two targeted interventions both failed: keeping block-16's MLP fully DENSE (full-rank lindist maps) → 5.40, worse; front-loading ranks (480 for blocks 0-16, 298 after, to reduce the drift flowing into the cliff) → 5.38, also worse. Conclusion: **the cliff is not caused by compression — it is the teacher's own function**: the ~18% nonlinear residue arriving at its input (insensitive to affordable rank increases) gets amplified by its native large-magnitude nonlinear computation into 73% unrecoverable entanglement at its output. Capacity, allocation, and linear correction are all powerless. The closed-form limit locks in at ~5.1.

### Appendix B: What Models at Each Loss Level Actually Generate

Text samples give the qualitative ladder (temperature 0.8):

| loss | What it reads like |
|---|---|
| 8.5 (collapsed) | word salad: ". and. f, to. i,.,. as the.." |
| 5.6 (lindist init) | phrase fragments + number loops: "the 1400-1400- 1922-1131..." |
| 4.0 (100 steps) | grammatical but a broken record: "...explorations to explore the history of mathematics that form the history of mathematics..." |
| 3.2 (500 steps) | **repetition gone**, narrative structure emerges; remaining failure mode is factual hallucination ("a 7.5-mile-long section of the moon") |
| 2.1 (teacher) | fluent and factual |

The fading of repetition matches the prediction that degenerate strategies (collapse, loops) are cross-entropy-rational responses to missing capability — as capability returns, each degeneration dissolves in order.


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
  var postTitles = {zh: '低秩压缩系列（三）：闭式压缩的天花板——免税矫正器、oracle 上界与非线性放大器', en: 'Low-Rank Compression Series (3): The Closed-Form Ceiling — Tax-Free Correctors, an Oracle Bound, and a Nonlinear Amplifier'};
  var titleEl = document.querySelector('.post-title');
  if (titleEl) titleEl.textContent = postTitles[lang];
}
switchLang('zh');
</script>
