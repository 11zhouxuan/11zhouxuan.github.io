---
title: "Low-Rank Compression Series (3): The Closed-Form Ceiling — Full-Rank Correctors and a Nonlinear Amplifier"
date: 2026-08-22
mathjax: true
sticky: 30
tags: [math, linear-algebra, LLM, compression, distillation, low-rank]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 低秩压缩系列（三）：闭式压缩的天花板——全秩矫正器与非线性放大器

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

### 2. 全秩矫正器原理

先解释本篇反复出现的一个词：把一个矩阵砍到 rank 384 必然损失一部分逼近精度，这份不可避免的损失我们称为**截断税**。全网 252 个低秩层，每层都在交税。

lm\_head 矫正为什么这么值？出口流水线是：36 个 block → final RMSNorm → lm\_head（4096→151936，未压缩）→ logits。所有压缩方法只矫正被压缩的层，而 lm\_head "没被压缩所以没人碰"——但它收到的**输入**已经漂移了。解一个 4096×4096 的全秩回归 $P$（学生 final hidden → 教师 final hidden），吸收进头部权重：

$$W\_{lm}' = W\_{lm}P, \qquad b = W\_{lm}(\bar{x}\_t - P\bar{x}\_s)$$

形状不变、计算量不变、零新增参数（除一个 0.15M 的 bias）。它强在四点：final norm 处的漂移 74% 线性可恢复（$R^2=0.742$）；矫正是**全秩**的——这是全网唯一不付截断税的位置；它是最后一跳，矫正直通 logits；每个 token 都经过它。

这提炼出一个通用原理：**把预算花在不付截断税的位置**。全网还有一类这样的位置——block 之间的残差流。每 6 个 block 插一个满秩矫正 $h \leftarrow Ph + b$（拟合学生残差 → 教师残差，$W=I$ 无需截断，16.8M 参数/个），block rank 从 384 降到 353 配平预算：

| 同预算配置（2.29B） | val loss |
|---|---|
| v2 | 5.60 |
| + lm\_head 矫正 | 5.31 |
| + 5 个残差矫正器（K=6，rank 353） | 5.11 |
| + 11 个残差矫正器（K=3，rank 316） | **5.10** |
| 对照：v2 rank=768（**2 倍参数**） | 5.12 |

同预算的全秩矫正器追平甚至略胜两倍参数的暴力加 rank。K=6→3 只再赚 0.01，已在折间波动量级——收益在此饱和。**闭式同预算纪录：5.10。**

本篇的完整方法如下。第 1–4 步就是[第二篇](/2026/08/19/trajectory-correcting-linear-distillation/)的算法 1 原封不动，**加粗的第 5 步和循环后的收尾是本篇新增**：

> **算法：轨迹矫正线性蒸馏 + 全秩矫正器（完整流程）**
>
> **输入**：教师模型的全部权重矩阵 $W\_1, \dots, W\_N$，按前向顺序编号（$N = 252$：模型共 36 个 block，每个含 7 个）、校准数据 $D$、目标秩 $r$、正则强度 $\lambda$、矫正间隔 $K$（本文 $K = 3$；为配平矫正器的参数预算，$r$ 从 384 降到 316）
>
> **对 $\ell = 1, 2, \dots, N$ 依次执行**（循环体内的变量都属于当前的 $W\_\ell$，为简洁省略编号）：
>
> 1. 教师与当前学生（前 $\ell-1$ 个矩阵已替换）在 $D$ 上配对前向，在 $W\_\ell$ 的入口采集教师输入 $x\_t$ 与学生输入 $x\_s$，累积均值 $\bar{x}\_t, \bar{x}\_s$ 和中心化协方差 $\Sigma\_{ts}, \Sigma\_{ss}$
> 2. 岭回归：$M^\* = W\_\ell \Sigma\_{ts} (\Sigma\_{ss} + \lambda I)^{-1}$
> 3. 白化截断：$L = \mathrm{chol}(\Sigma\_{ss} + \lambda I)$，对 $M^\* L$ 做 SVD 保留前 $r$ 项得 $U\_r \Sigma\_r V\_r^T$，令 $A = U\_r \Sigma\_r$、$B = V\_r^T L^{-1}$
> 4. 偏置：$b = W\_\ell \bar{x}\_t - AB\bar{x}\_s$；把 $W\_\ell$ 替换为 $x \mapsto ABx + b$
> 5. **残差流矫正器（新增）**：若 $W\_\ell$ 是某个 block 的第 7 个矩阵、且该 block 编号是 $K$ 的倍数（全网共 11 处），在该 block 出口采集教师残差流 $h\_t$ 与学生残差流 $h\_s$，解全秩回归 $(P, b\_h) = \arg\min\ \mathbb{E}\lVert h\_t - P h\_s - b\_h \rVert^2$（$P \in \mathbb{R}^{4096 \times 4096}$，闭式解同第 2 步，**不截断**），在该 block 后插入矫正步 $h \mapsto Ph + b\_h$；此后的统计采集自动包含它的效果，矫正链保持自洽
>
> **循环结束后——lm_head 矫正（新增）**：在 final RMSNorm 出口采集 $x\_t, x\_s$，同样解全秩回归 $P$，把它吸收进未压缩的输出头：$W\_{lm}' = W\_{lm}P$、$b\_{lm} = W\_{lm}(\bar{x}\_t - P\bar{x}\_s)$——形状与计算量不变，零新增参数（除 0.15M 的 bias）
>
> **输出**：学生模型 = $N$ 个低秩层 + 11 个残差流矫正器 + 矫正后的 lm_head（val loss = **5.10**；矫正器共约 185M 参数，由 $r$ 的下调支付，总参数量保持 2.29B）

新旧两类回归的关键区别一目了然：第 3 步的 $M^\*$ 必须截断到 rank $r$（付截断税），而第 5 步和收尾处的 $P$ 保持满秩——**没有 rank 约束，最优解不损失任何精度，"全秩矫正器"的价值全在这里**。

### 3. 剩余的差距在哪：几乎全在 block 16

5.10 之后我们继续追问：剩下的误差到底在哪里产生？做法很直接——沿着模型逐个 block 检查，问"走到这里为止，学生和教师的差距还有多少能用一个线性变换修回去"。答案非常集中：修不掉的部分几乎全部产生自**同一个位置**——block 16 的 FFN。

这个层在教师里本来就特殊：输出幅度是相邻层的两倍，属于文献里说的 massive activation 层（少数激活值特别大的层）。它对输入误差极其敏感——入口处一点修不掉的小误差，经过它的大幅度非线性计算就变成出口处的大误差：一个**非线性放大器**。我们试了三种补救——不压缩它、调整 rank 分配、加线性矫正——全部无效（实验细节见附录 A）。闭式方法的极限因此停在约 5.1 附近。

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

$$8.50\_{\text{坍缩假象}} \to \mathbf{5.10}\_{\text{闭式冠军}} \to 3.20\_{\text{训练@500}} \to 2.11\_{\text{教师}}$$

2. **三条可迁移的原理**：闭式压缩的正确原语是回归而不是分解（第二篇）；预算应优先花在**不付截断税**的位置（lm\_head、残差流）；动手设计精巧特征之前，先**审计所有"没被压缩所以没人管"的环节**——收益最大的一击往往在盲区里。

3. **闭式极限的成因**：卡在约 5.1 的直接原因是 block 16——教师原生的一个对输入误差极其敏感的层，把线性修不掉的小误差放大成大误差（这是教师自己的特性，不是压缩的错）。要低于它，就得放弃"每层模仿教师对应层"的做法、直接优化最终 loss——这正是训练在做的事。

4. **工程结论**：压缩-恢复的最优路线 = 闭式轨迹矫正 init（一次 90 分钟的 GPU 计算，5.10）+ 继续预训练。闭式研究的全部价值在于把训练起点从 8.5 拉到 5.1、把"可用模型"的到达时间提前数百步。

5.10 是不是闭式的尽头？把剩下的差距拆开、看看还能从哪里再挤一点，是[第四篇](/2026/08/25/closed-form-anatomy/)的主题。


---

## 附录

### 附录 A：block 16 调查——闭式极限为什么停在 5.1

**怎么测**。教师和学生读同样的文本，在每个 block 的出口比较两边的残差流，用回归的 $R^2$ 度量"学生的残差流还有多少能用一个线性变换拉回教师的样子"（$R^2$ 从 0 到 1，越接近 1 表示越能修回去）。

**结果非常集中**。36 个 block 里，其余 35 个的每个子层对 $R^2$ 的影响都在 ±0.02 以内；唯独经过 block 16 的 FFN 时，$R^2$ 从 0.815 掉到 0.272——"修不回去"的部分有一大半产生在这一个位置。

**这个层在教师里本来就特殊**：它的 FFN 输出幅度是相邻层的 2 倍，输出大小达到残差流本身的 0.60 倍（邻居只有 0.25）。文献里把这类层叫 massive activation 层——少数激活值特别大的层，很多大模型里都存在。

**两个针对它的补救实验都失败了**：

- 完全不压缩 block 16 的 FFN（保留原始全秩权重）：5.40，比 5.10 更差；
- 给前半段多分 rank（block 0–16 用 480、17–35 用 298），想让流进 block 16 的误差更小：5.38，也更差。

**结论**：不是我们把 block 16 压坏了，而是教师的这个层天生就会放大输入里的误差。走到它入口时，误差里约有 18% 是线性变换修不掉的；经过它的大幅度非线性计算之后，出口处修不掉的比例被放大到 73%。给它容量、给它 rank、给它线性矫正都碰不到这个机制，那 18% 对我们付得起的 rank 提升也不敏感。闭式方法的极限因此停在约 5.1。

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

## Low-Rank Compression Series (3): The Closed-Form Ceiling — Full-Rank Correctors and a Nonlinear Amplifier

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

### 2. The Full-Rank Corrector Principle

First, a word used throughout this post: cutting a matrix to rank 384 inevitably loses some approximation accuracy; we call this unavoidable loss the **truncation tax**. All 252 low-rank layers in the network pay it.

Why is the lm\_head fix so valuable? The exit pipeline is: 36 blocks → final RMSNorm → lm\_head (4096→151936, uncompressed) → logits. Every compression method corrects only compressed layers, and lm\_head is "uncompressed, so nobody touches it" — yet its **input** has drifted. Solve a full-rank 4096×4096 regression $P$ (student final hidden → teacher final hidden) and absorb it:

$$W\_{lm}' = W\_{lm}P, \qquad b = W\_{lm}(\bar{x}\_t - P\bar{x}\_s)$$

Same shape, same FLOPs, zero new parameters (besides a 0.15M bias). Four multipliers: final-norm drift is 74% linearly recoverable ($R^2=0.742$); the correction is **full-rank** — the only spot in the network that pays no truncation tax; it is the last hop, feeding straight into logits; every token passes through it.

This generalizes into a principle: **spend budget where corrections pay no truncation tax**. One more family of such spots exists — the residual stream between blocks. Insert a full-rank corrector $h \leftarrow Ph + b$ every 6 blocks (student residual → teacher residual; $W=I$, no truncation needed; 16.8M params each), shrinking block ranks 384→353 to stay on budget:

| Equal-budget configuration (2.29B) | val loss |
|---|---|
| v2 | 5.60 |
| + lm\_head fix | 5.31 |
| + 5 residual correctors (K=6, rank 353) | 5.11 |
| + 11 residual correctors (K=3, rank 316) | **5.10** |
| Reference: v2 rank=768 (**2× params**) | 5.12 |

Equal-budget full-rank correctors match and slightly beat brute-force rank at twice the parameters. K=6→3 buys only 0.01 more, already at the fold-noise level — the gains saturate here. **Closed-form equal-budget record: 5.10.**

The complete method of this post is stated below. Steps 1–4 are exactly Algorithm 1 from [part 2](/2026/08/19/trajectory-correcting-linear-distillation/), unchanged; **the bold step 5 and the post-loop finish are new in this post**:

> **Algorithm: Trajectory-Correcting Linear Distillation + Full-Rank Correctors (complete pipeline)**
>
> **Input**: all of the teacher's weight matrices $W\_1, \dots, W\_N$ in forward order ($N = 252$: 36 blocks with 7 matrices each), calibration data $D$, target rank $r$, regularization strength $\lambda$, corrector interval $K$ ($K = 3$ here; to pay for the correctors, $r$ drops from 384 to 316)
>
> **For $\ell = 1, 2, \dots, N$** (inside the loop, all variables belong to the current $W\_\ell$; the index is dropped for brevity):
>
> 1. Run teacher and the current student (matrices $1..\ell-1$ already replaced) on $D$ in paired forwards; at $W\_\ell$'s entrance collect the teacher input $x\_t$ and student input $x\_s$; accumulate the means $\bar{x}\_t, \bar{x}\_s$ and centered covariances $\Sigma\_{ts}, \Sigma\_{ss}$
> 2. Ridge regression: $M^\* = W\_\ell \Sigma\_{ts} (\Sigma\_{ss} + \lambda I)^{-1}$
> 3. Whitened truncation: $L = \mathrm{chol}(\Sigma\_{ss} + \lambda I)$; take the top-$r$ SVD of $M^\* L$ as $U\_r \Sigma\_r V\_r^T$; set $A = U\_r \Sigma\_r$, $B = V\_r^T L^{-1}$
> 4. Bias: $b = W\_\ell \bar{x}\_t - AB\bar{x}\_s$; replace $W\_\ell$ with $x \mapsto ABx + b$
> 5. **Residual-stream corrector (new)**: if $W\_\ell$ is the 7th matrix of its block and the block index is a multiple of $K$ (11 spots in total), collect the teacher residual stream $h\_t$ and student residual stream $h\_s$ at that block's exit, solve the full-rank regression $(P, b\_h) = \arg\min\ \mathbb{E}\lVert h\_t - P h\_s - b\_h \rVert^2$ ($P \in \mathbb{R}^{4096 \times 4096}$, closed form as in step 2, **no truncation**), and insert the correction $h \mapsto Ph + b\_h$ after that block; later statistics automatically include its effect, keeping the corrector chain self-consistent
>
> **After the loop — lm_head fix (new)**: collect $x\_t, x\_s$ at the final RMSNorm's exit, solve the same full-rank regression $P$, and absorb it into the uncompressed output head: $W\_{lm}' = W\_{lm}P$, $b\_{lm} = W\_{lm}(\bar{x}\_t - P\bar{x}\_s)$ — same shape, same FLOPs, zero new parameters (besides a 0.15M bias)
>
> **Output**: student model = $N$ low-rank layers + 11 residual-stream correctors + corrected lm_head (val loss = **5.10**; the correctors cost about 185M parameters, paid for by the lower $r$; the total stays at 2.29B)

The contrast between the two kinds of regression is immediate: the $M^\*$ of step 3 must be truncated to rank $r$ (pays the truncation tax), while the $P$ of step 5 and the finish stays full-rank — **no rank constraint, no accuracy lost at the optimum: that is the entire value of a full-rank corrector**.

### 3. Where the Remaining Gap Lives: Almost Entirely in Block 16

After 5.10 we kept asking: where is the remaining error actually created? The probe is simple — walk through the model block by block and ask "up to this point, how much of the student–teacher gap can still be repaired by one linear map?" The answer is extremely concentrated: nearly all of the unrepairable part is created at **one spot** — block 16's FFN.

That layer is special in the teacher to begin with: its output is twice its neighbors' magnitude, one of the massive-activation layers documented in the literature (a few layers with unusually large activations). It is hypersensitive to input error — a small unrepairable residue at its entrance comes out amplified into a large one: a **nonlinear amplifier**. Three rescues — leaving it uncompressed, reallocating rank, adding linear correction — all fail (details in Appendix A). The closed-form limit therefore stops near ~5.1.

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

$$8.50\_{\text{collapse illusion}} \to \mathbf{5.10}\_{\text{closed-form champion}} \to 3.20\_{\text{trained@500}} \to 2.11\_{\text{teacher}}$$

2. **Three transferable principles**: the right closed-form primitive is regression, not factorization (part 2); spend budget where **no truncation tax is paid** (lm\_head, residual stream); before engineering clever features, **audit every "uncompressed, so unmanaged" station** — the biggest win tends to hide in the blind spot.

3. **Why the ceiling sits where it does**: the direct cause of the ~5.1 limit is block 16 — a layer in the teacher that is natively hypersensitive to input error, amplifying the small linearly-unrepairable residue into a large one (the teacher's own trait, not compression's fault). Going lower means abandoning "each layer imitates its teacher counterpart" and optimizing the final loss directly — which is exactly what training does.

4. **The engineering takeaway**: the optimal compress-and-recover pipeline = closed-form trajectory-correcting init (one 90-minute GPU computation, 5.10) + continued pretraining. The entire value of the closed-form program is moving the training start from 8.5 to 5.1 and pulling the arrival of a usable model forward by hundreds of steps.

Is 5.10 the end of the closed-form road? Taking the remaining gap apart to see where a little more can be squeezed out is the subject of [part 4](/2026/08/25/closed-form-anatomy/).


---

## Appendix

### Appendix A: The Block-16 Investigation — Why the Closed-Form Limit Stops at 5.1

**How we measured.** Teacher and student read the same text; at every block's exit we compare the two residual streams and use a regression's $R^2$ to quantify "how much of the student's residual stream can be pulled back to the teacher's by one linear map" ($R^2$ runs from 0 to 1; closer to 1 means more repairable).

**The result is extremely concentrated.** Of the 36 blocks, every sublayer in the other 35 moves $R^2$ by at most ±0.02; crossing block 16's FFN alone drops it from 0.815 to 0.272 — more than half of all the unrepairable error is created at this single spot.

**The layer is special in the teacher to begin with**: its FFN output is 2× its neighbors' magnitude and reaches 0.60 of the residual stream's own size (neighbors: 0.25). The literature calls these massive-activation layers — a few layers with unusually large activations, present in many large models.

**Two targeted rescue attempts both failed**:

- Leave block 16's FFN entirely uncompressed (original full-rank weights): 5.40, worse than 5.10;
- Give the first half more rank (480 for blocks 0–16, 298 after) so that less error flows into block 16: 5.38, also worse.

**Conclusion**: we did not break block 16 by compressing it — this teacher layer natively amplifies whatever error reaches it. At its entrance, about 18% of the error cannot be removed by any linear map; after its large-magnitude nonlinear computation, the unremovable share at the exit grows to 73%. Capacity, rank allocation, and linear correction all miss this mechanism, and the 18% barely responds to any rank increase we can afford. That is why the closed-form limit stops at ~5.1.

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
  var postTitles = {zh: '低秩压缩系列（三）：闭式压缩的天花板——全秩矫正器与非线性放大器', en: 'Low-Rank Compression Series (3): The Closed-Form Ceiling — Full-Rank Correctors and a Nonlinear Amplifier'};
  var titleEl = document.querySelector('.post-title');
  if (titleEl) titleEl.textContent = postTitles[lang];
}
switchLang('zh');
</script>
