---
title: "Low-Rank Compression Series (4): Anatomy of the Ceiling — an Oracle Decomposition of the Remaining Gap"
date: 2026-08-25
mathjax: true
sticky: 20
tags: [math, linear-algebra, LLM, compression, distillation, low-rank, oracle-bound, rank-allocation]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 低秩压缩系列（四）：天花板的解剖——用 oracle 分解剩余差距

> 📖 如果你不熟悉语言模型的基本词汇（loss、残差流、SVD、蒸馏……），建议先读[预备知识篇](/2026/08/30/lord-compression-primer/)，10 分钟即可补齐全部背景。

[三部曲的完结篇](/2026/08/22/closed-form-ceiling/)宣布闭式赛道在 5.10 收官。然后我们又跑了四个实验方向，把等预算纪录推到 **5.05**，并且这一次拿到了"真的到头了"的结构性证据。这篇讲这 0.05 nat 是怎么挣的——以及一路上收获的、比 0.05 值钱得多的四个机制发现：loss 敏感度的模式/内容通路二分、RMSNorm×巨通道的乘性放大器、**截断税的深度分布**（一个 oracle 分解实验），和"小改进不叠加"的规律。

背景（一句话版）：Qwen3-8B（val loss 2.11）→ 每个 linear 换 rank-384 的 $AB$（2.29B，删 72% 参数），闭式方法 = 逐层轨迹矫正回归 + lm_head 矫正 + 残差流矫正器，起点 5.10。

（本文的 loss 均为严格协议重测值：800 段 × 8192 token、8 折，折间波动约 ±0.02。第 3 节 oracle 分解的诊断数字是一次性实验、保持当时的测量口径，只用于相对比较。）

### 1. 差距的三个组成部分：5.06 到 2.11 之间差在哪里

在动手之前先回答一个架构级的问题：闭式为什么恰好卡在 5 附近？把全部实验证据（oracle 地板、block-16 诊断、消融）拼起来，从教师到闭式的 ~2.9 nat 可以拆成三个**互相独立、机制不同**的组成部分：

$$2.11 \xrightarrow{\text{①截断税} \approx 2.0} 4.15\_{\text{oracle}} \xrightarrow{\text{②漂移残余} \approx 0.9} 5.06 \qquad 3.2\_{\text{训练}} \xleftarrow{\text{③重组收益}}$$

（"oracle 地板"来自一个允许作弊的诊断实验：运行时把每个子层的输入替换成教师的干净值，于是每层只贡献自己的局部截断误差。这个作弊模型的 loss（测得 4.15）就是"每层模仿教师对应层"这条路线无论怎么矫正都到不了的下限——任何矫正器最多把输入还原干净，不可能做得比干净输入更好。）

**① 截断税（最大头）**：即使每个子层都收到教师的干净输入，rank-384 的局部截断误差之和也要付 ~2 nat。为什么这么贵？白化谱给出答案——下表是 rank 384 下各类矩阵保留的能量（白化坐标里奇异值平方和的占比，即最优截断能留住多大比例的"有效信号"）：

| 矩阵 | 保留能量@384 | 达到 95% 需要的 rank |
|---|---|---|
| k_proj | 98.0% | 211 |
| q_proj | 94.7% | 402 |
| v_proj | 93.4% | 432 |
| o_proj | 89.7% | 642 |
| gate_proj | 89.5% | 687 |
| up_proj | 82.9% | 983 |
| **down_proj** | **66.2%** | **1839** |

网络分成两条通路：**模式通路**（q/k → softmax）本质低秩且对漂移鲁棒（softmax 饱和，attention 分布接近 argmax）；**内容通路**（v→o、up→down 的写路径）**谱衰减极慢、用满全宽**——down_proj 要把 12288 维乘积空间里以"超位置"方式存储的特征（superposition：把远多于维度数的特征重叠地存在同一组维度里）解码回残差流，这样的解码矩阵天生没有低秩结构可利用。这部分损失是信息论性质的，对"更好的逼近算法"免疫。

**② 漂移残余**：实际 5.06 与 oracle 4.15 之间，是穿过非线性复合后线性矫正器修不掉的漂移，载体是两个乘性非线性——SwiGLU 的 $\delta\_g\delta\_u$ 二阶项和 RMSNorm 的除法（附录 B 有它的精确机制）。

**③ 重组收益**：训练不受这个地板约束，因为它从不要求每层模仿教师对应层——残差架构下最终计算是 36 个 block 贡献的**和**，同一个总函数有无数种拆成 36 份的方式，梯度下降可以找到比"模仿教师原有分工"更适合 rank-384 的新分工。逐层模仿的目标从定义上就拿不到这部分收益。

三个组成部分对应三个不同的瓶颈。本篇的四个实验方向全部发生在 ① 和 ② 的边界上。

### 2. 实验方向一：loss 敏感度驱动的 rank 分配——方向对，但要阻尼

我们之前试过从白化谱做分配（重构误差驱动），失败（5.73）。文献一致认可的是 **loss 敏感度驱动**（六篇独立工作同向；调研坐标见附录 A）——这是两回事：前者问"砍掉这个方向，本层输出变多少"，后者问"砍掉这个方向，最终 loss 变多少"。做法：在每个保留的 rank-1 方向上挂一个乘性门 $\alpha\_i=1$，反传时只求门的梯度，累积 $f\_i = \sum(\partial L/\partial\alpha\_i)^2$（梯度平方和，Fisher 信息的对角近似，衡量 loss 对该方向的敏感度）；用尾部的 $f$ 校准白化谱（$f\_{l,i}\approx c\_l\sigma\_{l,i}^2$），然后在总预算固定的约束下把 rank 优先分给敏感度高的层，直到预算用完。

Fisher 给出的分配极其激进，且完美对应模式/内容通路二分：

| 矩阵 | Fisher 最优 rank（均值） | 对均匀 316 |
|---|---|---|
| o_proj / v_proj | 709 / 637 | ↑ 2 倍 |
| down_proj | 476 | ↑ 1.5 倍 |
| q_proj / k_proj / gate_proj | 77 / 92 / 128 | ↓ 3-4 倍 |

**照单全收 → 5.27（失败）**；把偏离限制在均匀值的 0.6~1.6 倍（即"阻尼"）→ **5.06（当时的新纪录）**。教训：这种敏感度是局部量——它只回答"在当前解附近动一动会怎样"，砍 4 倍已经远超它的适用范围；三点剂量曲线（不倾斜 5.10 / 阻尼 5.06 / 极端 5.27）显示阻尼点就在最优附近。方向本身的机制解释很干净：q/k 只决定经 softmax 归一化的 attention 权重，天然抗漂移；v/o/down 直接携带写进残差流的内容。

**把本篇的改动写成公式**。前三篇里每层的 rank 是固定常数 $r$；本篇把每层的 rank $\{r\_\ell\}$ 变成待求的变量：

$$\min\_{\{r\_\ell\}}\ \sum\_\ell \underbrace{\sum\_{i > r\_\ell} f\_{\ell,i}}\_{\text{层 }\ell\text{ 被砍方向的预期损害}} \qquad \text{s.t.}\quad \sum\_\ell r\_\ell (m\_\ell + n\_\ell) = \text{预算}, \qquad \underbrace{r\_\ell \in [0.6\bar{r},\ 1.6\bar{r}]}\_{\text{阻尼约束（不可省略）}}$$

其中 $f\_{\ell,i}$ 是第 $\ell$ 层第 $i$ 个方向的 loss 敏感度（前文的梯度平方和）。没有阻尼约束时这个问题的解就是"照单全收"的 5.27；加上它才得到 5.06——**约束本身是改进的一半**。

实验方向二、三都是干净的负结果，但各自揭示了机制，细节在附录：巨激活通道天然不付截断税（**附录 B**——顺带定位了"②漂移残余"的放大器：RMSNorm 的分母被两条巨通道主导，它们一漂移、全部 4096 维跟着乘性重缩放）；rms-lift 非线性矫正器能修乘性漂移，但参数效率与直接加 rank 持平（**附录 C**）。主线于是回到 rank 的放置：

### 3. 实验方向四：oracle 分解——截断税住在哪些 block 里

[第三篇](/2026/08/22/closed-form-ceiling/)的完整 oracle（把全部子层输入换成教师的干净值）只给出一个总量。把它拆开——每次只换一部分输入（本表是一次性诊断实验，用当时的评估口径，看相对差值即可）：

| 换什么 | loss | 相对基线 5.13 |
|---|---|---|
| 全部（完整 oracle） | 4.15 | +0.98 |
| 只换 q/k/v 入口 | 4.51 | +0.62 |
| 只换 gate/up 入口 | 4.57 | +0.56 |
| **只换 block 18-35** | **4.14** | **+0.99 ≈ 完整 oracle！** |
| 只换 block 0-17 | 5.53 | **−0.40（变差）** |
| 只换 block 12-19（block 16 附近） | 5.49 | −0.36（变差） |

两个重磅结论：

**其一：截断税几乎全部集中在后半段。** 只把 block 18-35 的输入换干净，loss 就到了完整 oracle 的水平——意味着前半段的局部截断误差对 logits 的直接贡献只有 ~0.01 nat，它的全部危害走"制造漂移"这条间接通道；而漂移是（部分）可矫正的，后半段的截断税则无药可救。**推论：边际 rank 在后半段的价值远高于前半段。** 我们此前试过的恰好是反方向（front-load，5.38 失败——当时的理论"减少流进 block 16 的漂移"错得离谱）。

**其二：局部"清洁"上游反而伤害下游（−0.40）。** 顺序管线里每一层和矫正器都是**适配上游特定误差模式**拟合的，突然给它们干净的中间值，下游的补偿全部失配。这是矫正链自洽性的第三次独立验证（前两次：不动点迭代 5.95、跨层联合优化的文献负结果）。

据此做 back-load（把 rank 往后半段倾斜）：前半 rank ×0.8、后半按预算配平放大（×1.24）。**5.05——最终纪录**，剂量曲线在 ×0.8 附近最优（不倾斜和 ×0.65 都略差）。不过要诚实标注幅度：它比阻尼分配的 5.06 只低约 0.014，与折间波动（±0.015）同量级——方向与 oracle 分解的预言一致，但收益已经小到接近测量精度。

最后一个实验是把两个已验证的小改进叠加（back-load + rms-lift-256）：**失败，比单独 back-load 还差**。0.02 级的改进在同一块残余误差上重叠收割，合并后互相侵蚀。

### 4. 结论：三个瓶颈、三条出路与三条经验

**最终版图**（85% 压缩率、等预算 2.29B）：

$$8.50\_{\text{坍缩假象}} \to 5.60\_{\text{轨迹矫正}} \to 5.10\_{\text{免税矫正器}} \to \mathbf{5.05}\_{\text{+分配+back-load}} \to 4.15\_{\text{oracle}} \to 3.2\_{\text{训练@500}} \to 2.11\_{\text{教师}}$$

| 瓶颈 | 架构根源 | 出路 |
|---|---|---|
| 截断税 ~2.0（集中在后半段） | 内容通路的超位置存储、谱衰减极慢 | 改变 rank 的放置方式（跨矩阵联合分解、稀疏残差等）——在现有范式之外 |
| 漂移残余 ~0.9 | SwiGLU 乘法 + RMSNorm×巨通道 | 非线性矫正器可修，但参数效率与加 rank 持平——已榨干 |
| 重组收益 ~0.9 | 残差和的可重新分工 | 只有全局优化（训练） |

三条从这轮实验里提炼的、可迁移的经验：

1. **oracle 分解是比 oracle 本身强得多的诊断工具。** 完整 oracle 只告诉你天花板在哪，部分 oracle 告诉你**误差住在哪**——"late ≈ all"一个数字就推翻了我们持有数周的 front-load 直觉，并直接换来了新纪录。
2. **矫正链是一个自洽整体。** 顺序拟合的压缩管线里，每个组件都适配上游的特定误差。任何"局部改善"（不动点重解、跨层联合目标、部分输入清洁）都会破坏下游补偿——三次独立验证。诊断时可以作弊，部署时只能整链重拟合。
3. **小改进不叠加。** 当各手段（分配、直通、非线性 lift）都在收割同一块残余误差时，0.02 级的收益合并后互相侵蚀。判断两个改进能否叠加，看它们的机制是否攻击差距的不同组成部分——攻击同一部分的改进，合并前先打折。

闭式赛道就此真正收官：等预算 ~5.05 是"逐层回归 + 线性/轻非线性矫正 + rank 分配"这套工具箱的终点。通往 2.5 的路在训练上（本文发稿时训练臂已到 2.87 且在下降），通往 4.15 以下的闭式路在范式外。

第三篇宣布过一次收官，被本篇推翻；本篇的收官宣言又能站多久？[第五篇](/2026/08/30/closed-form-moving-ceiling/)见分晓。




---

## 附录

### 附录 A：文献坐标——我们在前沿的什么位置

动手前做了一轮系统调研（2024-2026 的 training-free 低秩压缩文献），两个结论校准了预期：

1. **没有任何已发表的 training-free 方法在 70%+ 参数削减下报告过可用结果。** 纯闭式的最远数据点是 SVD-LLM 在 LLaMA-7B 删 60% 时 PPL 53.7——折成 loss 约 4.0，与我们的 oracle 界惊人吻合：文献从侧面证实这堵墙的量级是真的。我们在删 72% 拿 5.0x，在前沿之上。
2. **非均匀 rank 分配是文献中最一致的收益来源**（Dobi-SVD、SVD-LLM v2、D-Rank、UniRank、AIR、LACE-SVD 六篇独立工作同向）：高压缩下"分配比分解本身更重要"。这直接触发了实验方向一。

### 附录 B：实验方向二——巨激活通道的双负结果与真正的放大机制

测量教师的输出侧异常，发现巨幅激活（massive activations，[第三篇](/2026/08/22/closed-form-ceiling/)附录 A 提过的现象）在这个模型里是**两条贯穿全网的全局通道**：维度 1838 和 2276，幅度是中位通道的 60-110 倍，从 block 0 单调增长到 block 33（mean|x| 从 4.9 涨到 607）；几乎每一层 o_proj/down_proj 的最大输出行都对准它们。

顺着第三篇"免税矫正器"的思路做了两个等预算直通实验：巨输入列不进 SVD 截断（每层 top-24 列原样保留）、巨输出行不进截断（o/down 的 top-8 行）。**双双打平。** 这个负结果反而更有价值：白化 SVD 的目标函数本来就被巨行巨列主导，最靠前的奇异方向**天然优先服务它们**——巨通道根本不付截断税。这与 lm_head/残差矫正器的成功形成干净对比：那两处是整个矩阵没人矫正，这里是矩阵内部早已被照顾好的通道。

但测量顺带揭示了"②漂移残余"的精确机制：**RMSNorm 的分母被这两维主导——它们一漂移，全部 4096 维跟着乘性重缩放**。这是一个把局部小误差全局化的耦合器，也是第三篇发现的 block 16 放大器（一层砍掉残差流 R² 0.54）的本体。伤害全部来自漂移的流动而非权重的近似——静态权重侧的任何安排（dense、front-load、直通，四连败）都碰不到它。

### 附录 C：实验方向三——rms-lift 非线性矫正器，机制证实、收益归零

既然放大器是 RMSNorm 的乘性耦合，对症的做法就是给残差矫正器一个能表达"缩放校正"的特征空间：把回归输入从 $h$ 提升到 $[h;\ h\cdot(\bar r/\mathrm{rms}(h)-1)]$——第二块是每 token 的**尺度异常分量**。对参数仍然线性，ridge 依旧闭式可解。

两个教训先于结果到来。第一枪用了朴素形式 $[h;\ h/\mathrm{rms}(h)]$：两块特征高度共线且尺度差百倍，ridge 给出一对巨大且互相抵消的权重——校准集上的 R² 暴涨、held-out 反而变差（典型的过拟合信号）。**lift 特征必须中心化 + 尺度匹配。** 修好条件数（异常分量形式 + $\lambda=10^{-3}$）后：

| 配置 | val loss | 判决 |
|---|---|---|
| 线性矫正器（基线） | 5.06 | |
| rms-lift 全秩（**+8% 参数**） | **4.99** | 闭式首次低于 5.0；block 16 之后的矫正器 R² 0.36→0.66 |
| rms-lift 截断 rank-256（等预算） | 5.07 | 打平 |

**机制证实，收益归零**：乘性漂移确实可修（R² 提升与低于 5.0 为证），但修复它的参数效率与直接加 rank 恰好持平——等预算下没有额外便宜可占。这与免税矫正器原理形成闭环：lm_head 和线性残差矫正器赢在修复"没人管的漂移"，而 lift 的非线性部分在修复"已被线性工具照顾过的残余"，边际收益自然持平。


</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Low-Rank Compression Series (4): Anatomy of the Ceiling — an Oracle Decomposition of the Remaining Gap

> 📖 New to language-model vocabulary (loss, residual stream, SVD, distillation...)? Read [the primer](/2026/08/30/lord-compression-primer/) first — ten minutes covers all the background.

[The trilogy finale](/2026/08/22/closed-form-ceiling/) declared the closed-form track finished at 5.10. We then ran four more experimental directions, pushed the equal-budget record to **5.05**, and — this time — obtained structural evidence that it is truly over. This post covers how that 0.05 nat was earned, and four mechanistic findings worth far more than the 0.05: the pattern/content pathway split in loss sensitivity, the RMSNorm×massive-channel multiplicative amplifier, **the depth distribution of the truncation tax** (an oracle-decomposition experiment), and the rule that small improvements don't stack.

Background in one line: Qwen3-8B (val loss 2.11) → every linear replaced by rank-384 $AB$ (2.29B params, 72% removed); closed-form = layerwise trajectory-correcting regression + lm_head fix + residual-stream correctors, starting at 5.10.

(All losses in this post are re-measured under the rigorous protocol: 800 validation passages × 8192 tokens, 8 folds, fold-to-fold spread about ±0.02. The oracle-decomposition diagnostics in Section 3 are one-off experiments kept in their original measurement window, used for relative comparison only.)

### 1. Three Components of the Gap: What Separates 5.06 from 2.11

Before running anything, an architecture-level question: why is closed-form stuck near 5? Assembling all the evidence (oracle floor, block-16 diagnosis, ablations), the ~2.9 nats from teacher to closed-form split into three **independent components with distinct mechanisms**:

$$2.11 \xrightarrow{\text{① truncation tax} \approx 2.0} 4.15\_{\text{oracle}} \xrightarrow{\text{② drift residue} \approx 0.9} 5.06 \qquad 3.2\_{\text{trained}} \xleftarrow{\text{③ reorganization gain}}$$

(The "oracle floor" comes from a diagnostic experiment that is allowed to cheat: at runtime, every sublayer's input is replaced with the teacher's clean value, so each layer contributes only its own local truncation error. This cheating model's loss — measured at 4.15 — is a lower limit that the "each layer imitates its teacher counterpart" route can never beat, however good the correction: no corrector can do better than restoring the inputs to clean.)

**① Truncation tax (the bulk)**: even if every sublayer received the teacher's clean input, the summed local truncation errors of rank 384 cost ~2 nats. Why so expensive? The whitened spectra answer — the table shows the energy retained at rank 384 per matrix family (the share of squared singular values in whitened coordinates, i.e. how much of the usable signal an optimal truncation can keep):

| Matrix | Energy kept @384 | Rank needed for 95% |
|---|---|---|
| k_proj | 98.0% | 211 |
| q_proj | 94.7% | 402 |
| v_proj | 93.4% | 432 |
| o_proj | 89.7% | 642 |
| gate_proj | 89.5% | 687 |
| up_proj | 82.9% | 983 |
| **down_proj** | **66.2%** | **1839** |

The network splits into two pathways: the **pattern pathway** (q/k → softmax) is intrinsically low-rank and drift-robust (softmax saturates; attention distributions are near-argmax), while the **content pathway** (v→o, up→down write path) has **slowly decaying spectra using the full width** — down_proj must decode features stored in superposition (packing far more features than dimensions, overlapped in the same coordinates) in the 12288-dim product space back into the residual stream, and such decoding matrices have no low-rank structure to exploit. This loss is information-theoretic; it is immune to "a better approximation algorithm".

**② Drift residue**: the gap between the practical 5.06 and the oracle 4.15 is drift that survives linear correction after compounding through nonlinearities, carried by two multiplicative nonlinearities — SwiGLU's second-order $\delta\_g\delta\_u$ term and RMSNorm's division (its precise mechanism in Appendix B).

**③ Reorganization gain**: training is not bound by this floor, because it never asks each layer to imitate its teacher counterpart — in a residual architecture the final computation is a **sum** of 36 block contributions; the same total function admits infinitely many decompositions, and gradient descent can find one better suited to rank 384 than the teacher's own. Layerwise-imitation objectives cannot capture this gain by definition.

Three components, three different bottlenecks. All four experimental directions below operate on the boundary between ① and ②.

### 2. Direction 1: Loss-Aware Rank Allocation — Right Direction, Needs Damping

We had previously tried spectrum-driven allocation (reconstruction-error-based) and failed (5.73). What the literature consistently endorses is **loss-sensitivity-driven** allocation (six independent works agree; the survey coordinates are in Appendix A) — a different thing: the former asks "how much does this layer's output change if I cut this direction," the latter asks "how much does the final loss change." Method: hang a multiplicative gate $\alpha\_i=1$ on every kept rank-1 direction, backprop into the gates only, accumulate $f\_i = \sum(\partial L/\partial\alpha\_i)^2$ (a sum of squared gradients — the diagonal approximation of Fisher information, measuring the loss's sensitivity to that direction); calibrate the whitened spectra with the tail values ($f\_{l,i}\approx c\_l\sigma\_{l,i}^2$), then hand rank to the most sensitive layers first until the fixed budget runs out.

Fisher's optimal allocation is drastic, and maps perfectly onto the pattern/content split:

| Matrix | Fisher-optimal rank (mean) | vs uniform 316 |
|---|---|---|
| o_proj / v_proj | 709 / 637 | ↑ 2× |
| down_proj | 476 | ↑ 1.5× |
| q_proj / k_proj / gate_proj | 77 / 92 / 128 | ↓ 3-4× |

**Taken at face value → 5.27 (fail)**; deviation clipped to 0.6-1.6× of uniform ("damping") → **5.06 (record at the time)**. Lesson: this sensitivity is a local quantity — it only answers "what happens if you wiggle around the current solution"; a 4× cut is far outside its domain. The three-point dose-response (untilted 5.10 / damped 5.06 / extreme 5.27) puts the optimum near the damped point.

**This post's change, written as a formula.** In the first three posts every layer's rank is a fixed constant $r$; this post turns the per-layer ranks $\{r\_\ell\}$ into unknowns to be solved for:

$$\min\_{\{r\_\ell\}}\ \sum\_\ell \underbrace{\sum\_{i > r\_\ell} f\_{\ell,i}}\_{\text{expected damage of layer }\ell\text{'s cut directions}} \qquad \text{s.t.}\quad \sum\_\ell r\_\ell (m\_\ell + n\_\ell) = \text{budget}, \qquad \underbrace{r\_\ell \in [0.6\bar{r},\ 1.6\bar{r}]}\_{\text{damping constraint (not optional)}}$$

where $f\_{\ell,i}$ is direction $i$'s loss sensitivity in layer $\ell$ (the squared-gradient sums above). Without the damping constraint the solution is the face-value 5.27; with it, 5.06 — **the constraint is half the improvement**. The direction has a clean mechanism: q/k only shape attention weights, normalized by softmax and thus naturally drift-robust; v/o/down carry the content written into the residual stream.

Directions 2 and 3 are both clean negatives, each revealing a mechanism — details in the appendices: massive-activation channels pay no truncation tax to begin with (**Appendix B** — which also located the amplifier behind ② drift residue: RMSNorm's denominator is dominated by two giant channels; when they drift, all 4096 dimensions get multiplicatively rescaled); rms-lift nonlinear correctors do repair multiplicative drift, but at parameter efficiency exactly on par with just buying rank (**Appendix C**). The main line therefore returns to where rank is placed:

### 3. Direction 4: The Oracle Decomposition — Which Blocks the Tax Lives In

[Part 3](/2026/08/22/closed-form-ceiling/)'s full oracle (swap all sublayer inputs with the teacher's clean values) only gives a total. Decompose it — swap one subset at a time (this table is a one-off diagnostic kept in its original measurement window; read the relative differences):

| Swapped | loss | vs baseline 5.13 |
|---|---|---|
| everything (full oracle) | 4.15 | +0.98 |
| q/k/v entrances only | 4.51 | +0.62 |
| gate/up entrances only | 4.57 | +0.56 |
| **blocks 18-35 only** | **4.14** | **+0.99 ≈ the full oracle!** |
| blocks 0-17 only | 5.53 | **−0.40 (worse)** |
| blocks 12-19 only (around block 16) | 5.49 | −0.36 (worse) |

Two heavyweight conclusions:

**First: the truncation tax lives almost entirely in the second half.** Cleaning only blocks 18-35's inputs reaches the full oracle — meaning the first half's local truncation errors contribute ~0.01 nat directly to the logits; all their harm travels the indirect "manufacture drift" channel, and drift is (partially) correctable, while the second half's tax is incurable. **Corollary: marginal rank is worth far more in late blocks.** Our earlier attempt was the exact opposite direction (front-loading, 5.38, fail — its rationale "reduce drift flowing into block 16" was badly wrong).

**Second: locally "cleaning" upstream hurts downstream (−0.40).** In a sequential pipeline every layer and corrector is fit to its upstream's specific error pattern; hand them clean intermediates and downstream compensations misfire en masse. This is the third independent proof of corrector-chain self-consistency (after fixed-point iteration 5.95, and the literature's cross-layer joint-optimization negative).

Hence back-loading (tilting rank toward the second half): early ranks ×0.8, late ranks budget-balanced up (×1.24). **5.05 — the final record**, with the dose-response optimal near ×0.8 (both untilted and ×0.65 are slightly worse). Honesty about the size: it beats damped allocation's 5.06 by only ~0.014, on the order of the fold-to-fold spread (±0.015) — the direction matches the oracle decomposition's prediction, but the gain sits close to measurement precision.

The last experiment stacked the two verified small improvements (back-load + rms-lift-256): **fail — worse than back-load alone.** Improvements at the 0.02 level harvest the same residue and erode each other when combined.

### 4. Conclusions: Three Bottlenecks, Three Ways Out, Three Lessons

**The final landscape** (85% compression, equal 2.29B budget):

$$8.50\_{\text{collapse illusion}} \to 5.60\_{\text{traj. correction}} \to 5.10\_{\text{tax-free correctors}} \to \mathbf{5.05}\_{\text{+alloc+backload}} \to 4.15\_{\text{oracle}} \to 3.2\_{\text{trained@500}} \to 2.11\_{\text{teacher}}$$

| Bottleneck | Architectural root | Way out |
|---|---|---|
| Truncation tax ~2.0 (concentrated late) | superposition storage on the content pathway, slowly decaying spectra | change where rank lives (cross-matrix joint decompositions, sparse residuals) — outside the current paradigm |
| Drift residue ~0.9 | SwiGLU product + RMSNorm×massive channels | nonlinear correctors work but at par parameter efficiency — exhausted |
| Reorganization gain ~0.9 | the residual sum's freedom to re-divide labor | global optimization (training) only |

Three transferable lessons distilled from these experiments:

1. **The oracle decomposition is a far stronger diagnostic than the oracle itself.** The full oracle tells you where the ceiling is; partial oracles tell you **where the error lives** — the single number "late ≈ all" overturned a weeks-held front-loading intuition and converted directly into a record.
2. **The corrector chain is a self-consistent whole.** In a sequentially-fit pipeline every component adapts to its upstream's specific errors. Any "local improvement" (fixed-point re-solving, cross-layer joint objectives, partial input cleaning) breaks downstream compensation — verified three independent ways. Cheat during diagnosis; refit the whole chain for deployment.
3. **Small improvements don't stack.** When separate levers (allocation, passthrough, nonlinear lift) harvest the same residue, 0.02-level gains erode each other on combination. Before adding two improvements, check whether their mechanisms attack different components of the gap — gains on the same component discount each other before they add.

The closed-form track is now genuinely closed: equal-budget ~5.05 is the terminus of the "layerwise regression + linear/light-nonlinear correctors + rank allocation" toolbox. The road to 2.5 runs through training (the training arm was at 2.87 and descending as this was published); the road below 4.15 for closed-form runs outside the paradigm.

Part 3 declared closure once and this post overturned it — how long will this post's own declaration stand? [Part 5](/2026/08/30/closed-form-moving-ceiling/) has the answer.




---

## Appendix

### Appendix A: Literature Coordinates — Where We Sit on the Frontier

A systematic literature sweep (training-free low-rank LLM compression, 2024-2026) calibrated expectations with two findings:

1. **No published training-free method reports usable results at 70%+ parameter removal.** The furthest pure closed-form data point is SVD-LLM at 60% removal on LLaMA-7B: PPL 53.7 ≈ loss 4.0 — startlingly consistent with our oracle bound. The literature independently confirms the wall's magnitude. At 72% removal and 5.0x, we sit ahead of the frontier.
2. **Non-uniform rank allocation is the literature's most consistent source of gains** (six independent works — Dobi-SVD, SVD-LLM v2, D-Rank, UniRank, AIR, LACE-SVD — all agree: at high compression, allocation matters more than the decomposition itself). This triggered Direction 1.

### Appendix B: Direction 2 — Massive-Activation Channels, a Double Negative and the True Amplifier

Measuring the teacher's output-side outliers revealed that massive activations (the phenomenon noted in [part 3](/2026/08/22/closed-form-ceiling/), Appendix A) are in this model **two global channels running the full depth**: dims 1838 and 2276, 60-110× the median channel, growing monotonically from block 0 to 33 (mean|x| 4.9 → 607); nearly every o_proj/down_proj's largest output row targets them.

Following part 3's tax-free-corrector principle we ran two equal-budget passthrough experiments: exempt the massive input COLUMNS from SVD truncation (top-24 per layer, kept exact), and exempt the massive output ROWS (top-8 of o/down). **Both washed out.** The conclusion is worth more than a win: the whitened SVD's objective is dominated by those very rows/columns, so the top singular directions **serve them first** — massive channels pay no truncation tax to begin with. A clean contrast with the lm_head/residual-corrector successes: those fixed whole matrices nobody was correcting; these are channels inside matrices that were already well served.

But the measurement exposed the precise mechanism of ② (drift residue): **RMSNorm's denominator is dominated by those two dims — when they drift, all 4096 channels get multiplicatively rescaled.** It is a coupler that globalizes local errors, and the true identity of the block-16 amplifier found in part 3 (one layer destroying residual R² by 0.54). The damage comes entirely from drift flowing at runtime, not from weight approximation — which is why every static weight-side arrangement (dense block, front-load, passthroughs; four failures) never touched it.

### Appendix C: Direction 3 — rms-Lift Nonlinear Correctors, Mechanism Confirmed, Gain Zero

If the amplifier is RMSNorm's multiplicative coupling, the targeted fix is a corrector feature space that can express scale corrections: lift the regression input from $h$ to $[h;\ h\cdot(\bar r/\mathrm{rms}(h)-1)]$ — the second block being the per-token **scale-anomaly component**. Still linear in parameters; the ridge solve stays closed-form.

Two lessons arrived before the result. The first shot used the naive form $[h;\ h/\mathrm{rms}(h)]$: the two blocks are highly collinear with a 100× scale mismatch, so ridge produced huge mutually-canceling weights — in-sample R² soared while held-out got worse (a classic overfitting signature). **Lift features must be centered and scale-matched.** After reconditioning (anomaly form + $\lambda=10^{-3}$):

| Configuration | val loss | Verdict |
|---|---|---|
| Linear correctors (baseline) | 5.06 | |
| rms-lift, full-rank ( **+8% params**) | **4.99** | first closed-form below 5.0; corrector R² after block 16 0.36→0.66 |
| rms-lift, truncated to rank 256 (equal budget) | 5.07 | wash |

**Mechanism confirmed, gain zero**: multiplicative drift is genuinely repairable (the R² gain and the sub-5.0 prove it), but the repair's parameter efficiency exactly matches just buying more rank — no extra bargain at equal budget. This closes the loop on the tax-free-corrector principle: lm_head and the linear residual correctors won by fixing drift nobody was managing; the lift's nonlinear part fixes residue the linear tools had already tended, so its marginal return is at par.


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
  var postTitles = {zh: '低秩压缩系列（四）：天花板的解剖——用 oracle 分解剩余差距', en: "Low-Rank Compression Series (4): Anatomy of the Ceiling — an Oracle Decomposition of the Remaining Gap"};
  var titleEl = document.querySelector('.post-title');
  if (titleEl) titleEl.textContent = postTitles[lang];
}
switchLang('zh');
</script>
