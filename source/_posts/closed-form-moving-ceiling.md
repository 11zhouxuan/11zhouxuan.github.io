---
title: "Low-Rank Compression Series (4): From 5.10 to 4.59 — Rank Allocation, Sparse Residuals, a Better Metric, and Calibration Data"
date: 2026-08-30
mathjax: true
sticky: 10
tags: [math, linear-algebra, LLM, compression, distillation, low-rank, rank-allocation, sparse, calibration]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 低秩压缩系列（四）：从 5.10 到 4.59——rank 分配、稀疏残差、更好的度量与校准数据

> 📖 如果你不熟悉语言模型的基本词汇（loss、残差流、SVD、蒸馏……），建议先读[预备知识篇](/2026/08/30/lord-compression-primer/)，10 分钟即可补齐全部背景。

[第三篇](/2026/08/22/closed-form-ceiling/)把闭式（零训练）压缩的纪录停在 5.10，并把剩余误差的来源锁定到了 block 16。之后我们又做了四类改进，把纪录推到 **4.59**（对照：端到端蒸馏训练的模型为 3.79），仍不涉及任何训练。本篇按发生顺序描述这四步，每节一个改动、一个结果。

背景（一句话）：教师 Qwen3-8B（val loss 2.11）→ 全部 252 个权重矩阵换成低秩因子 $AB$（2.29B 等预算，删 72% 参数），方法 = 第二篇的逐矩阵回归 + 第三篇的全秩矫正器，起点 5.10。

（本文 loss 均为严格协议重测值：800 段 × 8192 token、8 折，折间波动约 ±0.02。第 6 节的诊断实验保持当时的测量口径，只看相对差值。）

### 0. 先看终点：全部改动写在一个式子里

第二篇确立的逐层做法是：解出回归目标 $M^\*$ 后，在白化度量下截断到 rank $r$，即 $\min\_{A,B} \lVert (M^\* - AB) L \rVert\_F^2$。本篇终点的逐层目标变成：

$$\min\_{A, B, S}\ \Big\lVert \underbrace{\mathrm{diag}(w)}\_{\text{改动③：按 loss 敏感度加权}} \big(M^\* - AB - \underbrace{S}\_{\text{改动②：稀疏残差}}\big) L \Big\rVert\_F^2 \qquad \text{s.t.}\quad \mathrm{rank}(AB) \le \underbrace{r\_\ell}\_{\text{改动①：逐矩阵分配}}, \quad \mathrm{nnz}(S) \le k$$

四处改动，两处在式子里、两处在式子外：

- **改动①**：rank 不再是全网统一的常数，每个矩阵分到自己的 $r\_\ell$（第 1 节）；
- **改动②**：低秩项旁边加一个稀疏项 $S$（第 2 节）；
- **改动③**：误差不再一视同仁，按"这个输出分量错了对最终 loss 伤害多大"加权，$w\_j^2 = \mathbb{E}[(\partial \mathcal{L}/\partial y\_j)^2]$（第 3 节）；
- **改动④**在式子的原料里：所有统计量（$\Sigma\_{ss}, \Sigma\_{ts}, L, w$）改用大得多、杂得多的校准数据估计（第 4 节）——它的贡献不小于任何一处算法改动。

矫正器一侧还有最后一小步（**改动⑤**，第 5 节）：给矫正器的输入补一块尺度信息。

### 1. 第一步：给每个矩阵分不同的 rank —— 5.10 → 5.05

此前全网 252 个矩阵统一用 rank 384。但它们的"可压缩性"天差地别：q/k 只负责算注意力分布，本质低秩；v→o 和 up→down 要把内容写回残差流，谱衰减极慢（down\_proj 想保住 95% 的信号需要 rank 1839）。统一的 rank 显然不是最优分配，文献里也有六篇独立工作同向支持"高压缩下分配比分解本身更重要"（调研见附录 A）。

怎么知道每个方向值多少？直接问 loss：在每个保留的 rank-1 方向上挂一个乘性门 $\alpha\_i = 1$，反传时只求门的梯度，累积 $f\_{\ell,i} = \sum (\partial \mathcal{L}/\partial \alpha\_i)^2$——它衡量"砍掉第 $\ell$ 个矩阵的第 $i$ 个方向，最终 loss 会多痛"。然后解一个分配问题：

$$\min\_{\lbrace r\_\ell \rbrace}\ \sum\_\ell \underbrace{\sum\_{i > r\_\ell} f\_{\ell,i}}\_{\text{矩阵 }\ell\text{ 被砍方向的预期损害}} \qquad \text{s.t.}\quad \sum\_\ell r\_\ell (m\_\ell + n\_\ell) = \text{总预算}, \qquad \underbrace{r\_\ell \in [0.6\bar{r},\ 1.6\bar{r}]}\_{\text{幅度限制（不可省略）}}$$

第二个约束是必要的。敏感度 $f$ 给出的分配幅度很大（v/o 加倍、q/k 砍到 1/4），直接采用它反而变差（5.27）：$f$ 是在当前解附近测得的局部量，rank 砍到 1/4 已超出它的适用范围。把偏离限制在均匀值的 0.6~1.6 倍之内，得到 5.06。三档对照（不限制 5.27 / 限制 5.06 / 不分配 5.10）显示限制本身贡献了改进的一半。

在此之上还有一小步：一个诊断实验（细节见附录 B）显示，截断造成的损失几乎全部集中在模型的**后半段** block 里，于是把每层的 rank 再向后半段倾斜（前半 ×0.8、后半按预算配平放大）。结果 **5.05**。需要说明：这一步只比 5.06 低约 0.014，与折间波动同量级——方向与诊断实验一致，但幅度已接近测量精度。

### 2. 第二步：低秩旁边加一个稀疏项 —— 5.05 → 5.01

低秩形式有一个数学上的盲区：如果误差集中在**少数几个位置任意、又大又互不相关的矩阵元素**上，低秩因子表达不了这种结构（它擅长的是"整片方向"）。而写回矩阵恰好就是这样——它们以超位置方式存储特征（superposition：把远多于维度数的特征重叠地存在同一组维度里），误差呈重尾分布。补这种误差的天然形式是**稀疏项**：少数任意位置的大元素。文献里这也是证据最一致的方向（OATS、HASSLE-free、LoSparse：等预算下"低秩+稀疏"一致优于纯低秩，压缩率越高差距越大）。

做法是在每层的截断步里交替求解（3 轮）：

$$AB \leftarrow \text{whiten-SVD}(M^\* - S,\ r), \qquad S \leftarrow \text{top-}k\big(|M^\* - AB| \cdot \sigma\_{col}\big)$$

$AB$ 步就是现成的白化截断；$S$ 步给残差打分、取最大的 $k$ 个元素。预算严格配平：$S$ 的每个非零值按 1.5 个参数计入（含存储位置索引的开销），从 rank 里扣。

结果：$S$ 占总预算 21% 时最优，**5.05 → 5.01**（份额再加大反而回退）。文献推荐的"稀疏为主（70/30）"配比在这里不成立：他们的低秩部分是朴素 SVD，稀疏项承担了大量本可由低秩完成的逼近；我们的低秩部分是轨迹矫正回归，稀疏项只需要处理真正的重尾。配比这类超参数依附于基线强度，移植文献结论前需要先对齐基线。

### 3. 第三步：换度量 —— 单独 4.98，与稀疏项组合 4.88

"度量"指的是：你在最小化**哪一种**误差。白化截断最小化的是本层输出的 L2 误差，这在数学上是最优的（Eckart–Young 定理）——但"本层输出错多少"和"最终 loss 伤多少"不是一回事：q/k 的某些方向误差会被 softmax 和下游放大几十倍，down\_proj 的很多大能量方向却会被残差流稀释。此前的截断是在一个与最终目标不一致的度量下做到最优的。

修法仍是闭式的。设 $y\_j$ 为该层输出向量的第 $j$ 个分量，测 $w\_j^2 = \mathbb{E}[(\partial \mathcal{L}/\partial y\_j)^2]$（一次反传就能测完），截断目标改为

$$\min\_{AB}\ \lVert \mathrm{diag}(w)\ (M^\* - AB)\ L \rVert\_F^2$$

实现只是 SVD 前逐行乘上 $w$、解出后除回。这一步单独贡献 −0.07（4.98），不增加参数。两个实现细节是必要条件：$w$ 需要尺度归一并截到 [0.01, 100] 区间；$w$ 在一个"邻近"的模型上测一次即可（在新模型上迭代重测反而更差，见附录 D）。

组合效应超出相加：稀疏单独 −0.04、度量单独 −0.07，组合为 −0.17（**4.88**），比线性相加多 0.06。原因：换了度量后，稀疏项的打分也跟着升级（从"能量大的残差"变成"对 loss 关键的残差"），而度量加权的 SVD 留下的残差恰好更适合稀疏项接。可迁移的规律：**同一度量下的小改进互相挤压，换度量的改进会放大其他所有手段**——因为度量作用于每一个有 rank 约束的决策点。

（顺带一个干净的推论：对角加权对**全秩**回归没有影响——逐行独立求解时，每行乘个常数不改变那一行的解。所以残差矫正器和 lm\_head 矫正吃不到这份收益；度量的收益只存在于有 rank 约束的地方。这也回头解释了为什么当年 lm\_head 矫正不加任何权重就已经很强。）

### 4. 第四步：校准数据加量、加多样性 —— 4.88 → 4.61

闭式方法知道的一切都来自校准数据（所有 $\Sigma$、$L$、$w$ 都由它估计）。文献的标配是 128~256 条短文本，没有人把校准数据当成需要认真对待的工程对象。我们逐档改变它的用量和多样性：

| 统计配置 | val loss | 说明 |
|---|---|---|
| 32 批 × 1 个数据分片 | 4.88 | 组合配方的基线 |
| 128 × 1 | 4.73 | 用量 ×4：−0.15 |
| 128 × 4 分片 | 4.68 | **用量不变、来源 ×4**：再 −0.05 |
| **512 × 16** | **4.61** | 两个方向推到饱和 |

从 4.88 到 4.61：校准数据这一项贡献约 0.27 nat，超过本篇任何单个算法改动。两个定量结论：

1. **用量和多样性是两个独立的方向。** 加量压的是采样波动（32→128 批：−0.15，第一档收益最大）；量吃饱之后，换更多的数据来源仍有收益（−0.05）——同一个分片里的数据再多，它自带的分布偏色一分不动，只有换来源才能压掉。
2. 各自饱和：多样性在 8 个分片处饱和，用量到 512 批仍有小幅收益。饱和点大约出现在"统计波动降到单个改进的量级（±0.01）"处。

### 5. 收尾：给矫正器补上尺度信息 —— 4.61 → 4.59

最后一小步来自一个此前被搁置的想法。诊断发现（附录 C）：残差流里有两个激活值异常大的通道主导着 RMSNorm 的分母，它们一漂移，全部 4096 维跟着一起被乘性缩放——线性矫正器修不了这种"每个 token 缩放系数不同"的误差。对症的做法是给矫正器的输入补一块**尺度信息**：把回归输入从 $h$ 扩展为 $[h;\ h \cdot (\bar{r}/\mathrm{rms}(h) - 1)]$，第二块正是"这个 token 的幅度偏离了多少"。对参数仍是线性的，回归照样闭式可解。

这个做法在旧配方下与直接加 rank 打平、曾被判无收益；换上新度量和加固后的统计再试，它贡献了 −0.02：**4.61 → 4.59**，闭式方法的最终纪录。0.02 约等于一倍折间波动，属于边际改进；它的主要价值是再次验证"换度量会重新激活此前无效的手段"。

### 6. 离一个训练出来的模型还有多远

衡量剩余空间要靠一个允许作弊的诊断实验（**oracle 实验**）：运行时把每个子层的输入替换成教师的干净值，这样每层只剩自己的局部截断误差。这个作弊模型的 loss 是"逐层模仿教师"这条路线无论怎么矫正都碰不到的下限——任何矫正器最多把输入还原干净，不可能比干净输入更好。

给最终配方重测这个下限：**3.75**——已经**低于**一个端到端蒸馏训练出来的模型（3.79）。换句话说，闭式构造出的层已经好到"只要喂给它们干净的输入，就能追平真训练"；当前纪录 4.59 与训练之间的差距，大头已经不是层不够好（截断损失），而是**层间误差的传播**（漂移）——它锁在 SwiGLU 和 RMSNorm 的非线性后面，线性矫正器修不掉。要拿到这部分，得放弃"每层模仿教师对应层"的做法、直接优化最终 loss——那正是训练在做的事，也是这个系列下一阶段的方向。

（旧配方下这个下限是 4.15：稀疏+度量两步实打实把"层本身的损失"削掉了 0.4——改进是结构性的，不只是数字。）

### 7. 结论

**最终链条**（85% 线性层压缩、等预算 2.29B、零训练）：

$$8.50 \to 5.60 \to 5.10 \to \underbrace{5.05}\_{\text{①分配}} \to \underbrace{5.01}\_{\text{②稀疏}} \to \underbrace{4.88}\_{\text{③度量}} \to \underbrace{4.61}\_{\text{④校准}} \to \underbrace{\mathbf{4.59}}\_{\text{⑤尺度补偿}} \to 3.79\_{\text{端到端蒸馏}} \to 2.11\_{\text{教师}}$$

三条可迁移的经验：

1. **逼近的最优性总是相对某个度量而言。** 换成 loss 感知的度量后，此前测到饱和的每个手段都重新有了改进空间。分析任何压缩方法时，先确认它的截断在哪个度量下最优。
2. **校准数据是重要的自由度。** 用量和多样性独立贡献、各自饱和，逐档测量的成本很低；相对文献的默认配置，这里存在约 0.27 nat 的改进空间。
3. **在校准数据有限时，参数少的估计器表现更好。** 对角优于完整协方差、单遍优于迭代、限制幅度优于全量采用（附录 D）。校准数据的信息量是硬约束，估计器的额外自由度只能被噪声填充。

到此，配方内的每个维度（表达形式、度量、分配、矫正器、统计）都各自测到了饱和。2.29B 这一个预算点上的故事讲完了——方法离开这个点还站得住吗？[第五篇](/2026/09/04/lord-rank-sweep/)把 rank 从 384 一路砍到 24，交出整条曲线。

---

## 附录

### 附录 A：文献调研——我们在前沿的什么位置

动手前做了一轮系统调研（2024-2026 的 training-free 低秩压缩文献），两个结论校准了预期：

1. **没有任何已发表的 training-free 方法在 70%+ 参数削减下报告过可用结果。** 纯闭式的最远数据点是 SVD-LLM 在 LLaMA-7B 删 60% 时 PPL 53.7（折成 loss 约 4.0）。我们在删 72% 处拿到 4.59，在前沿之上。
2. **非均匀 rank 分配是文献中最一致的收益来源**（Dobi-SVD、SVD-LLM v2、D-Rank、UniRank、AIR、LACE-SVD 六篇独立工作同向）：高压缩下"分配比分解本身更重要"。这直接触发了第 1 节。

### 附录 B：诊断实验——截断损失住在哪些 block 里

第 6 节的 oracle 实验把全部子层输入换成教师的干净值，只给出一个总量。把它拆开——每次只换一部分输入（一次性诊断，旧测量口径，看相对差值）：

| 换干净什么 | loss | 相对基线 5.13 |
|---|---|---|
| 全部 | 4.15 | +0.98 |
| 只换 q/k/v 入口 | 4.51 | +0.62 |
| 只换 gate/up 入口 | 4.57 | +0.56 |
| **只换 block 18-35** | **4.14** | **+0.99，≈ 换全部！** |
| 只换 block 0-17 | 5.53 | **−0.40（反而变差）** |

两个结论。**其一：截断损失几乎全部住在后半段。** 只把后半段的输入换干净就达到了换全部的效果——前半段的截断误差对最终输出的直接伤害只有约 0.01 nat，它的全部危害走"制造漂移"这条间接通道，而漂移是（部分）可矫正的。第 1 节的"rank 向后半段倾斜"就是据此设计的。**其二：只把上游换干净反而伤害下游（−0.40）。** 顺序拟合的管线里，每一层和矫正器都是适配上游特定误差模式的；突然给它们干净的中间值，下游的补偿全部失配。诊断时可以作弊，部署时只能整链重新拟合。

### 附录 C：激活值异常大的通道与 RMSNorm 放大器

测量教师的激活分布，发现有**两个贯穿全网的通道**（维度 1838 和 2276）幅度是中位通道的 60~110 倍，且从 block 0 到 33 单调增长。两个针对它们的实验（让这些行列不参与截断、原样保留）均无改善——白化 SVD 的目标函数本来就被这些大行大列主导，最靠前的奇异方向优先拟合它们，因此它们几乎不受截断损失。

但测量顺带解释了漂移为什么难修：**RMSNorm 的分母被这两个通道主导——它们一漂移，全部 4096 维跟着乘性重缩放**。这是一个把局部小误差放大成全局误差的机制，也是第三篇发现的 block 16 放大器的本体。第 5 节的尺度补偿特征就是对着它设计的。一个实现教训：补偿特征的第一版用了 $[h;\ h/\mathrm{rms}(h)]$，两块特征高度相关且尺度差百倍，回归给出一对巨大且互相抵消的权重——校准集上拟合暴涨、新数据上反而变差（典型的过拟合信号）。**扩展特征必须中心化、尺度对齐**，改成"偏离量"形式 $h(\bar{r}/\mathrm{rms}(h)-1)$ 后才可用。

### 附录 D：四个未带来收益的复杂化尝试

与四步改进并行，我们还测试了四个更复杂的版本，均未带来收益：

| 精巧化尝试 | 结果 | 简单版 |
|---|---|---|
| 在新模型上重测 $w$ 再重建（迭代） | 4.89 | 测一次 4.88 |
| 完整协方差加权（64 维特征子空间 + 对角） | 4.74 | 纯对角 4.73 |
| 稀疏份额加大到 32% | 4.90 | 21% 的 4.88 |
| 按诊断结果定向的矩阵类型×深度分配 | 5.08 | 限制幅度的分配 5.05 |

另有一例：把两个各自有效的小改进（后倾分配 + 尺度补偿）直接叠加，效果比单独使用更差——0.02 量级的改进作用于同一块残余误差时互相抵消。判断两个改进能否叠加，要看它们处理的是否是误差的不同组成部分。

幅度说明：严格协议下这些回退多在 0.01~0.03，部分在折间波动之内，所以稳健的表述是"四个更复杂的版本没有一个带来可测的收益"。机制各不相同（迭代破坏矫正链的一致性、特征子空间对少量校准数据过拟合），指向同一个结论：校准数据的信息量有限时，参数少的估计器表现更好。

</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Low-Rank Compression Series (4): From 5.10 to 4.59 — Rank Allocation, Sparse Residuals, a Better Metric, and Calibration Data

> 📖 New to language-model vocabulary (loss, residual stream, SVD, distillation...)? Read [the primer](/2026/08/30/lord-compression-primer/) first — ten minutes covers all the background.

[Part 3](/2026/08/22/closed-form-ceiling/) left the closed-form (training-free) record at 5.10 and located the remaining error in block 16. Four further families of changes pushed the record to **4.59** (for reference, an end-to-end-distilled trained model sits at 3.79), still without any training. This post describes the four steps in the order they happened, one change and one result per section.

Background in one line: teacher Qwen3-8B (val loss 2.11) → all 252 weight matrices replaced by low-rank factors $AB$ (2.29B equal budget, 72% of parameters removed); method = part 2's per-matrix regression + part 3's full-rank correctors; starting point 5.10.

(All losses are re-measured under the rigorous protocol: 800 validation passages × 8192 tokens, 8 folds, fold-to-fold spread about ±0.02. The diagnostics in Section 6 keep their original measurement window; read relative differences only.)

### 0. The Endpoint First: Every Change in One Formula

Part 2's per-layer procedure: solve the regression target $M^\*$, then truncate to rank $r$ in the whitened metric, $\min\_{A,B} \lVert (M^\* - AB) L \rVert\_F^2$. This post's endpoint objective per layer:

$$\min\_{A, B, S}\ \Big\lVert \underbrace{\mathrm{diag}(w)}\_{\text{change ③: loss-sensitivity weighting}} \big(M^\* - AB - \underbrace{S}\_{\text{change ②: sparse residual}}\big) L \Big\rVert\_F^2 \qquad \text{s.t.}\quad \mathrm{rank}(AB) \le \underbrace{r\_\ell}\_{\text{change ①: per-matrix allocation}}, \quad \mathrm{nnz}(S) \le k$$

Four changes, two inside the formula and two outside:

- **Change ①**: rank is no longer one network-wide constant; each matrix gets its own $r\_\ell$ (Section 1);
- **Change ②**: a sparse term $S$ next to the low-rank one (Section 2);
- **Change ③**: errors are weighted by how much each output component matters to the final loss, $w\_j^2 = \mathbb{E}[(\partial \mathcal{L}/\partial y\_j)^2]$ (Section 3);
- **Change ④** lives in the formula's inputs: all statistics ($\Sigma\_{ss}, \Sigma\_{ts}, L, w$) are estimated from far more, and far more diverse, calibration data (Section 4) — its contribution is no smaller than any algorithmic change.

On the corrector side there is one final step (**change ⑤**, Section 5): a scale feature added to the correctors' input.

### 1. Step One: a Different Rank for Every Matrix — 5.10 → 5.05

Until now all 252 matrices used rank 384. Their compressibility differs widely: q/k only compute the attention pattern and are intrinsically low-rank; v→o and up→down write content back into the residual stream and have slowly decaying spectra (down\_proj needs rank 1839 to keep 95% of its signal). A uniform rank is clearly not the optimal allocation, and six independent papers agree that at high compression, allocation matters more than the decomposition itself (survey in Appendix A).

How much is each direction worth? Ask the loss directly: attach a multiplicative gate $\alpha\_i = 1$ to every kept rank-1 direction, backpropagate into the gates only, and accumulate $f\_{\ell,i} = \sum (\partial \mathcal{L}/\partial \alpha\_i)^2$ — a measure of how much the final loss suffers if direction $i$ of matrix $\ell$ is cut. Then solve an allocation problem:

$$\min\_{\lbrace r\_\ell \rbrace}\ \sum\_\ell \underbrace{\sum\_{i > r\_\ell} f\_{\ell,i}}\_{\text{expected damage of matrix }\ell\text{'s cut directions}} \qquad \text{s.t.}\quad \sum\_\ell r\_\ell (m\_\ell + n\_\ell) = \text{budget}, \qquad \underbrace{r\_\ell \in [0.6\bar{r},\ 1.6\bar{r}]}\_{\text{deviation limit (required)}}$$

The second constraint is required. The sensitivities $f$ prescribe large deviations (double v/o, cut q/k to a quarter), and adopting them at face value makes things worse (5.27): $f$ is a local quantity measured around the current solution, and a 4× cut is outside its range of validity. Limiting the deviation to 0.6–1.6× of uniform gives 5.06. A three-point comparison (unlimited 5.27 / limited 5.06 / no allocation 5.10) shows the limit itself contributes half of the improvement.

One further step: a diagnostic experiment (details in Appendix B) shows the truncation loss is concentrated almost entirely in the model's **second half**, so each layer's rank is additionally tilted toward late blocks (early ×0.8, late scaled up to balance the budget). Result: **5.05**. A caveat: this step beats 5.06 by only ~0.014, on the order of the fold-to-fold spread — the direction agrees with the diagnostic, but the size is close to measurement precision.

### 2. Step Two: a Sparse Term Next to the Low-Rank One — 5.05 → 5.01

Low-rank factors have a mathematical blind spot: they cannot express error that concentrates in **a few large, mutually unrelated entries at arbitrary positions** (their strength is whole directions). The write-back matrices are exactly like that — they store features in superposition (packing far more features than dimensions, overlapped in the same coordinates), and their error is heavy-tailed. The natural form for this kind of error is a **sparse term**: a few large entries at arbitrary positions. It is also the literature's most consistently supported direction (OATS, HASSLE-free, LoSparse: low-rank + sparse beats pure low-rank at equal budget, with the gap widening at higher compression).

The method alternates inside each layer's truncation step (3 rounds):

$$AB \leftarrow \text{whiten-SVD}(M^\* - S,\ r), \qquad S \leftarrow \text{top-}k\big(|M^\* - AB| \cdot \sigma\_{col}\big)$$

The $AB$ step is the existing whitened truncation; the $S$ step scores the residual and keeps its largest $k$ entries. The budget is strictly balanced: every nonzero of $S$ is charged 1.5 parameters (including index storage) and paid out of rank.

Result: the optimum puts 21% of the budget in $S$, **5.05 → 5.01** (larger shares regress). The literature's sparse-heavy (70/30) ratio does not hold here: their low-rank part is plain SVD, so the sparse term absorbs approximation work the low-rank part could not do; our low-rank part is trajectory-correcting regression, so the sparse term only needs the genuine heavy tails. Ratio-type hyperparameters are attached to baseline strength; align baselines before transferring conclusions.

### 3. Step Three: Change the Metric — 4.98 Alone, 4.88 Combined

The "metric" is the question of **which** error you minimize. Whitened truncation minimizes the layer output's L2 error, optimally so (the Eckart–Young theorem) — but "how wrong this layer's output is" and "how much the final loss suffers" are different quantities: some q/k error directions get amplified tens of times by softmax and downstream computation, while many high-energy down\_proj directions get diluted in the residual stream. The truncation so far had been optimal under a metric misaligned with the final objective.

The fix stays closed-form. Let $y\_j$ be the $j$-th component of the layer's output and measure $w\_j^2 = \mathbb{E}[(\partial \mathcal{L}/\partial y\_j)^2]$ (one backward pass), then truncate under

$$\min\_{AB}\ \lVert \mathrm{diag}(w)\ (M^\* - AB)\ L \rVert\_F^2$$

Implementation: multiply rows by $w$ before the SVD, divide after. This step alone contributes −0.07 (4.98) at no parameter cost. Two implementation details are necessary conditions: $w$ must be scale-normalized and clamped to [0.01, 100]; and one measurement on a "nearby" model suffices (re-measuring iteratively on the new model performs worse — Appendix D).

The combination exceeds the sum: sparse alone −0.04, metric alone −0.07, combined −0.17 (**4.88**), 0.06 more than additive. The reason: the metric also upgrades $S$'s selection (its score shifts from energy-heavy residuals to loss-critical ones), and the metric-weighted SVD leaves behind exactly the kind of residue the sparse term handles well. The transferable rule: **improvements under the same metric crowd each other out; a metric change amplifies every other lever**, because the metric acts at every rank-constrained decision point.

(A clean corollary: diagonal weighting cannot affect a **full-rank** regression — rows solve independently, and scaling a row does not change that row's solution. Hence the residual correctors and the lm\_head fix cannot collect this gain; the metric's gain exists only where a rank constraint does. This also explains retroactively why the lm\_head fix was strong without any weighting.)

### 4. Step Four: More and More Diverse Calibration Data — 4.88 → 4.61

Everything a closed-form method knows comes from calibration data (every $\Sigma$, $L$, $w$ is estimated from it). The literature's standard is 128–256 short texts; calibration data is generally not treated as a quantity worth engineering. We varied its amount and diversity in steps:

| Statistics | val loss | Note |
|---|---|---|
| 32 batches × 1 data shard | 4.88 | combined-recipe baseline |
| 128 × 1 | 4.73 | amount ×4: −0.15 |
| 128 × 4 shards | 4.68 | **same amount, sources ×4**: another −0.05 |
| **512 × 16** | **4.61** | both directions at saturation |

From 4.88 to 4.61: calibration data contributes about 0.27 nat, more than any single algorithmic change in this post. Two quantitative findings:

1. **Amount and diversity are independent directions.** More data reduces sampling variance (32 → 128 batches: −0.15, the first installment is the largest). Once the amount saturates, adding data sources still pays (−0.05) — more data from the same shard leaves its distributional tint unchanged; only changing sources removes it.
2. Each saturates on its own: diversity at 8 shards, amount still paying mildly at 512 batches. Saturation arrives roughly where statistical fluctuation drops to the size of a single improvement (±0.01).

### 5. The Finish: Scale Information for the Correctors — 4.61 → 4.59

The last step revisits a previously shelved idea. A diagnostic (Appendix C) shows that two abnormally large channels in the residual stream dominate RMSNorm's denominator; when they drift, all 4096 dimensions get rescaled multiplicatively — an error with a per-token scale factor, which a linear corrector cannot express. The targeted fix adds **scale information** to the correctors' input: extend the regression input from $h$ to $[h;\ h \cdot (\bar{r}/\mathrm{rms}(h) - 1)]$, the second block being how far this token's magnitude deviates. Still linear in parameters; the regression stays closed-form.

Under the old recipe this tied with simply buying more rank and was shelved. Retried with the new metric and the reinforced statistics, it contributes −0.02: **4.61 → 4.59**, the closed-form final record. 0.02 is about one fold-to-fold spread — a marginal improvement whose main value is confirming, once more, that a metric change re-activates previously ineffective levers.

### 6. How Far From an Actually Trained Model

The remaining room is measured with a diagnostic that is allowed to cheat (an **oracle experiment**): at runtime, replace every sublayer's input with the teacher's clean value, so each layer contributes only its own local truncation error. This cheating model's loss is a lower limit that the "imitate the teacher layer by layer" route cannot beat with any corrector — no corrector can do better than restoring inputs to clean.

Re-measured for the final recipe, the limit is **3.75** — below an end-to-end-distilled trained model (3.79). In other words, the closed-form layers are now good enough that with clean inputs they would match actual training. The gap between the current record (4.59) and training is therefore no longer dominated by layer quality (truncation loss) but by **error propagation between layers** (drift), which sits behind the SwiGLU and RMSNorm nonlinearities where linear correctors cannot reach. Collecting that part requires abandoning per-layer imitation and optimizing the final loss directly — which is what training does, and where this series goes next.

(Under the old recipe this limit was 4.15: the sparse and metric steps removed 0.4 nat from the layers' own loss — a structural improvement, not just a number.)

### 7. Conclusions

**The final chain** (85% linear-layer compression, equal 2.29B budget, no training):

$$8.50 \to 5.60 \to 5.10 \to \underbrace{5.05}\_{\text{①allocation}} \to \underbrace{5.01}\_{\text{②sparse}} \to \underbrace{4.88}\_{\text{③metric}} \to \underbrace{4.61}\_{\text{④calibration}} \to \underbrace{\mathbf{4.59}}\_{\text{⑤scale feature}} \to 3.79\_{\text{e2e distilled}} \to 2.11\_{\text{teacher}}$$

Three transferable findings:

1. **Approximation optimality is always relative to a metric.** After switching to a loss-aware metric, every previously saturated lever had room again. When analyzing any compression method, first determine under which metric its truncation is optimal.
2. **Calibration data is an important degree of freedom.** Amount and diversity contribute independently and saturate independently; measuring both in steps is cheap. Relative to the literature's default setup, about 0.27 nat of improvement was available here.
3. **With limited calibration data, estimators with fewer parameters perform better.** Diagonal beats full covariance, single-pass beats iterated, limited deviation beats face-value adoption (Appendix D). The information in the calibration set is the hard constraint; an estimator's extra degrees of freedom get filled by noise.

At this point every dimension of the recipe (representation, metric, allocation, correctors, statistics) has been measured to its own saturation. This concludes the story at the single 2.29B budget point — does the method hold away from it? [Part 5](/2026/09/04/lord-rank-sweep/) cuts rank from 384 down to 24 and reports the full curve.

---

## Appendix

### Appendix A: Literature Survey — Where We Sit on the Frontier

A systematic sweep of the training-free low-rank compression literature (2024–2026) calibrated expectations with two findings:

1. **No published training-free method reports usable results at 70%+ parameter removal.** The furthest pure closed-form data point is SVD-LLM at 60% removal on LLaMA-7B: PPL 53.7, about loss 4.0. At 72% removal we measure 4.59, ahead of that frontier.
2. **Non-uniform rank allocation is the literature's most consistent source of gains** (six independent works — Dobi-SVD, SVD-LLM v2, D-Rank, UniRank, AIR, LACE-SVD — agree that at high compression, allocation matters more than the decomposition itself). This motivated Section 1.

### Appendix B: A Diagnostic — Which Blocks the Truncation Loss Lives In

Section 6's oracle experiment swaps all sublayer inputs for the teacher's clean values and yields one total. Decomposing it — swapping one subset at a time (one-off diagnostic, original measurement window, read relative differences):

| Cleaned | loss | vs baseline 5.13 |
|---|---|---|
| everything | 4.15 | +0.98 |
| q/k/v entrances only | 4.51 | +0.62 |
| gate/up entrances only | 4.57 | +0.56 |
| **blocks 18-35 only** | **4.14** | **+0.99, ≈ everything** |
| blocks 0-17 only | 5.53 | **−0.40 (worse)** |

Two conclusions. **First, the truncation loss lives almost entirely in the second half.** Cleaning only the late blocks' inputs matches cleaning everything — the early blocks' truncation errors contribute about 0.01 nat directly to the output; all their harm travels the indirect route of manufacturing drift, and drift is (partially) correctable. Section 1's late-block rank tilt is based on this. **Second, cleaning only the upstream hurts (−0.40).** In a sequentially fitted pipeline, every layer and corrector adapts to its upstream's specific error pattern; handing them clean intermediates invalidates the downstream compensation. Cheating is fine for diagnosis; deployment requires refitting the whole chain.

### Appendix C: Abnormally Large Channels and the RMSNorm Amplifier

Measuring the teacher's activation distribution reveals **two channels running the full depth** (dimensions 1838 and 2276) at 60–110× the median channel's magnitude, growing monotonically from block 0 to 33. Two experiments that gave these rows and columns special treatment (exempting them from truncation, kept exact) produced no improvement — the whitened SVD's objective is already dominated by these large rows and columns, so the leading singular directions fit them first; they suffer almost no truncation loss to begin with.

The same measurement explains why drift is hard to repair: **RMSNorm's denominator is dominated by these two channels — when they drift, all 4096 dimensions get multiplicatively rescaled.** This mechanism turns small local errors into global ones, and is the substance of the block-16 amplifier found in part 3. Section 5's scale feature is designed against it. One implementation lesson: the first version used $[h;\ h/\mathrm{rms}(h)]$, whose two blocks are highly correlated with a 100× scale mismatch — the regression produced a pair of huge mutually canceling weights, fitting the calibration set better while degrading on new data (an overfitting signature). Extension features must be centered and scale-matched; the deviation form $h(\bar{r}/\mathrm{rms}(h)-1)$ fixed it.

### Appendix D: Four Refinements That Did Not Pay

Alongside the four improvements we tested four more complex versions, none of which paid:

| Refinement | Result | Simple version |
|---|---|---|
| Re-measure $w$ on the new model and rebuild (iterate) | 4.89 | one measurement: 4.88 |
| Full-covariance weighting (64-dim eigen-subspace + diagonal) | 4.74 | diagonal only: 4.73 |
| Sparse share raised to 32% | 4.90 | 21% at 4.88 |
| Allocation targeted by the diagnostic's family×depth map | 5.08 | deviation-limited allocation: 5.05 |

One more case: stacking two individually effective small improvements (late-block tilt + scale feature) performed worse than either alone — improvements at the 0.02 level cancel when they act on the same residue. Whether two improvements stack depends on whether they address different components of the error.

Sizes, for the record: under the rigorous protocol these regressions are mostly 0.01–0.03, some within fold noise, so the robust statement is that none of the four more complex versions delivered measurable gains. The mechanisms differ (iteration breaks the corrector chain's consistency; the eigen-subspace overfits the small calibration set), but they point the same way: with limited calibration data, estimators with fewer parameters perform better.

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
  var postTitles = {zh: '低秩压缩系列（四）：从 5.10 到 4.59——rank 分配、稀疏残差、更好的度量与校准数据', en: 'Low-Rank Compression Series (4): From 5.10 to 4.59 — Rank Allocation, Sparse Residuals, a Better Metric, and Calibration Data'};
  var titleEl = document.querySelector('.post-title');
  if (titleEl) titleEl.textContent = postTitles[lang];
}
switchLang('zh');
</script>
