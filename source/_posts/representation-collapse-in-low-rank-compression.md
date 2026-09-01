---
title: "Representation Collapse in Low-Rank Compression: When Low Loss Means a Broken Model"
date: 2026-08-17
mathjax: true
tags: [math, linear-algebra, LLM, compression, representation-collapse, SwiGLU, MLP]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 低秩压缩中的表示坍缩：为什么 val loss 好不等于模型好

> 📖 如果你不熟悉语言模型的基本词汇（loss、残差流、SVD、蒸馏……），建议先读[预备知识篇](/2026/08/30/lord-compression-primer/)，10 分钟即可补齐全部背景。

### 1. 一个意外的发现

实验对象是 Qwen3-8B（36 层，作为教师模型，val loss 2.11）。我们把它的每个线性层 $W$ 都替换成两个瘦矩阵的乘积 $AB$（低秩分解），把总参数量压到 2.29B，然后比较两种截断方式：

- **plain SVD**：直接对 $W$ 做 SVD 截断，每层统一保留 rank 384。val loss = 18.65。
- **ASVD（激活加权 SVD）**：截断前先按每个输入通道的平均激活幅度加权——直觉是"常被用到的通道要逼近得更准"，加权强度由指数 $\alpha$ 控制（$\alpha=1.0$ 即按幅度原样加权）。再配合**逐层 rank 分配**：重要性（$\lVert WS\rVert\_F$）高的层多给 rank、低的少给，平均 rank 357，分得最少的 Block 0 只有 32。val loss = **8.50**。

8.50 对 18.65，ASVD 看起来好了一倍多。但当我们检查模型实际预测的 token 时，发现了一个惊人的事实：

| 方法 | val loss | 预测的 unique token 数 | top-1 预测占比 | top-1 是什么 |
|---|---|---|---|---|
| 教师（满秩） | 2.11 | 1933 | 8.2% | " the" |
| Plain SVD | 18.65 | 433 | 15.9% | " :\r\n" |
| Random 权重 | 17.77 | 1104 | 13.7% | "fcc" |
| **ASVD** | **8.50** | **1** | **100%** | **","** |

**ASVD 模型对所有 8191 个位置都预测同一个 token——逗号。**（单序列测量；在 30 个 batch 上复测为 12 个 unique token、逗号占 98.6%，现象相同。）

### 2. 为什么坍缩反而 loss 更低？

这看似矛盾：一个"更坏"的模型（只能输出一个 token）怎么比"更好"的模型（能输出 433 种 token）loss 更低？

原因在交叉熵 loss 的定义里：它只看模型给**正确答案**分了多少概率——每个位置计 $-\log q(\text{正确 token})$，再在所有位置上取平均。

当模型完全没有区分能力时，最优策略不是乱猜，而是**每个位置都输出同一个固定的概率分布**，把概率按各 token 在语料中的真实频率来分。这样虽然每个位置都"不知道答案"，但至少高频 token（比如逗号，占英文文本约 3.6% 的位置）稳定拿到不低的概率，平均惩罚最小。坍缩模型正是这么做的：它在每个位置都给逗号约 10% 的概率（实测 top-1 prob = 0.0995），得到 CE ≈ 8.5。

反过来，一个**试图区分 token 但区分得很差**的模型（random 权重、plain SVD）会把概率押在错误的 token 上，正确答案分到的概率极小，$-\log$ 惩罚巨大。参照系：把概率在全部 151936 个词表 token 上平均分配，CE = $\ln 151936 \approx 11.93$；押错注比平均分配更糟，所以 random 和 plain SVD 冲到了 17~19。

**"全押高频 token"是预测能力为零时的 loss-最优退化策略。** ASVD 的 8.50 不代表"逼近得好"，而是代表"模型已经放弃预测，退化成了常数函数"。

信息论上可以精确验证这一点。如果模型是常数预测器（每个位置输出同一个分布 $q$），它的 CE 就等于语料的 unigram 分布 $p$（即各 token 的出现频率分布）与 $q$ 之间的交叉熵，其下界是 $p$ 自己的熵：

$$\min\_q H(p, q) = H(p) \approx 7.51 \text{ nats（在我们的 val 数据上实测）}$$

坍缩模型的 8.50 恰好落在这个下界之上 1 nat——它是一个接近最优的常数预测器。直接测量也证实了"常数函数"：采样 64 个位置，输出分布之间的两两 KL 散度（衡量两个概率分布差异的量，完全相同时为 0）平均只有 **0.007**，而正常模型在 1~10 量级。

### 3. 坍缩的机制

**为什么 ASVD 导致坍缩而 plain SVD 不会？**

ASVD 对每层用 $S = \text{diag}(\mathbb{E}[\lvert x\_i\rvert])$ 加权——$S$ 是个对角矩阵，第 $i$ 个对角元就是第 $i$ 个输入通道激活幅度的平均值。关键观察：**相邻层的 $S$ 高度相似**（残差连接让激活分布逐层变化缓慢）。这意味着：

- 每一层的 SVD 截断都优先保留 $S$ 大的方向（高激活通道）
- 每一层都丢掉 $S$ 小的方向
- 36 层叠加后，$S$ 小的方向被**反复压制 36 次**
- 最终 hidden state 只剩 $S$ 大的那几个方向 → **所有 token 的表示坍缩到同一方向**

Plain SVD 不做加权，保留的方向由各层矩阵自身的奇异值决定——不同层的主方向不同，所以不会系统性地压制同一组方向。

实验验证了这一点。逐层测量 hidden state 与教师的 cosine 相似度：

| 层 | ASVD cos | 解释 |
|---|---|---|
| 0（embedding 后） | 1.000 | 输入完美 |
| 1（Block 0 后） | 0.718 | 第一层就严重偏离 |
| 10 | 0.496 | 触底 |
| 11~24 | 0.49~0.53 | 在 0.5 附近震荡（**平台**） |
| 35（最后） | 0.830 | 恢复 |
| 36（final norm 后） | 0.950 | 看似"对齐" |

最后一层 cos=0.95 看起来"对齐得很好"——但这是一个**假象**。不是模型对每个 token 都和教师对齐，而是**所有 token 都坍缩到了同一个方向**，这个固定方向恰好和教师的某个"平均方向" cos=0.95。

### 3.5 坍缩的微观机制：MLP 是凶手

进一步的诊断实验追踪了 **effective rank（有效秩，下称 erank）** 在网络每个子组件中的变化。erank 衡量一批向量实际张开了多少个独立方向：hidden state 名义上是 4096 维，但如果所有 token 的向量都挤在少数几个方向上，erank 就只有个位数。它是"表示多样性"的直接读数。

> **测量对象说明**：本节的追踪数据来自均匀 rank=384、$\alpha=0.5$ 的 ASVD 变体（val loss 10.83）。它没有完全坍缩成常数函数（保留了 546 个 unique 预测），但表示已高度退化：最终 erank=7，不同 token 输出向量的两两平均 cosine 相似度（下称 pcos，越接近 1 表示所有 token 的输出越趋同）高达 0.92——是观察"多样性如何被逐层耗尽"的合适对象。完全坍缩的 8.50 模型（$\alpha=1.0$ + 逐层 rank）是同一机制的更极端版本。

**Effective rank（erank）在 residual stream 中的变化：**

| Block | Teacher | ASVD | Plain SVD |
|---|---|---|---|
| 0（embedding 后） | 264 | 264 | 264 |
| 3 | 353 | 233 | 4.5 |
| 4 | 355 | 99 | 10 |
| 7~16 | 65~120 | 124~152 | 13~14 |
| 35 | 301 | 57 | 25 |
| 36（final norm 后） | 156 | **7** | **111** |

惊人的发现：**Plain SVD 在中间层比 ASVD 坍缩得更严重**（Block 3 erank=4.5 vs ASVD 的 233），但最终 plain SVD 恢复到 erank=111 而 ASVD 坍缩到 7。

**SwiGLU MLP 是逐层坍缩的主要驱动力。** 逐 block 分解显示：

| Block | Attention 贡献 | MLP 贡献 | 净变化 |
|---|---|---|---|
| 0 | +19.5 | -14.5 | +5.0 |
| 2 | +7.5 | -31.1 | -23.5 |
| **3** | +7.3 | **-141.8** | **-134.5** |
| 4 | +5.2 | -23.4 | -18.3 |

Block 3 的 MLP 一次性将 erank 从 233 砍到 ~99——这是整个网络中最致命的一击。

**MLP 内部的追踪**（以 Block 3 为例）：

| 子步骤 | Teacher erank | ASVD erank | 说明 |
|---|---|---|---|
| MLP 输入 | 245 | 163 | |
| gate\_proj | 53 | 20 | gate 投影本身极低秩 |
| SiLU(gate) | 334 | 165 | 非线性恢复多样性 |
| up\_proj | 315 | 148 | |
| gate×up | 352 | **71** | 乘法二次化误差 |
| MLP 输出 | 329 | **28** | down\_proj 再次压缩 |

**为什么 ASVD 不能恢复而 plain SVD 可以？**

关键不是 MLP 输出的幅度，而是**MLP 输出对不同 token 是否不同**：

| 指标（36 层平均） | Teacher | ASVD | Plain SVD |
|---|---|---|---|
| MLP 输出 pcos | 0.20 | **0.73** | 0.27 |
| MLP override ratio | 0.40 | 0.30 | 1.40 |

（override ratio = MLP 输出与残差流本身的幅度之比，衡量 MLP 对残差流的改写力度。）

**ASVD 的 MLP 对所有 token 输出几乎相同的向量**（pcos=0.73），而 plain SVD 的 MLP 输出对不同 token 是不同的（pcos=0.27）。

这意味着在残差加法 $h\_{new} = h\_{old} + \text{MLP}(h\_{old})$ 中，ASVD 给每个 token 加了近乎相同的偏移，36 层累积后所有表示收敛到同一方向。而 plain SVD 虽然 MLP 幅度更大（override ratio=1.40），但每个 token 的偏移不同，所以不会收敛。

**根本原因**：down\_proj 将 12288 维的 gate×up 结果映射回 4096 维时，rank=384 的 ASVD down\_proj 只保留了 384 个线性组合——这些组合恰好是所有 token **共享的成分**（因为 ASVD 的加权偏好保留高激活通道），而 **token-specific 的差异成分被丢弃**。

### 4. 文献中的对应

这个现象在 transformer 研究中已有理论解释：

- **Signal Propagation in Transformers (2022)**：堆叠 self-attention 层导致 token 表示的 rank collapse
- **Attention Masks and LayerNorm (2024)**："With proper choice of value matrices, sequences may not converge to rank one subspace" —— 暗示 value 矩阵选择不当会导致坍缩
- **BA-LoRA (2024)**：在 LoRA 训练中识别出 representation collapse，解决方案是 diversity 正则化

我们的情况属于"value 矩阵选择不当"的极端版本——低秩截断让所有层的投影方向系统性地对齐，满足了坍缩的条件。

### 5. 打破坍缩的尝试

既然坍缩来自"36 层反复保留同一组方向"，一个自然的对策是强迫相邻层保留的方向互相错开：在每层截断时加一个正交惩罚项（表中的 0.3 是惩罚权重），或者干脆让 ASVD 和 plain SVD 逐层交替：

| 方法 | val loss | unique 预测 | top-1 占比 |
|---|---|---|---|
| ASVD（无约束） | 8.50 | 1 | 100% |
| 正交约束 0.3 | 12.80 | 23 | 93.6% |
| 交替 ASVD/plain SVD | 14.56 | 41 | 35.2% |
| Plain SVD（无加权） | 18.65 | 401 | 15.9% |

**正交约束确实打破了坍缩**（预测多样性从 1 到 23 到 401），但 **val loss 同时变差**。

进一步的 rank 重分配实验（在均匀 rank=384、$\alpha=0.5$ 变体上做，其自身基线为 10.83）显示，即使给关键组件多得多的参数，也无法接近坍缩态的 loss：

| 配置 | 参数量 | val loss | 是否坍缩 |
|---|---|---|---|
| 坍缩的 ASVD（$\alpha=1.0$ + 逐层 rank） | 2.29B | **8.50** | 是（98.6% 逗号） |
| 均匀 rank=384，$\alpha=0.5$（重分配实验的基线） | 2.29B | 10.83 | 否 |
| + gate\_proj 满秩 | 3.88B | 14.22 | 否 |
| + down\_proj 满秩 | 3.88B | 11.32 | 否 |
| + down\_proj rank=1024 | 2.67B | 11.33 | 否 |
| + down\_proj rank=768（预算匹配） | 2.29B | 12.31 | 否 |

**没有任何非坍缩配置能接近坍缩态的 8.50**——即使用了 70% 更多的参数（3.88B vs 2.29B）。这揭示了一个根本的 tradeoff：在这一族配置里，**坍缩得越彻底，loss 越低；打破坍缩，loss 就变差**。

原因回到第 2 节：在模型**真的没有足够能力区分 token** 的情况下（rank=384、85% 压缩率），坍缩到高频 token 就是 loss-最优的策略，打破坍缩等于强迫模型做它做不到的事。

### 5.5 验证：坍缩是真实的，但对配置敏感

由于"整个模型只预测逗号"的结论过于反直觉，我们对它做了独立的复核：重新评估保存的 checkpoint、在 held-out 数据上复测、检查输出分布的位置间 KL、并与信息论下界对照。结论：**坍缩完全可复现**（val loss 8.5008、逗号 98.6%、位置间 KL=0.007、top-1 prob 0.095 ≈ 常数预测器）。

但复核也暴露了一个重要事实：**坍缩对分解配置高度敏感**——

| 配置 | val loss | 是否坍缩 |
|---|---|---|
| diag，$\alpha=1.0$，逐层 rank（Block 0 = 32） | **8.50** | **是**（KL=0.007） |
| diag，$\alpha=0.5$，逐层 rank | 9.66 | 否（389 unique，KL=3.1） |
| diag，$\alpha=0.5$，均匀 rank=384 | 10.83 | 否（546 unique，KL=0.67） |

只有"更强的激活加权（$\alpha=1.0$）+ 被饿死的 Block 0（rank 32）"这个组合触发了完全坍缩。把 $\alpha$ 减半或改用均匀 rank，模型就停在"高度退化但未坍缩"的状态——而 loss 反而更差。这与第 5 节的 tradeoff 完全一致：**在这组变体里，坍缩得越彻底，loss 越低**（8.50 < 9.66 < 10.83）。

### 6. 结论

1. **val loss 不是模型质量的可靠指标**——一个完全坍缩的模型（只输出逗号）可以比一个"有预测多样性但不准"的模型 loss 更低。

2. **ASVD 的激活加权在极端压缩率下会导致 representation collapse**——加权越强（$\alpha$ 越大）、关键层被 rank 分配饿得越狠，坍缩越彻底。

3. **这个 collapse 是 loss-optimal 的退化**——在当前的参数预算下，模型确实没有能力做好 token 区分，坍缩到高频 token 是理性的"放弃策略"。定量上：坍缩模型的 8.50 距离常数预测器的信息论下界 H(unigram)=7.51 只有 1 nat，且打败了所有实测的非坍缩变体（9.66~14.2）。

4. ~~要恢复真正的预测能力，必须经过端到端训练——闭式方法无法同时打破坍缩又降低 loss。~~ **【后续更正】这个结论被推翻了**：错的不是闭式本身，而是当时所有方法共享的"逐层逼近 $W$"目标。把逐层目标换成"从学生的漂移输入回归教师轨迹的干净输出"，闭式方法可以同时打破坍缩并把 loss 降到 **5.59**（低于常数预测器下界 7.51，携带真实互信息）。详见[《轨迹矫正线性蒸馏》](/2026/08/19/trajectory-correcting-linear-distillation/)。

5. **在评估压缩模型时，除了 val loss 还必须检查预测多样性**——否则可能被"坍缩到高频 token"的假象误导。同样，**对比不同压缩方法时必须固定全部配置**（加权强度 $\alpha$、rank 分配方式）——我们自己就曾把两个不同配置的模型（8.50 与 10.83）误当作同一个基线。

</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Representation Collapse in Low-Rank Compression: When Low Loss Means a Broken Model

> 📖 New to language-model vocabulary (loss, residual stream, SVD, distillation...)? Read [the primer](/2026/08/30/lord-compression-primer/) first — ten minutes covers all the background.

### 1. An Unexpected Finding

The subject is Qwen3-8B (36 layers, our teacher model, val loss 2.11). We replace every linear layer $W$ with a product of two thin matrices $AB$ (low-rank factorization), squeezing the total parameter count to 2.29B, and compare two truncation schemes:

- **Plain SVD**: truncate the SVD of $W$ directly, keeping a uniform rank 384 in every layer. Val loss = 18.65.
- **ASVD (activation-weighted SVD)**: before truncating, weight each input channel by its average activation magnitude — the intuition being "channels that get used a lot deserve a more accurate approximation." The weighting strength is an exponent $\alpha$ ($\alpha=1.0$ means weighting by the raw magnitudes). Combined with **per-layer rank allocation**: layers with higher importance ($\lVert WS\rVert\_F$) get more rank, others less — mean rank 357, with Block 0 starved at rank 32. Val loss = **8.50**.

8.50 versus 18.65 — ASVD appears more than twice as good. But when we inspect the model's actual token predictions:

| Method | val loss | Unique tokens predicted | Top-1 fraction | Top-1 token |
|---|---|---|---|---|
| Teacher (full rank) | 2.11 | 1933 | 8.2% | " the" |
| Plain SVD | 18.65 | 433 | 15.9% | " :\r\n" |
| Random weights | 17.77 | 1104 | 13.7% | "fcc" |
| **ASVD** | **8.50** | **1** | **100%** | **","** |

**The ASVD model predicts the same token — a comma — for all 8191 positions.** (Single-sequence measurement; re-measured over 30 batches: 12 unique tokens, comma at 98.6% — the same phenomenon.)

### 2. Why Does Collapse Produce Lower Loss?

This seems paradoxical: how can a "worse" model (only one token) have lower loss than a "better" one (433 distinct tokens)?

The answer is in the definition of cross-entropy loss: it only measures how much probability the model assigns to the **correct answer** — each position scores $-\log q(\text{correct token})$, averaged over all positions.

When a model has no ability to distinguish tokens, the optimal strategy is not to guess wildly, but to **output the same fixed probability distribution at every position**, allocating probability according to each token's true frequency in the corpus. Every position is still "wrong," but at least the frequent tokens (like the comma, which fills about 3.6% of positions in English text) reliably receive decent probability, minimizing the average penalty. That is exactly what the collapsed model does: it gives the comma about 10% probability at every position (measured top-1 prob = 0.0995), yielding CE ≈ 8.5.

Conversely, a model that **tries to distinguish tokens but fails** (random weights, plain SVD) bets its probability on wrong tokens — the correct answer gets a tiny share, and the $-\log$ penalty explodes. For reference: spreading probability uniformly over all 151936 vocabulary tokens gives CE = $\ln 151936 \approx 11.93$; betting wrong is worse than spreading uniformly, which is how random and plain SVD reach 17–19.

**"All-in on frequent tokens" is the loss-optimal degenerate strategy when predictive ability is zero.** ASVD's 8.50 does not mean "good approximation" — it means "the model has given up predicting and collapsed to a constant function."

This can be verified information-theoretically. If the model is a constant predictor (outputting the same distribution $q$ at every position), its CE equals the cross-entropy between $q$ and the corpus unigram distribution $p$ (the frequency distribution of tokens), lower-bounded by the entropy of $p$ itself:

$$\min\_q H(p, q) = H(p) \approx 7.51 \text{ nats (measured on our val data)}$$

The collapsed model's 8.50 sits just 1 nat above this floor — a near-optimal constant predictor. Direct measurement confirms constancy: across 64 sampled positions, the mean pairwise KL divergence between output distributions (a measure of how much two distributions differ; 0 means identical) is only **0.007**, versus 1–10 for normal models.

### 3. The Collapse Mechanism

**Why does ASVD cause collapse while plain SVD does not?**

ASVD weights each layer by $S = \text{diag}(\mathbb{E}[\lvert x\_i\rvert])$ — a diagonal matrix whose $i$-th entry is the average activation magnitude of input channel $i$. The key observation: **adjacent layers have highly similar $S$** (residual connections keep activation distributions changing slowly from layer to layer). This means:

- Every layer's SVD truncation preferentially retains directions where $S$ is large
- Every layer discards directions where $S$ is small
- Over 36 layers, small-$S$ directions are **suppressed 36 times over**
- The hidden state ultimately retains only the few large-$S$ directions → **all tokens' representations collapse to the same direction**

Plain SVD uses no weighting; the retained directions are determined by each layer's own singular structure — different layers have different principal directions, so there is no systematic suppression of the same set of directions.

Experiments confirm this. Measuring the cosine similarity between student and teacher hidden states layer by layer:

| Layer | ASVD cos | Interpretation |
|---|---|---|
| 0 (after embedding) | 1.000 | Input is perfect |
| 1 (after Block 0) | 0.718 | First layer already deviates |
| 10 | 0.496 | Hits bottom |
| 11–24 | 0.49–0.53 | Oscillates on a **plateau** |
| 35 (final block) | 0.830 | Recovers |
| 36 (after final norm) | 0.950 | Appears "aligned" |

The final cos=0.95 looks like "good alignment" — but it is an **illusion**. It is not that each token aligns with its teacher counterpart, but that **all tokens have collapsed to the same direction**, and that fixed direction happens to have cos=0.95 with some "average direction" of the teacher.

### 3.5 Microscopic Mechanism: The MLP Is the Killer

Further diagnostic experiments traced the **effective rank (erank)** through every sub-component of the network. Erank measures how many independent directions a batch of vectors actually spans: the hidden state is nominally 4096-dimensional, but if all tokens' vectors crowd into a few directions, the erank is in the single digits. It is a direct readout of representation diversity.

> **What was measured**: the traces in this section come from the uniform rank=384, $\alpha=0.5$ ASVD variant (val loss 10.83). It does not fully collapse into a constant function (it retains 546 unique predictions), but its representations are severely degenerate: final erank=7, and the mean pairwise cosine similarity between different tokens' output vectors (hereafter pcos; closer to 1 means the outputs are more alike) reaches 0.92 — a suitable subject for observing how token diversity gets depleted layer by layer. The fully collapsed 8.50 model ($\alpha=1.0$ + per-layer ranks) is a more extreme instance of the same mechanism.

**Effective rank in the residual stream:**

| Block | Teacher | ASVD | Plain SVD |
|---|---|---|---|
| 0 (after embedding) | 264 | 264 | 264 |
| 3 | 353 | 233 | 4.5 |
| 4 | 355 | 99 | 10 |
| 7–16 | 65–120 | 124–152 | 13–14 |
| 35 | 301 | 57 | 25 |
| 36 (after final norm) | 156 | **7** | **111** |

Surprising finding: **Plain SVD collapses MORE severely in intermediate layers** (Block 3 erank=4.5 vs ASVD's 233), yet plain SVD recovers to erank=111 while ASVD collapses to 7.

**The SwiGLU MLP drives the per-block collapse.** Per-block decomposition:

| Block | Attention contribution | MLP contribution | Net change |
|---|---|---|---|
| 0 | +19.5 | -14.5 | +5.0 |
| 2 | +7.5 | -31.1 | -23.5 |
| **3** | +7.3 | **-141.8** | **-134.5** |
| 4 | +5.2 | -23.4 | -18.3 |

Block 3's MLP delivers a fatal blow: erank drops from 233 to ~99 in a single step.

**Tracing inside the MLP** (Block 3):

| Sub-step | Teacher erank | ASVD erank | Explanation |
|---|---|---|---|
| MLP input | 245 | 163 | |
| gate\_proj | 53 | 20 | Gate projection is inherently low-rank |
| SiLU(gate) | 334 | 165 | Nonlinearity restores diversity |
| up\_proj | 315 | 148 | |
| gate×up | 352 | **71** | Multiplication squares the error |
| MLP output | 329 | **28** | down\_proj compresses again |

**Why ASVD cannot recover while plain SVD can:**

The key is not the MLP output magnitude, but **whether the MLP output differs across tokens**:

| Metric (36-layer average) | Teacher | ASVD | Plain SVD |
|---|---|---|---|
| MLP output pcos | 0.20 | **0.73** | 0.27 |
| MLP override ratio | 0.40 | 0.30 | 1.40 |

(Override ratio = the magnitude of the MLP output relative to the residual stream itself — how strongly the MLP rewrites the stream.)

**ASVD's MLP outputs are nearly identical across all tokens** (pcos=0.73), while plain SVD's MLP outputs differ per token (pcos=0.27).

In the residual addition $h\_{new} = h\_{old} + \text{MLP}(h\_{old})$, ASVD adds nearly the same offset to every token. Over 36 layers, all representations converge to the same direction. Plain SVD's larger but token-diverse offsets do not cause convergence.

**Root cause**: When down\_proj maps the 12288-dim gate×up result back to 4096 dimensions with rank=384, ASVD's activation weighting causes it to retain the **shared components** across tokens (high-activation channels) while discarding the **token-specific differences**.

### 4. Connection to Literature

This phenomenon has theoretical explanations in transformer research:

- **Signal Propagation in Transformers (2022)**: Stacking self-attention layers causes rank collapse of token representations
- **Attention Masks and LayerNorm (2024)**: "With proper choice of value matrices, sequences may not converge to rank one subspace" — implying that improper value matrices cause collapse
- **BA-LoRA (2024)**: Identifies representation collapse in LoRA training; solution is diversity regularization

Our case is an extreme version of "improper value matrices" — low-rank truncation systematically aligns all layers' projection directions, fulfilling the conditions for collapse.

### 5. Attempts to Break the Collapse

Since the collapse comes from "36 layers repeatedly keeping the same set of directions," a natural countermeasure is to force adjacent layers to keep different directions: add an orthogonality penalty during each layer's truncation (0.3 in the table is the penalty weight), or simply alternate ASVD and plain SVD across layers:

| Method | val loss | Unique predictions | Top-1 fraction |
|---|---|---|---|
| ASVD (no constraint) | 8.50 | 1 | 100% |
| Orthogonality 0.3 | 12.80 | 23 | 93.6% |
| Alternating ASVD/plain | 14.56 | 41 | 35.2% |
| Plain SVD (no weighting) | 18.65 | 401 | 15.9% |

**Orthogonality constraints do break the collapse** (diversity goes from 1 to 23 to 401), but **val loss simultaneously worsens**.

Further rank-reallocation experiments (run on the uniform rank=384, $\alpha=0.5$ variant, whose own baseline is 10.83) show that even with far more parameters for key components, no configuration approaches the collapsed model's loss:

| Configuration | Params | val loss | Collapsed? |
|---|---|---|---|
| Collapsed ASVD ($\alpha=1.0$ + per-layer ranks) | 2.29B | **8.50** | Yes (98.6% comma) |
| Uniform rank=384, $\alpha=0.5$ (baseline for the boosts) | 2.29B | 10.83 | No |
| + gate\_proj full rank | 3.88B | 14.22 | No |
| + down\_proj full rank | 3.88B | 11.32 | No |
| + down\_proj rank=1024 | 2.67B | 11.33 | No |
| + down\_proj rank=768 (budget-matched) | 2.29B | 12.31 | No |

**No non-collapsed configuration comes close to the collapsed 8.50** — even with 70% more parameters (3.88B vs 2.29B). This reveals a fundamental tradeoff: within this family of configurations, **the more complete the collapse, the lower the loss; breaking the collapse makes the loss worse**.

The reason is Section 2 again: when the model **genuinely lacks the capacity to distinguish tokens** (rank 384 at 85% compression), collapsing to frequent tokens IS the loss-optimal strategy, and breaking the collapse forces the model to attempt what it cannot do.

### 5.5 Verification: The Collapse Is Real, but Config-Sensitive

Because "the whole model predicts only commas" is so counter-intuitive, we independently re-verified it: re-evaluating the saved checkpoint, re-measuring on held-out data, checking pairwise KL between positions, and comparing against the information-theoretic floor. Verdict: **the collapse is fully reproducible** (val loss 8.5008, comma at 98.6%, cross-position KL = 0.007, top-1 prob 0.095 ≈ a constant predictor).

The verification also exposed an important fact: **collapse is highly sensitive to the factorization configuration** —

| Configuration | val loss | Collapsed? |
|---|---|---|
| diag, $\alpha=1.0$, per-layer ranks (Block 0 = 32) | **8.50** | **Yes** (KL=0.007) |
| diag, $\alpha=0.5$, per-layer ranks | 9.66 | No (389 unique, KL=3.1) |
| diag, $\alpha=0.5$, uniform rank=384 | 10.83 | No (546 unique, KL=0.67) |

Only the combination "stronger activation weighting ($\alpha=1.0$) + a starved Block 0 (rank 32)" triggers full collapse. Halving $\alpha$ or switching to uniform ranks leaves the model in a "severely degenerate but not collapsed" state — with worse loss. This is fully consistent with the tradeoff in Section 5: **within this family of variants, the more complete the collapse, the lower the loss** (8.50 < 9.66 < 10.83).

### 6. Conclusions

1. **Val loss is not a reliable indicator of model quality** — a fully collapsed model (only outputs comma) can have lower loss than a model with prediction diversity but poor accuracy.

2. **ASVD's activation weighting causes representation collapse at extreme compression ratios** — the stronger the weighting (larger $\alpha$) and the more severely key layers are starved by rank allocation, the more complete the collapse.

3. **This collapse is a loss-optimal degeneration** — at the given parameter budget, the model genuinely cannot distinguish tokens well, so collapsing to the most frequent token is a rational "give-up strategy." Quantitatively: the collapsed model's 8.50 sits just 1 nat above the constant-predictor floor H(unigram)=7.51, and beats every non-collapsed variant we measured (9.66–14.2).

4. ~~Recovering true predictive ability requires end-to-end training — closed-form methods cannot simultaneously break collapse and lower loss.~~ **[Later correction] This conclusion was overturned**: the flaw was not closed-form methods per se, but the "approximate $W$ per layer" objective every tested method shared. Switching the layerwise objective to "regress the teacher-trajectory output from the student's drifted input" lets a closed-form method break the collapse AND reach **5.59** (below the constant-predictor floor 7.51, carrying genuine mutual information). See [Trajectory-Correcting Linear Distillation](/2026/08/19/trajectory-correcting-linear-distillation/).

5. **When evaluating compressed models, prediction diversity must be checked alongside val loss** — otherwise the "collapse to frequent token" illusion may mislead. Likewise, **when comparing compression methods, every configuration knob must be held fixed** (weighting strength $\alpha$, rank allocation scheme) — we ourselves once conflated two differently-configured models (8.50 and 10.83) as the same baseline.

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
  var postTitles = {zh: '低秩压缩中的表示坍缩：为什么 val loss 好不等于模型好', en: 'Representation Collapse in Low-Rank Compression: When Low Loss Means a Broken Model'};
  var titleEl = document.querySelector('.post-title');
  if (titleEl) titleEl.textContent = postTitles[lang];
}
switchLang('zh');
</script>
