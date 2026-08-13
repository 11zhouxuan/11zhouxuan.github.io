---
title: "Representation Collapse in Low-Rank Compression: When Low Loss Means a Broken Model"
date: 2026-08-08
mathjax: true
tags: [math, linear-algebra, LLM, compression, representation-collapse]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 低秩压缩中的表示坍缩：为什么 val loss 好不等于模型好

### 1. 一个意外的发现

对一个 36 层的 LLM 做低秩分解（每个线性层 $W$ 替换为 $AB$，rank=384），使用激活加权 SVD（ASVD）得到的模型 val loss = 8.50。作为对比，plain SVD 的 val loss = 18.65——ASVD 看似好了一倍多。

但当我们检查模型实际预测的 token 时，发现了一个惊人的事实：

| 方法 | val loss | 预测的 unique token 数 | top-1 预测占比 | top-1 是什么 |
|---|---|---|---|---|
| 教师（满秩） | 2.11 | 1933 | 8.2% | " the" |
| Plain SVD | 18.65 | 433 | 15.9% | " :\r\n" |
| Random 权重 | 17.77 | 1104 | 13.7% | "fcc" |
| **ASVD** | **8.50** | **1** | **100%** | **","** |

**ASVD 模型对所有 8191 个位置都预测同一个 token——逗号。**

### 2. 为什么坍缩反而 loss 更低？

这看似矛盾：一个"更坏"的模型（只能输出一个 token）怎么比"更好"的模型（能输出 433 种 token）loss 更低？

原因是交叉熵 loss 的性质。当模型**完全没有预测能力**时，最优策略是：

$$\text{把所有概率集中到出现频率最高的 token 上}$$

逗号在英文文本中出现频率约 3.6%。如果模型以 ~10% 的概率输出逗号（softmax 后 top-1 prob = 0.0995），这给出 CE ≈ 8.5。而如果模型试图**区分不同 token 但区分得很差**（random/plain SVD），每个预测都高 confidence 指向错误的 token → CE 远超 11.93（均匀分布）甚至到 17~19。

**"全押一个高频 token"是预测能力为零时的 loss-最优退化策略。** ASVD 的 8.50 不代表"逼近得好"，而是代表"模型已经放弃预测，退化成了常数函数"。

### 3. 坍缩的机制

**为什么 ASVD 导致坍缩而 plain SVD 不会？**

ASVD 对每层用 $S = \text{diag}(\mathbb{E}[\lvert x_i\rvert])$ 加权。关键观察：**相邻层的 $S$ 高度相似**（因为残差连接让激活分布变化缓慢）。这意味着：

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

### 4. 文献中的对应

这个现象在 transformer 研究中已有理论解释：

- **Signal Propagation in Transformers (2022)**：堆叠 self-attention 层导致 token 表示的 rank collapse
- **Attention Masks and LayerNorm (2024)**："With proper choice of value matrices, sequences may not converge to rank one subspace" —— 暗示 value 矩阵选择不当会导致坍缩
- **BA-LoRA (2024)**：在 LoRA 训练中识别出 representation collapse，解决方案是 diversity 正则化

我们的情况属于"value 矩阵选择不当"的极端版本——低秩截断让所有层的投影方向系统性地对齐，满足了坍缩的条件。

### 5. 打破坍缩的尝试

我们尝试了对相邻层的输出方向施加正交约束：

| 方法 | val loss | unique 预测 | top-1 占比 |
|---|---|---|---|
| ASVD（无约束） | 8.50 | 1 | 100% |
| 正交约束 0.3 | 12.80 | 23 | 93.6% |
| 交替 ASVD/plain SVD | 14.56 | 41 | 35.2% |
| Plain SVD（无加权） | 18.65 | 401 | 15.9% |

**正交约束确实打破了坍缩**（预测多样性从 1 到 23 到 401），但 **val loss 同时变差**。这揭示了一个根本的 tradeoff：

$$\text{坍缩程度} \uparrow \quad \Longleftrightarrow \quad \text{val loss} \downarrow$$

在模型**真的没有足够能力区分 token** 的情况下（rank=384 在 85% 压缩率下），坍缩到高频 token 就是 loss-最优的策略。**打破坍缩等于强迫模型做它做不到的事 → loss 变差。**

### 6. 结论

1. **val loss 不是模型质量的可靠指标**——一个完全坍缩的模型（只输出逗号）可以比一个"有预测多样性但不准"的模型 loss 更低。

2. **ASVD 的激活加权在极端压缩率下会导致 representation collapse**——因为所有层的截断方向被系统性地对齐。

3. **这个 collapse 是 loss-optimal 的退化**——在当前的参数预算下，模型确实没有能力做好 token 区分，坍缩到高频 token 是理性的"放弃策略"。

4. **要恢复真正的预测能力（从坍缩的 8.50 到有意义的 3.79），必须经过端到端训练**——闭式方法无法同时打破坍缩又降低 loss，因为问题在于**预测能力本身**（需要学习），不在于逼近精度。

5. **在评估压缩模型时，除了 val loss 还必须检查预测多样性**——否则可能被"坍缩到高频 token"的假象误导。

</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Representation Collapse in Low-Rank Compression: When Low Loss Means a Broken Model

### 1. An Unexpected Finding

Applying low-rank factorization to a 36-layer LLM (replacing each linear layer $W$ with $AB$, rank=384) using activation-weighted SVD (ASVD) yields a val loss of 8.50. For comparison, plain SVD gives 18.65 — ASVD appears more than twice as good.

But when we inspect the model's actual token predictions:

| Method | val loss | Unique tokens predicted | Top-1 fraction | Top-1 token |
|---|---|---|---|---|
| Teacher (full rank) | 2.11 | 1933 | 8.2% | " the" |
| Plain SVD | 18.65 | 433 | 15.9% | " :\r\n" |
| Random weights | 17.77 | 1104 | 13.7% | "fcc" |
| **ASVD** | **8.50** | **1** | **100%** | **","** |

**The ASVD model predicts the same token — a comma — for all 8191 positions.**

### 2. Why Does Collapse Produce Lower Loss?

This seems paradoxical: how can a "worse" model (only one token) have lower loss than a "better" one (433 distinct tokens)?

The answer lies in the nature of cross-entropy loss. When a model has **zero predictive ability**, the optimal strategy is:

$$\text{Concentrate all probability on the highest-frequency token}$$

Commas appear in ~3.6% of positions in English text. If the model outputs ~10% probability for comma (softmax top-1 prob = 0.0995), this gives CE ≈ 8.5. Meanwhile, a model that **tries to distinguish tokens but fails** (random/plain SVD) outputs high-confidence predictions pointing at wrong tokens → CE far exceeds 11.93 (uniform) and reaches 17–19.

**"All-in on one frequent token" is the loss-optimal degenerate strategy when predictive ability is zero.** ASVD's 8.50 does not mean "good approximation" — it means "the model has given up predicting and collapsed to a constant function."

### 3. The Collapse Mechanism

**Why does ASVD cause collapse while plain SVD does not?**

ASVD weights each layer by $S = \text{diag}(\mathbb{E}[\lvert x_i\rvert])$. The key observation: **adjacent layers have highly similar $S$** (because residual connections keep activation distributions stable). This means:

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

### 4. Connection to Literature

This phenomenon has theoretical explanations in transformer research:

- **Signal Propagation in Transformers (2022)**: Stacking self-attention layers causes rank collapse of token representations
- **Attention Masks and LayerNorm (2024)**: "With proper choice of value matrices, sequences may not converge to rank one subspace" — implying that improper value matrices cause collapse
- **BA-LoRA (2024)**: Identifies representation collapse in LoRA training; solution is diversity regularization

Our case is an extreme version of "improper value matrices" — low-rank truncation systematically aligns all layers' projection directions, fulfilling the conditions for collapse.

### 5. Attempts to Break the Collapse

We tried imposing orthogonality constraints between adjacent layers' output directions:

| Method | val loss | Unique predictions | Top-1 fraction |
|---|---|---|---|
| ASVD (no constraint) | 8.50 | 1 | 100% |
| Orthogonality 0.3 | 12.80 | 23 | 93.6% |
| Alternating ASVD/plain | 14.56 | 41 | 35.2% |
| Plain SVD (no weighting) | 18.65 | 401 | 15.9% |

**Orthogonality constraints do break the collapse** (diversity goes from 1 to 23 to 401), but **val loss simultaneously worsens**. This reveals a fundamental tradeoff:

$$\text{Collapse} \uparrow \quad \Longleftrightarrow \quad \text{val loss} \downarrow$$

When the model **genuinely lacks the capacity to distinguish tokens** (rank=384 at 85% compression), collapsing to the highest-frequency token IS the loss-optimal strategy. **Breaking collapse forces the model to attempt what it cannot do → loss increases.**

### 6. Conclusions

1. **Val loss is not a reliable indicator of model quality** — a fully collapsed model (only outputs comma) can have lower loss than a model with prediction diversity but poor accuracy.

2. **ASVD's activation weighting causes representation collapse at extreme compression ratios** — because all layers' truncation directions become systematically aligned.

3. **This collapse is a loss-optimal degeneration** — at the given parameter budget, the model genuinely cannot distinguish tokens well, so collapsing to the most frequent token is a rational "give-up strategy."

4. **Recovering true predictive ability (from collapsed 8.50 to meaningful 3.79) requires end-to-end training** — closed-form methods cannot simultaneously break collapse and lower loss, because the problem is about **predictive capacity itself** (which must be learned), not approximation accuracy.

5. **When evaluating compressed models, prediction diversity must be checked alongside val loss** — otherwise the "collapse to frequent token" illusion may mislead.

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
