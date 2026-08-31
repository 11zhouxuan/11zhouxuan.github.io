---
title: "Anatomy of the Closed-Form Ceiling: Three Ledgers, an Oracle Decomposition, and Dividends That Don't Stack"
date: 2026-08-25
mathjax: true
tags: [math, linear-algebra, LLM, compression, distillation, low-rank, oracle-bound, rank-allocation]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 闭式天花板的解剖：三笔账、oracle 分解与不可叠加的红利

> 📖 如果你不熟悉语言模型的基本词汇（loss、残差流、SVD、蒸馏……），建议先读[预备知识篇](/2026/08/30/lord-compression-primer/)，10 分钟即可补齐全部背景。

[三部曲的完结篇](/2026/08/22/closed-form-ceiling/)宣布闭式赛道在 5.09 收官。然后我们又打了四个实验方向，把等预算纪录推到 **5.02**，并且这一次拿到了"真的到头了"的结构性证据。这篇讲这最后 0.07 nat 是怎么挣的——以及一路上收获的四个比 0.07 值钱得多的机制发现：loss 敏感度的模式/内容通路二分、RMSNorm×巨通道的乘性放大器、**截断税的深度分布**（一个 oracle 分解实验），和"小红利不叠加定律"。

背景（一句话版）：Qwen3-8B（val loss 2.11）→ 每个 linear 换 rank-384 的 $AB$（2.29B，删 72% 参数），闭式方法 = 逐层轨迹矫正回归 + lm_head 矫正 + 残差流矫正器，起点 5.09。

### 1. 三笔账：5.05 到 2.11 之间的钱花在哪了

在动手之前先回答一个架构级的问题：闭式为什么恰好卡在 5 附近？把全部实验证据（oracle 界、悬崖诊断、消融）拼起来，从 teacher 到闭式的 ~2.9 nat 可以拆成三笔**互相独立、机制不同**的账：

$$2.11 \xrightarrow{\text{账1: 截断税} \approx 2.0} 4.15\_{\text{oracle}} \xrightarrow{\text{账2: 漂移残余} \approx 0.9} 5.05 \qquad 3.2\_{\text{训练}} \xleftarrow{\text{账3: 重组红利}}$$

**账 1（截断税，最大头）**：即使每个子层都收到教师的干净输入，rank-384 的局部截断误差之和也要付 ~2 nat。为什么这么贵？白化谱给出答案——rank-384 下各类矩阵保留的能量：

| 矩阵 | 保留能量@384 | 达到 95% 需要的 rank |
|---|---|---|
| k_proj | 98.0% | 211 |
| q_proj | 94.7% | 402 |
| v_proj | 93.4% | 432 |
| o_proj | 89.7% | 642 |
| gate_proj | 89.5% | 687 |
| up_proj | 82.9% | 983 |
| **down_proj** | **66.2%** | **1839** |

网络分成两条通路：**模式通路**（q/k → softmax）本质低秩且对漂移鲁棒（softmax 饱和，attention 分布接近 argmax）；**内容通路**（v→o、up→down 的写路径）**重尾谱、用满全宽**——down_proj 是把 12288 维乘积空间里超位置（superposition）存储的特征字典解码回残差流的字典矩阵，天生没有低秩结构。这笔税是信息论性质的，对"更好的逼近算法"免疫。

**账 2（漂移残余）**：实际 5.05 与 oracle 4.15 之间是穿过非线性复合后线性矫正器修不掉的漂移，载体是两个乘性非线性——SwiGLU 的 $\delta\_g\delta\_u$ 二阶项和 RMSNorm 的除法（第 4 节有它的精确机制）。

**账 3（重组红利）**：训练（3.2）打穿 oracle（4.15）的唯一解释——残差架构下最终计算是 36 个 block 贡献的**和**，同一总函数有无数种分解方式，梯度下降找到了比"模仿教师原分解"更适合 rank-384 的新分工。逐层模仿目标定义上拿不到这笔钱。

三笔账对应三把不同的锁。本篇的四个实验方向全部发生在账 1 和账 2 的边界上。

### 2. 文献坐标：我们到底在前沿的什么位置

动手前做了一轮系统调研（2024-2026 的 training-free 低秩压缩文献），两个结论校准了预期：

1. **没有任何已发表的 training-free 方法在 70%+ 参数削减下报告过可用结果。** 纯闭式的最远数据点是 SVD-LLM 在 LLaMA-7B 删 60% 时 PPL 53.7——折成 loss 约 4.0，与我们的 oracle 界惊人吻合：文献从侧面证实这堵墙的量级是真的。我们在删 72% 拿 5.0x，在前沿之上。
2. **非均匀 rank 分配是文献中最一致的红利来源**（Dobi-SVD、SVD-LLM v2、D-Rank、UniRank、AIR、LACE-SVD 六篇独立工作同向）：高压缩下"分配比分解本身更重要"。这直接触发了实验方向一。

### 3. 实验方向一：loss 敏感度驱动的 rank 分配——方向对，但要阻尼

我们之前试过从白化谱做分配（重构误差驱动），失败（5.72）。文献说有效的是 **loss 敏感度驱动**——这是两回事。做法：在每个保留的 rank-1 方向上挂一个乘性门 $\alpha\_i=1$，反传只求门的梯度，累积 Fisher $= \sum(\partial L/\partial\alpha\_i)^2$；用尾部 Fisher 校准白化谱（$f\_{l,i}\approx c\_l\sigma\_{l,i}^2$），等预算 water-filling。

Fisher 给出的分配极其激进，且完美对应模式/内容通路二分：

| 矩阵 | Fisher 最优 rank（均值） | 对均匀 316 |
|---|---|---|
| o_proj / v_proj | 709 / 637 | ↑ 2 倍 |
| down_proj | 476 | ↑ 1.5 倍 |
| q_proj / k_proj / gate_proj | 77 / 92 / 128 | ↓ 3-4 倍 |

**照单全收 → 5.30（失败）**；把偏离限制在均匀值的 0.6~1.6 倍 → **5.05（当时的新纪录）**。教训：Fisher 门是局部量（"在当前解附近动一动"的敏感度），砍 4 倍已经不在它的适用域里；三点剂量曲线（不倾斜 5.09 / 阻尼 5.05 / 极端 5.30）显示阻尼点就在最优附近。方向的机制解释很干净：q/k 只决定经 softmax 归一化的 attention 权重，天然抗漂移；v/o/down 直接携带写进残差流的内容。

### 4. 实验方向二：巨激活通道——双负结果与真正的放大机制

测量教师的输出侧异常，发现文献意义上的 massive activations 在这个模型里是**两条贯穿全网的全局通道**：维度 1838 和 2276，幅度是中位通道的 60-110 倍，从 block 0 单调增长到 block 33（mean|x| 从 4.9 涨到 607）；几乎每一层 o_proj/down_proj 的最大输出行都对准它们。

顺着"免税矫正器"的思路做了两个等预算直通实验：巨输入列不进 SVD 截断（每层 top-24 列全秩直通）、巨输出行不进截断（o/down 的 top-8 行）。**双双打平。** 结论反而更有价值：白化 SVD 的目标函数被巨行巨列主导，top 奇异方向**天然优先伺候它们**——巨通道根本不付截断税。这与 lm_head/残差矫正器的成功形成干净对比：那两处是整个矩阵没人矫正，这里是矩阵内部早已被照顾好的通道。

但测量顺带揭示了账 2 的精确机制：**RMSNorm 的分母被这两维主导——它们一漂移，全部 4096 维跟着乘性重缩放**。这是一个把局部小误差全局化的耦合器，也是此前"悬崖"（block 16 一层砍掉残差流 R² 0.54）的本体。伤害全部来自漂移的流动而非权重的近似——静态权重侧的任何安排（dense、front-load、直通，四连败）都碰不到它。

### 5. 实验方向三：rms-lift 非线性矫正器——机制证实，红利归零

既然放大器是 RMSNorm 的乘性耦合，对症的钥匙就是给残差矫正器一个能表达"缩放校正"的特征空间：把回归输入从 $h$ 提升到 $[h;\ h\cdot(\bar r/\mathrm{rms}(h)-1)]$——第二块是每 token 的**尺度异常分量**。对参数仍然线性，ridge 依旧闭式可解。

两个教训先于结果到来。第一枪用了朴素形式 $[h;\ h/\mathrm{rms}(h)]$：两块特征高度共线且尺度差百倍，ridge 给出巨大的相互抵消权重——in-sample R² 暴涨、held-out 反而变差。**lift 特征必须中心化 + 尺度匹配。** 重条件化（异常分量形式 + $\lambda=10^{-3}$）后：

| 配置 | held-out | 判决 |
|---|---|---|
| 线性矫正器（基线） | 5.13 / 5.05 | |
| rms-lift 全秩（**+8% 参数**） | **5.06 / 5.00** | 闭式首破 5.0；悬崖后矫正器 R² 0.36→0.66 |
| rms-lift 截断 rank-256（等预算） | 5.13 / 5.04 | 打平 |

**机制证实，红利归零**：乘性漂移确实可修（R² 与破 5.0 为证），但修复的参数效率与直接加 rank 恰好平价——等预算下没有套利空间。这与免税矫正器原理形成闭环：lm_head 和线性残差矫正器赢在修复"没人管的漂移"，而 lift 的非线性部分在修复"已被线性工具照顾过的残余"，边际收益自然平价。

### 6. 实验方向四：oracle 分解——逐位置测量截断损失的分布

完整 oracle（换掉全部子层输入）只给总量。把它拆开——每次只换一部分输入：

| 换什么 | loss | 相对基线 5.13 |
|---|---|---|
| 全部（完整 oracle） | 4.15 | +0.98 |
| 只换 q/k/v 入口 | 4.51 | +0.62 |
| 只换 gate/up 入口 | 4.57 | +0.56 |
| **只换 block 18-35** | **4.14** | **+0.99 ≈ 完整 oracle！** |
| 只换 block 0-17 | 5.53 | **−0.40（变差）** |
| 只换 block 12-19（悬崖区） | 5.49 | −0.36（变差） |

两个重磅结论：

**其一：截断税几乎全部集中在后半段。** 只把 block 18-35 的输入换干净，loss 就到了完整 oracle 的水平——意味着前半段的局部截断误差对 logits 的直接贡献只有 ~0.01 nat，它的全部危害走"制造漂移"这条间接通道；而漂移是（部分）可矫正的，后半段的截断税则无药可救。**推论：边际 rank 在后半段的价值远高于前半段。** 我们此前试过的恰好是反方向（front-load，5.38 失败——当时的理论"减少流入悬崖的漂移"错得离谱）。

**其二：局部"清洁"上游反而伤害下游（−0.40）。** 顺序管线里每一层和矫正器都是**适配上游特定误差模式**拟合的，突然给它们干净的中间值，下游的补偿全部失配。这是矫正链自洽性的第三次独立验证（前两次：不动点迭代 5.94、跨层联合优化的文献负结果）。

据此做 back-load：前半 rank ×0.8、后半按预算配平放大（×1.24）。**5.11 / 5.02——最终纪录。** 剂量曲线：×1.0 → 5.09、×0.8 → 5.07（均值，最优）、×0.65 → 5.08（回弹）。

最后一个实验是把两个已验证的小红利叠加（back-load + rms-lift-256）：**5.13 / 5.11，失败**。0.02 级的红利在同一块残余误差上重叠收割，不叠加。

### 7. 结论：三把锁、三把钥匙与三条定律

**最终版图**（85% 压缩率、等预算 2.29B，held-out）：

$$8.50\_{\text{坍缩假象}} \to 5.59\_{\text{轨迹矫正}} \to 5.09\_{\text{免税矫正器}} \to \mathbf{5.02}\_{\text{+分配+back-load}} \to 4.15\_{\text{oracle}} \to 3.2\_{\text{训练@500}} \to 2.11\_{\text{教师}}$$

**【后续更新】** 这个"终点"随后又被推到 **4.71/4.53**：稀疏残差与度量修正的超线性组合 + 校准数据工程。详见[第五篇《会移动的天花板》](/2026/08/30/closed-form-moving-ceiling/)。

| 锁 | 架构根源 | 钥匙 |
|---|---|---|
| 截断税 ~2.0（集中在后半段） | 内容通路的超位置重尾谱 | 改变 rank 的放置几何（联合分解、MLA、稀疏残差）——范式外 |
| 漂移残余 ~0.9 | SwiGLU 乘法 + RMSNorm×巨通道 | 非线性矫正器可修但参数效率平价——已榨干 |
| 重组红利 ~0.9 | 残差和的可重分解性 | 只有全局优化（训练） |

三条从这轮实验里提炼的、可迁移的定律：

1. **oracle 分解是比 oracle 本身强得多的诊断工具。** 完整 oracle 只告诉你天花板在哪，部分 oracle 告诉你**误差住在哪**——"late ≈ all"一个数字就推翻了我们持有数周的 front-load 直觉，并直接兑换成了纪录。
2. **矫正链是一个自洽整体。** 顺序拟合的压缩管线里，每个组件都适配上游的特定误差。任何"局部改善"（不动点重解、跨层联合目标、部分输入清洁）都会破坏下游补偿——三次独立验证。诊断时可以作弊，部署时只能整链重拟合。
3. **小红利不叠加。** 当各手段（分配、直通、非线性 lift）都在收割同一块残余误差时，0.02 级的收益合并后互相侵蚀。判断两个改进是否正交，看它们的机制是否攻击不同的账——同账的红利做加法之前先做减法。

闭式赛道就此真正收官：等预算 ~5.0-5.07 是"逐层回归 + 线性/轻非线性矫正 + rank 分配"这套工具箱的终点。通往 2.5 的路在训练上（本文发稿时训练臂已到 2.87 且在下降），通往 4.15 以下的闭式路在范式外。



</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Anatomy of the Closed-Form Ceiling: Three Ledgers, an Oracle Decomposition, and Dividends That Don't Stack

> 📖 New to language-model vocabulary (loss, residual stream, SVD, distillation...)? Read [the primer](/2026/08/30/lord-compression-primer/) first — ten minutes covers all the background.

[The trilogy finale](/2026/08/22/closed-form-ceiling/) declared the closed-form track finished at 5.09. We then fought four more campaigns, pushed the equal-budget record to **5.02**, and — this time — obtained structural evidence that it is truly over. This post covers how that last 0.07 nat was earned, and four mechanistic findings worth far more than the 0.07: the pattern/content pathway split in loss sensitivity, the RMSNorm×massive-channel multiplicative amplifier, **the depth distribution of the truncation tax** (an oracle-decomposition experiment), and the "small dividends don't stack" law.

Background in one line: Qwen3-8B (val loss 2.11) → every linear replaced by rank-384 $AB$ (2.29B params, 72% removed); closed-form = layerwise trajectory-correcting regression + lm_head fix + residual-stream correctors, starting at 5.09.

### 1. Three Ledgers: Where the Nats Between 5.05 and 2.11 Are Spent

Before running anything, an architecture-level question: why is closed-form stuck near 5? Assembling all the evidence (oracle bound, cliff diagnosis, ablations), the ~2.9 nats from teacher to closed-form split into three **independent ledgers with distinct mechanisms**:

$$2.11 \xrightarrow{\text{L1: truncation tax} \approx 2.0} 4.15\_{\text{oracle}} \xrightarrow{\text{L2: drift residue} \approx 0.9} 5.05 \qquad 3.2\_{\text{trained}} \xleftarrow{\text{L3: reorganization}}$$

**Ledger 1 (truncation tax, the bulk)**: even if every sublayer received the teacher's clean input, the summed local truncation errors of rank 384 cost ~2 nats. Why so expensive? The whitened spectra answer — energy retained at rank 384 per matrix family:

| Matrix | Energy kept @384 | Rank needed for 95% |
|---|---|---|
| k_proj | 98.0% | 211 |
| q_proj | 94.7% | 402 |
| v_proj | 93.4% | 432 |
| o_proj | 89.7% | 642 |
| gate_proj | 89.5% | 687 |
| up_proj | 82.9% | 983 |
| **down_proj** | **66.2%** | **1839** |

The network splits into two pathways: the **pattern pathway** (q/k → softmax) is intrinsically low-rank and drift-robust (softmax saturates; attention distributions are near-argmax), while the **content pathway** (v→o, up→down write path) has **heavy-tailed spectra using the full width** — down_proj is the dictionary matrix decoding a superposed feature dictionary from the 12288-dim product space back into the residual stream, and dictionary matrices have no low-rank structure to exploit. This tax is information-theoretic; it is immune to "a better approximation algorithm".

**Ledger 2 (drift residue)**: the gap between the practical 5.05 and the oracle 4.15 is drift that survives linear correction after compounding through nonlinearities, carried by two multiplicative nonlinearities — SwiGLU's second-order $\delta\_g\delta\_u$ term and RMSNorm's division (its precise mechanism in Section 4).

**Ledger 3 (reorganization dividend)**: the only explanation for training (3.2) breaking the oracle (4.15) — in a residual architecture the final computation is a **sum** of 36 block contributions; the same total function admits infinitely many decompositions, and gradient descent found one better suited to rank 384 than the teacher's own. Layerwise-imitation objectives cannot collect this money by definition.

Three ledgers, three different locks. All four campaigns below operate on the boundary between Ledgers 1 and 2.

### 2. Literature Coordinates: Where We Actually Sit on the Frontier

A systematic literature sweep (training-free low-rank LLM compression, 2024-2026) calibrated expectations with two findings:

1. **No published training-free method reports usable results at 70%+ parameter removal.** The furthest pure closed-form data point is SVD-LLM at 60% removal on LLaMA-7B: PPL 53.7 ≈ loss 4.0 — startlingly consistent with our oracle bound. The literature independently confirms the wall's magnitude. At 72% removal and 5.0x, we sit ahead of the frontier.
2. **Non-uniform rank allocation is the literature's most consistent dividend** (six independent works — Dobi-SVD, SVD-LLM v2, D-Rank, UniRank, AIR, LACE-SVD — all agree: at high compression, allocation matters more than the decomposition itself). This triggered Campaign 1.

### 3. Campaign 1: Loss-Aware Rank Allocation — Right Direction, Needs Damping

We had previously tried spectrum-driven allocation (reconstruction-error-based) and failed (5.72). What the literature endorses is **loss-sensitivity-driven** allocation — a different thing. Method: hang a multiplicative gate $\alpha\_i=1$ on every kept rank-1 direction, backprop into the gates only, accumulate Fisher $= \sum(\partial L/\partial\alpha\_i)^2$; calibrate the whitened spectra with the tail Fisher ($f\_{l,i}\approx c\_l\sigma\_{l,i}^2$), then water-fill at equal budget.

Fisher's optimal allocation is drastic, and maps perfectly onto the pattern/content split:

| Matrix | Fisher-optimal rank (mean) | vs uniform 316 |
|---|---|---|
| o_proj / v_proj | 709 / 637 | ↑ 2× |
| down_proj | 476 | ↑ 1.5× |
| q_proj / k_proj / gate_proj | 77 / 92 / 128 | ↓ 3-4× |

**Taken at face value → 5.30 (fail)**; deviation clipped to 0.6-1.6× of uniform → **5.05 (record at the time)**. Lesson: gate Fisher is a local quantity — the sensitivity of wiggling around the current solution; a 4× cut is far outside its domain. The three-point dose-response (untilted 5.09 / damped 5.05 / extreme 5.30) puts the optimum near the damped point. The direction has a clean mechanism: q/k only shape attention weights, normalized by softmax and thus naturally drift-robust; v/o/down carry the content written into the residual stream.

### 4. Campaign 2: Massive-Activation Channels — a Double Negative and the True Amplifier

Measuring the teacher's output-side outliers revealed that massive activations in this model are **two global channels running the full depth**: dims 1838 and 2276, 60-110× the median channel, growing monotonically from block 0 to 33 (mean|x| 4.9 → 607); nearly every o_proj/down_proj's largest output row targets them.

Following the tax-free-corrector principle we ran two equal-budget passthrough experiments: exempt the massive input COLUMNS from SVD truncation (top-24 per layer, kept exact), and exempt the massive output ROWS (top-8 of o/down). **Both washed out.** The conclusion is worth more than a win: the whitened SVD's objective is dominated by those very rows/columns, so the top singular directions **serve them first** — massive channels pay no truncation tax to begin with. A clean contrast with the lm_head/residual-corrector successes: those fixed whole matrices nobody was correcting; these are channels inside matrices that were already well served.

But the measurement exposed Ledger 2's precise mechanism: **RMSNorm's denominator is dominated by those two dims — when they drift, all 4096 channels get multiplicatively rescaled.** It is a coupler that globalizes local errors, and the true identity of the earlier "cliff" (block 16 destroying residual R² by 0.54). The damage comes entirely from drift flowing at runtime, not from weight approximation — which is why every static weight-side arrangement (dense block, front-load, passthroughs; four failures) never touched it.

### 5. Campaign 3: rms-Lift Nonlinear Correctors — Mechanism Confirmed, Dividend Zero

If the amplifier is RMSNorm's multiplicative coupling, the surgical key is a corrector feature space that can express scale corrections: lift the regression input from $h$ to $[h;\ h\cdot(\bar r/\mathrm{rms}(h)-1)]$ — the second block being the per-token **scale-anomaly component**. Still linear in parameters; the ridge solve stays closed-form.

Two lessons arrived before the result. The first shot used the naive form $[h;\ h/\mathrm{rms}(h)]$: the two blocks are highly collinear with a 100× scale mismatch, so ridge produced huge mutually-canceling weights — in-sample R² soared while held-out got worse. **Lift features must be centered and scale-matched.** After reconditioning (anomaly form + $\lambda=10^{-3}$):

| Configuration | held-out | Verdict |
|---|---|---|
| Linear correctors (baseline) | 5.13 / 5.05 | |
| rms-lift, full-rank ( **+8% params**) | **5.06 / 5.00** | first closed-form below 5.0; post-cliff corrector R² 0.36→0.66 |
| rms-lift, truncated to rank 256 (equal budget) | 5.13 / 5.04 | wash |

**Mechanism confirmed, dividend zero**: multiplicative drift is genuinely repairable (the R² and the sub-5.0 prove it), but the repair's parameter efficiency exactly matches just buying more rank — no arbitrage at equal budget. This closes the loop on the tax-free-corrector principle: lm_head and the linear residual correctors won by fixing drift nobody was managing; the lift's nonlinear part fixes residue the linear tools had already tended, so its marginal return is at par.

### 6. Campaign 4: The Oracle Decomposition — the Depth Structure of the Tax, and Back-Loading

The full oracle (swap all sublayer inputs with the teacher's clean values) only gives a total. Decompose it — swap one subset at a time:

| Swapped | loss | vs baseline 5.13 |
|---|---|---|
| everything (full oracle) | 4.15 | +0.98 |
| q/k/v entrances only | 4.51 | +0.62 |
| gate/up entrances only | 4.57 | +0.56 |
| **blocks 18-35 only** | **4.14** | **+0.99 ≈ the full oracle!** |
| blocks 0-17 only | 5.53 | **−0.40 (worse)** |
| blocks 12-19 only (cliff zone) | 5.49 | −0.36 (worse) |

Two heavyweight conclusions:

**First: the truncation tax lives almost entirely in the second half.** Cleaning only blocks 18-35's inputs reaches the full oracle — meaning the first half's local truncation errors contribute ~0.01 nat directly to the logits; all their harm travels the indirect "manufacture drift" channel, and drift is (partially) correctable, while the second half's tax is incurable. **Corollary: marginal rank is worth far more in late blocks.** Our earlier attempt was the exact opposite direction (front-loading, 5.38, fail — its rationale "reduce drift flowing into the cliff" was badly wrong).

**Second: locally "cleaning" upstream hurts downstream (−0.40).** In a sequential pipeline every layer and corrector is fit to its upstream's specific error pattern; hand them clean intermediates and downstream compensations misfire en masse. This is the third independent proof of corrector-chain self-consistency (after fixed-point iteration 5.94, and the literature's cross-layer joint-optimization negative).

Hence back-loading: early ranks ×0.8, late ranks budget-balanced up (×1.24). **5.11 / 5.02 — the final record.** Dose-response: ×1.0 → 5.09, ×0.8 → 5.07 (mean, optimum), ×0.65 → 5.08 (rebound).

The last experiment stacked the two verified dividends (back-load + rms-lift-256): **5.13 / 5.11, fail.** Dividends at the 0.02 level harvest the same residue and do not add.

### 7. Conclusions: Three Locks, Three Keys, Three Laws

**The final landscape** (85% compression, equal 2.29B budget, held-out):

$$8.50\_{\text{collapse illusion}} \to 5.59\_{\text{traj. correction}} \to 5.09\_{\text{tax-free correctors}} \to \mathbf{5.02}\_{\text{+alloc+backload}} \to 4.15\_{\text{oracle}} \to 3.2\_{\text{trained@500}} \to 2.11\_{\text{teacher}}$$

**[Later update]** This "endpoint" was subsequently pushed to **4.71/4.53** by the super-additive combination of sparse residuals and the metric fix, plus calibration-data engineering. See [part 5: The Ceiling That Kept Moving](/2026/08/30/closed-form-moving-ceiling/).

| Lock | Architectural root | Key |
|---|---|---|
| Truncation tax ~2.0 (concentrated late) | superposition heavy tails of the content pathway | change WHERE rank lives (joint decompositions, MLA, sparse residual) — outside the paradigm |
| Drift residue ~0.9 | SwiGLU product + RMSNorm×massive channels | nonlinear correctors work but at par parameter efficiency — exhausted |
| Reorganization dividend ~0.9 | re-decomposability of the residual sum | global optimization (training) only |

Three transferable laws distilled from these campaigns:

1. **The oracle decomposition is a far stronger diagnostic than the oracle itself.** The full oracle tells you where the ceiling is; partial oracles tell you **where the error lives** — the single number "late ≈ all" overturned a weeks-held front-loading intuition and converted directly into a record.
2. **The corrector chain is a self-consistent whole.** In a sequentially-fit pipeline every component adapts to its upstream's specific errors. Any "local improvement" (fixed-point re-solving, cross-layer joint objectives, partial input cleaning) breaks downstream compensation — verified three independent ways. Cheat during diagnosis; refit the whole chain for deployment.
3. **Small dividends don't stack.** When separate levers (allocation, passthrough, nonlinear lift) harvest the same residue, 0.02-level gains erode each other on combination. Before adding two improvements, check whether their mechanisms attack different ledgers — same-ledger dividends subtract before they add.

The closed-form track is now genuinely closed: equal-budget ~5.0-5.07 is the terminus of the "layerwise regression + linear/light-nonlinear correctors + rank allocation" toolbox. The road to 2.5 runs through training (the training arm was at 2.87 and descending as this was published); the road below 4.15 for closed-form runs outside the paradigm.



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
  var postTitles = {zh: '闭式天花板的解剖：三笔账、oracle 分解与不可叠加的红利', en: "Anatomy of the Closed-Form Ceiling: Three Ledgers, an Oracle Decomposition, and Dividends That Don't Stack"};
  var titleEl = document.querySelector('.post-title');
  if (titleEl) titleEl.textContent = postTitles[lang];
}
switchLang('zh');
</script>
