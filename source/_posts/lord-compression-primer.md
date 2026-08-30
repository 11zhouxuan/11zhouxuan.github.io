---
title: "A Primer for the Low-Rank Compression Series"
date: 2026-08-30 08:00:00
mathjax: true
tags: [primer, LLM, compression, linear-algebra, tutorial]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 预备知识：读懂"低秩压缩"系列需要的一切

这个系列讲的是一件事：**把一个 80 亿参数的语言模型压缩成 23 亿参数，并且尽量不让它变笨**。系列正文假设读者熟悉一些机器学习词汇，这一篇把需要的背景一次性补齐——只要你学过高等数学和线性代数，读完这篇就能顺畅读完全系列。已经熟悉语言模型的读者可以直接跳到正文，遇到不认识的词再回来查第 8 节的速查表。

### 1. 语言模型在做什么：预测下一个词

语言模型的任务简单得出乎意料：**给定前文，猜下一个词**。

输入"今天天气真"，模型输出一张概率表："好" 35%、"不错" 20%、"冷" 8%……模型的全部能力——写文章、答题、写代码——都是把这个"猜下一个词"连续做几千次的结果。

两个词汇：

- **token**：模型处理文本的最小单位。不完全等于"词"——英文里常见词是一个 token，长词会被拆成几段；中文常见字大约一字一个 token。本系列的模型认识 **151936** 种不同的 token，这张总表叫**词表**（vocabulary）。
- **logits**：模型对每种 token 打的原始分数（151936 个实数），经过 **softmax** 函数（$p_i = e^{z_i}/\sum_j e^{z_j}$，把任意实数组变成总和为 1 的概率）变成上面那张概率表。

### 2. 怎么给模型打分：交叉熵和 nat

怎么衡量模型猜得好不好？让它在一大段真实文本上逐个位置猜下一个 token，每个位置看它给**正确答案**分配了多少概率 $p$，记 $-\ln p$ 分（猜得越准，扣分越少），最后取平均。这个平均值就是全系列出现最多的数字——**交叉熵损失**（cross-entropy loss，简称 loss）：

$$\text{loss} = \text{平均}\big(-\ln p(\text{正确 token})\big)$$

它的单位叫 **nat**（因为用的是自然对数）。几个直觉：

- **loss 越低越好**。loss 每降低 1，模型给正确答案的（几何平均）概率就提高 $e \approx 2.72$ 倍。
- 系列里说的 val loss（validation loss），指在模型**没见过**的文本上算的 loss——防止"背题"造成的虚高分数。同义词还有 held-out loss。

几个贯穿全系列的锚点数字（都在同一份测试文本上）：

| loss | 什么水平 |
|---|---|
| 11.93 | 纯乱猜（对 151936 种 token 均匀分配概率，$\ln 151936 = 11.93$） |
| 7.51 | "从不看上下文"的最优策略：永远按词频猜（"的""是"这类高频词多给概率）。这是**常数预测器的极限**——想比它更低，模型必须真的在理解前文 |
| 2.11 | 压缩前的原模型（Qwen3-8B，80 亿参数） |
| ~2.5 | 我们给压缩模型定的目标 |

举个换算例子：loss 从 8.5 降到 5.6，意味着正确 token 的平均概率提高了 $e^{2.9} \approx 18$ 倍——看似不到一半的数字变化，实际是能力的巨大差距。

### 3. 模型内部长什么样：一条传送带和 36 个加工站

本系列的模型是 Transformer 结构。不需要完整学它，记住下面这幅文字版示意图就够了：

```
输入 token → [嵌入] → 4096 维向量
                ↓
     ┌── 加工站 1（block 0）──┐
     │  注意力部分 + MLP 部分  │ ← 每部分算完把结果"加"回主线
     └───────────↓───────────┘
              ……共 36 个加工站……
                ↓
        [归一化] → [lm_head 打分] → 151936 个 logits
```

- **残差流（residual stream）**：贯穿全程的那条"主线"。每个 token 对应一个 4096 维向量，从头传到尾；每个加工站（block）不是替换它，而是算出一个修正量**加上去**：$h \leftarrow h + \text{本站的输出}$。这个"只做加法"的设计叫**残差连接**，是理解系列中很多现象的关键——比如误差会沿主线一路累积。
- 每个 block 里有两个部分，共 **7 个大矩阵**（每个矩阵就是一次线性变换）：
  - **注意力（attention）**：让每个位置"回头看"前文。四个矩阵各司其职——**q**（query，我在找什么）、**k**（key，每个位置能提供什么）、**v**（value，实际取回的内容）、**o**（output，把取回的内容整理后写回主线）。q 和 k 只负责算"该看哪里"（它们的点积经 softmax 变成注意力权重），v 和 o 负责"搬运内容"——系列正文把前者叫**模式通路**、后者叫**内容通路**，区别就在这里。
  - **MLP**：一个两层的非线性变换。三个矩阵：**gate** 和 **up** 把 4096 维升到 12288 维（gate 那一路过一个非线性函数后与 up 那一路**逐元素相乘**——这个结构叫 SwiGLU，那个乘法在系列里反复出场），**down** 再降回 4096 维写回主线。
- **RMSNorm**：每个部分入口处的归一化层，把向量除以它自身的均方根（root mean square），防止数值越滚越大。注意它是个**除法**——分母由向量里幅度最大的那些分量主导，这个细节在第 4、5 篇会变得重要。
- **lm_head**：最后一站，一个 4096×151936 的矩阵，把 4096 维向量变成词表上每个 token 的得分。

36 个 block × 7 个矩阵 = 252 个大矩阵，占了模型参数的绝大部分。**压缩它们就是压缩模型。**

### 4. 低秩压缩：用两个瘦矩阵代替一个胖矩阵

一个 4096×4096 的矩阵 $W$ 有 1680 万个参数。**低秩近似**的想法：找一个 4096×384 的 $A$ 和一个 384×4096 的 $B$，用乘积 $AB$ 代替 $W$。参数量变成 $384\times(4096+4096) = 315$ 万——**省了 81%**。数字 384 叫这个近似的**秩**（rank）：$AB$ 作为矩阵，秩最多是 384。

代价是什么？$AB$ 的秩最多 384，而 $W$ 的秩可以高达 4096——如果 $W$ 里的信息真需要那么多独立方向，压缩必然丢东西。丢多少、怎么丢得最少，工具是**奇异值分解**（SVD）：任何矩阵都能写成 $W = \sum_i \sigma_i u_i v_i^T$，即一串秩 1 矩阵之和，权重 $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ 叫**奇异值**（可以理解为"$W$ 在各个独立方向上的作用强度"，是特征值概念对非方阵的推广）。有一条经典定理（Eckart–Young）：**只保留前 $r$ 项就是最优的秩 $r$ 近似**，丢掉的误差恰好是被扔掉的奇异值的平方和。

系列里常说的"某矩阵保留了 66% 的能量"，意思就是前 384 个奇异值的平方和占全部平方和的 66%——剩下 34% 的信息被硬生生扔掉了。奇异值衰减快的矩阵（信息集中在少数方向）适合低秩压缩；衰减慢的（信息摊在几千个方向上）怎么压都疼，这正是系列后半的核心矛盾。

### 5. 老师和学生：两条压缩路线

压缩里有两个固定角色：**教师**（teacher）= 压缩前的原模型（80 亿参数，loss 2.11）；**学生**（student）= 压缩后的小模型。让学生模仿教师的过程叫**蒸馏**（distillation）。

造出一个好学生有两条路线，理解它们的区别是读懂全系列的前提：

| | 训练路线 | 闭式路线（closed-form） |
|---|---|---|
| 做法 | 梯度下降：让学生猜词、算 loss、按梯度微调参数，重复几千次 | 解方程：把"学生该是什么样"写成数学问题，**直接解出答案** |
| 成本 | 一块 GPU 跑几十天 | 一块 GPU 跑几小时 |
| 效果 | 更好（能到 loss 3 以下） | 有极限（系列就是在探这个极限） |

打个比方：训练像让学生刷一万道题慢慢开窍；闭式像直接给学生抄一份浓缩笔记——便宜、快，但笔记浓缩得再好也有上限。系列的主线问题就是：**这份"笔记"最好能做到什么程度？** 顺带说明：训练路线也需要一个起点（参数初始值，称为**初始化**或 init），闭式方法的产物正好可以当训练的起点用——所以两条路线是接力而非对立。

还有几个训练路线的词会在正文出现：**step**（一次参数更新，我们每步用约 52 万 token 的数据）、**warmup**（训练开头先把更新步长从 0 缓慢升到设定值，防止一开始就把参数改坏）、**学习率**（learning rate，每步微调的幅度）、**checkpoint**（训练中途存档）。

### 6. 闭式方法的数学核心：一次最小二乘

系列里反复出现一个公式，这里推一遍——只用到线性代数。

问题：学生某一层收到输入向量 $x$，我们希望这一层的输出尽量接近某个目标 $y$（比如"教师在同样位置的输出"）。找一个矩阵 $M$，最小化在大量样本上的平均误差：

$$\min_M\ \mathbb{E}\,\lVert y - Mx \rVert^2$$

这就是**最小二乘回归**。对 $M$ 求导置零，得到解（一行推导，展开平方、逐项求导即可）：

$$M^* = \Sigma_{yx}\,\Sigma_{xx}^{-1}, \qquad \Sigma_{yx} = \mathbb{E}[yx^T],\ \ \Sigma_{xx} = \mathbb{E}[xx^T]$$

$\Sigma_{xx}$ 叫 $x$ 的**协方差矩阵**（描述输入在各方向上的分布强度），$\Sigma_{yx}$ 是**互协方差**。实际中 $\Sigma_{xx}$ 可能接近奇异（不可逆），所以给对角线加一个小量 $\lambda I$ 再求逆——这个稳定化技巧叫**岭回归**（ridge regression），$\lambda$ 是它唯一的超参数。

两个引申：

- **校准数据（calibration data）**：上面的期望 $\mathbb{E}[\cdot]$ 在实践中用一小批真实文本的样本平均来估计。这批文本就叫校准数据——它是闭式方法唯一"见过"的数据，它的数量和多样性直接决定协方差估得准不准（第 5 篇整整一章在讲这件事）。
- **白化（whitening）**：想在"输入的真实分布下"做最优的低秩截断，标准做法是先对 $\Sigma_{xx}$ 做 Cholesky 分解 $\Sigma_{xx} = LL^T$（把对称正定矩阵写成三角矩阵乘积，相当于给空间换一组坐标让输入分布变成"各向均匀"），在新坐标下做 SVD 截断，再换回来。系列里的"白化 SVD 截断"就是这三步。

### 7. 三个常用的诊断工具

系列里反复使用三个"体检工具"，先认识它们：

**R²（决定系数）**：衡量"$y$ 能被 $x$ 线性预测的程度"，取值 0 到 1。R²=1 表示完美线性关系，R²=0.3 表示线性回归只能解释 30% 的变化、剩下 70% 是线性工具够不着的成分。系列里用它回答"压缩模型的激活漂移中，有多少能用线性变换修回去"。

**oracle 实验**：oracle 直译是"神谕"，在这里指**允许作弊的对照实验**。比如想知道"如果每一层都能收到完美的输入，模型能好到什么程度"，就在运行时把学生每层的输入偷偷换成教师的（现实中做不到，所以是作弊），测出来的 loss 就是某一类方法的**理论极限**。它回答的问题是"这条路最远能走多远"，帮我们判断还值不值得继续投入。

**梯度敏感度（Fisher 信息）**：想知道模型里哪个参数重要、哪个可有可无？一个通用办法：算 loss 对它的导数（梯度）在很多样本上的平方平均。直觉：如果轻轻动一下某参数 loss 就剧烈变化（梯度大），它就重要；反之则不重要。这个"梯度平方的平均"叫 Fisher 信息，系列里用它来决定"哪些矩阵该多分一点秩"。

### 8. 术语速查表

读正文时遇到不认识的词，回这里查：

| 术语 | 一句话解释 |
|---|---|
| batch | 一次一起处理的一组文本（我们的场景：1 条 2048 个 token 的文本） |
| shard | 数据集存储时切成的块；同一块内的文本来源相近，跨块混合能提高多样性 |
| seed | 随机数种子；固定它可以让"随机"过程完全重现 |
| KL 散度 | 衡量两个概率分布差多远的量，0 = 完全相同 |
| PPL（困惑度） | loss 的另一种写法：$\text{PPL} = e^{\text{loss}}$，文献中常用 |
| 参数预算 / 等预算 | 限定总参数量（我们固定 22.9 亿）做比较，防止"用更大的模型赢"的不公平对比 |
| erank（有效秩） | 一组向量实际张开了多少个独立方向：1000 个向量若挤在一条直线附近 erank≈1，均匀散开则接近维数 |
| massive activations（巨幅激活） | 大模型中个别固定通道的数值比其他通道大几十到几百倍的现象，近年文献的热点之一 |
| 超位置（superposition） | 神经网络把远多于维数的"特征"挤在同一空间里存储的现象，代价是各特征方向互相不完全正交 |
| 蒸馏 init | 用闭式方法算出的学生参数，当作后续训练的起点 |
| in-sample / held-out | 在"拟合时用过的数据"上测 vs 在"没用过的数据"上测；前者虚高，后者才算数 |
| 消融（ablation） | 逐个拆掉系统的组件分别测效果，确定每个组件的贡献 |
| 校准数据 | 闭式方法用来估计统计量的那一小批文本（见第 6 节） |
| 矫正器（corrector） | 系列自创词：插在模型某处的一个小变换，负责把跑偏的中间结果拉回教师的轨迹 |
| 截断税 | 系列自创词：把矩阵压到固定秩时不可避免的那部分 loss 损失（见第 4 节的"扔掉 34% 能量"） |

最后一点阅读建议：系列各篇的实验结果都以"两个数字"报告（如 4.74 / 4.53），它们是**两段互不重叠的测试文本**上各自的 loss——相当于把每个实验做两次独立验证，两边同时变好才算真改进，只有一边变好多半是运气。



</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## A Primer for the Low-Rank Compression Series

This series is about one thing: **compressing an 8-billion-parameter language model down to 2.3 billion parameters without making it much dumber**. The main posts assume some machine-learning vocabulary; this page fills in all of it at once. If you know calculus and linear algebra, reading this primer should be enough to follow the whole series. If you already know language models, skip ahead and come back to the glossary (Section 8) as needed.

### 1. What a Language Model Does: Predict the Next Word

The task is surprisingly simple: **given the text so far, guess the next word.**

Feed in "The weather today is really", and the model outputs a probability table: "nice" 35%, "cold" 8%, ... Everything a model can do — write essays, answer questions, write code — is this guess repeated thousands of times.

Two words of vocabulary:

- **token**: the smallest unit of text the model handles. Not exactly a word — common English words are one token, long words get split. Our model knows **151,936** distinct tokens; the full list is the **vocabulary**.
- **logits**: the raw scores the model assigns to every token (151,936 real numbers), turned into the probability table by the **softmax** function ($p_i = e^{z_i}/\sum_j e^{z_j}$, which maps any real vector to probabilities summing to 1).

### 2. Scoring a Model: Cross-Entropy and Nats

To measure how well a model guesses, run it over a long stretch of real text, and at every position record the probability $p$ it assigned to the **correct** next token, scoring $-\ln p$ (the better the guess, the smaller the penalty). The average is the number that appears most in this series — the **cross-entropy loss**:

$$\text{loss} = \text{mean}\big(-\ln p(\text{correct token})\big)$$

Its unit is the **nat** (natural log). Intuitions:

- **Lower is better.** Each 1-nat drop means the (geometric-mean) probability on the correct answer grows by a factor of $e \approx 2.72$.
- **val loss** (validation loss) means loss measured on text the model has **never seen** — guarding against inflated scores from memorization. Also called held-out loss.

Anchor numbers used throughout the series (all on the same test text):

| loss | What it means |
|---|---|
| 11.93 | pure random guessing ($\ln 151936$) |
| 7.51 | the best "never look at the context" strategy: always guess by word frequency. To go below this, a model must genuinely read the context |
| 2.11 | the original uncompressed model (Qwen3-8B) |
| ~2.5 | our target for the compressed model |

Example conversion: going from loss 8.5 to 5.6 means the average probability on correct tokens grew by $e^{2.9} \approx 18\times$ — a modest-looking number change hiding a huge capability gap.

### 3. Inside the Model: a Conveyor Belt and 36 Stations

The model is a Transformer. You don't need the full picture — this text diagram suffices:

```
input token → [embedding] → 4096-dim vector
                 ↓
      ┌── station 1 (block 0) ──┐
      │  attention part + MLP   │ ← each part ADDS its result back
      └────────────↓────────────┘
              ... 36 stations ...
                 ↓
        [norm] → [lm_head scoring] → 151,936 logits
```

- **Residual stream**: the "conveyor belt" running end to end — one 4096-dim vector per token. Each station (block) doesn't replace it; it computes a correction and **adds** it: $h \leftarrow h + \text{station output}$. This add-only design (the **residual connection**) explains many phenomena in the series — e.g. errors accumulate along the belt.
- Each block has two parts, totaling **7 large matrices** (each is one linear map):
  - **Attention** lets each position look back at the context. Four matrices: **q** (query: what am I looking for), **k** (key: what each position offers), **v** (value: the content actually fetched), **o** (output: reorganize and write back). q and k only decide *where to look* (their dot products go through softmax into attention weights); v and o *carry the content*. The series calls the former the **pattern pathway** and the latter the **content pathway**.
  - **MLP**: a two-layer nonlinear map. Three matrices: **gate** and **up** lift 4096 → 12288 dims (the gate path passes a nonlinearity and is **elementwise-multiplied** with the up path — the SwiGLU structure; that multiplication recurs throughout the series), then **down** projects back to 4096.
- **RMSNorm**: the normalization at each part's entrance — divide the vector by its own root-mean-square. Note it is a **division**, with a denominator dominated by the largest components; this detail becomes important in parts 4-5.
- **lm_head**: the final 4096×151936 matrix mapping the internal vector to a score per vocabulary token.

36 blocks × 7 matrices = 252 large matrices — the bulk of all parameters. **Compressing them is compressing the model.**

### 4. Low-Rank Compression: Two Thin Matrices for One Fat One

A 4096×4096 matrix $W$ has 16.8M parameters. **Low-rank approximation**: find a 4096×384 matrix $A$ and a 384×4096 matrix $B$, and use $AB$ instead. Parameter count: $384\times(4096+4096) = 3.15$M — **81% saved**. The number 384 is the **rank** of the approximation.

The cost: $AB$ has rank at most 384 while $W$ can have rank 4096 — if $W$ genuinely uses that many independent directions, something must be lost. The tool for losing the least is the **singular value decomposition** (SVD): any matrix can be written $W = \sum_i \sigma_i u_i v_i^T$, a sum of rank-1 pieces with weights $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ (the **singular values** — the strengths of $W$'s action along independent directions; the generalization of eigenvalues to non-square matrices). A classical theorem (Eckart–Young): **keeping the first $r$ terms is the optimal rank-$r$ approximation**, and the error is exactly the sum of squares of the discarded singular values.

When the series says a matrix "keeps 66% of its energy at rank 384", it means the first 384 squared singular values are 66% of the total — the other 34% is simply thrown away. Matrices with fast-decaying spectra compress well; those spreading information across thousands of directions hurt no matter what — the central tension of the series' second half.

### 5. Teacher and Student: Two Routes to Compression

Two fixed roles: the **teacher** = the original model (8B params, loss 2.11); the **student** = the compressed one. Making the student imitate the teacher is called **distillation**. There are two routes, and their difference underpins the whole series:

| | Training route | Closed-form route |
|---|---|---|
| How | gradient descent: guess, compute loss, nudge parameters, repeat thousands of times | write "what the student should be" as a math problem and **solve it directly** |
| Cost | one GPU for weeks | one GPU for hours |
| Quality | better (below loss 3) | has a limit (the series maps that limit) |

An analogy: training is a student grinding through ten thousand exercises; closed-form is handing the student a condensed cheat sheet — cheap and fast, but no cheat sheet is perfect. The series' driving question: **how good can the cheat sheet get?** Note the routes relay rather than compete: training needs a starting point (the **initialization**, or init), and the closed-form output serves as exactly that.

Training-route vocabulary appearing in the posts: **step** (one parameter update; ours consumes ~0.5M tokens each), **warmup** (ramping the update size from 0 at the start to avoid early damage), **learning rate** (the nudge size), **checkpoint** (a mid-training save).

### 6. The Mathematical Core of Closed-Form: One Least-Squares Solve

One formula recurs throughout; here is its derivation, using only linear algebra.

Problem: a student layer receives input $x$ and we want its output to approximate a target $y$ (e.g. the teacher's output at the same position). Find the matrix $M$ minimizing the average error

$$\min_M\ \mathbb{E}\,\lVert y - Mx \rVert^2$$

This is **least-squares regression**. Setting the derivative in $M$ to zero gives

$$M^* = \Sigma_{yx}\,\Sigma_{xx}^{-1}, \qquad \Sigma_{yx} = \mathbb{E}[yx^T],\ \ \Sigma_{xx} = \mathbb{E}[xx^T]$$

$\Sigma_{xx}$ is the input **covariance matrix**; $\Sigma_{yx}$ the cross-covariance. In practice $\Sigma_{xx}$ can be near-singular, so a small $\lambda I$ is added before inverting — the stabilization known as **ridge regression**.

Two extensions:

- **Calibration data**: the expectations $\mathbb{E}[\cdot]$ are estimated by averaging over a small batch of real text — the only data the closed-form method ever "sees". Its quantity and diversity decide how well the covariances are estimated (part 5 devotes a chapter to this).
- **Whitening**: to truncate optimally under the input's true distribution, factor $\Sigma_{xx} = LL^T$ (Cholesky decomposition — a change of coordinates making the input distribution uniform in all directions), do the SVD truncation there, then map back. That three-step is the series' "whitened SVD truncation".

### 7. Three Diagnostic Tools

**R² (coefficient of determination)**: how much of $y$ is linearly predictable from $x$, from 0 to 1. R²=0.3 means linear regression explains 30% of the variation; the other 70% is beyond any linear tool. The series uses it to ask "how much of the compressed model's drift can be repaired linearly".

**Oracle experiments**: an oracle here means a **deliberately-cheating control experiment**. To learn "how good could the model be if every layer received perfect inputs", swap each student layer's input for the teacher's at runtime (impossible in deployment — hence cheating) and measure the loss: that is the **theoretical limit** of a whole family of methods, telling us whether a direction is worth further investment.

**Gradient sensitivity (Fisher information)**: which parameters matter? Average the squared gradient of the loss with respect to each over many samples. Intuition: if wiggling a parameter moves the loss a lot, it matters. The series uses this to decide which matrices deserve more rank.

### 8. Glossary

| Term | One-line explanation |
|---|---|
| batch | a group of text processed together (here: one 2048-token passage) |
| shard | a storage chunk of the dataset; texts within a shard are similar in origin, so mixing shards adds diversity |
| seed | random-number seed; fixing it makes "random" runs reproducible |
| KL divergence | a measure of how far apart two probability distributions are; 0 = identical |
| PPL (perplexity) | the loss in another guise: $\text{PPL} = e^{\text{loss}}$, common in the literature |
| parameter budget / equal budget | comparisons at a fixed total parameter count (ours: 2.29B), preventing "win by being bigger" |
| erank (effective rank) | how many independent directions a set of vectors really spans: ≈1 if they hug a line, ≈dimension if spread out |
| massive activations | the phenomenon of a few fixed channels being tens-to-hundreds of times larger than the rest — a recent literature topic |
| superposition | networks packing far more "features" than dimensions into one space, at the cost of non-orthogonal feature directions |
| distillation init | closed-form student parameters used as the starting point for training |
| in-sample / held-out | measured on data used for fitting vs on unseen data; the former flatters, the latter counts |
| ablation | removing components one at a time to attribute each one's contribution |
| calibration data | the small batch of text used to estimate statistics (Section 6) |
| corrector | series coinage: a small transform inserted somewhere in the model to pull drifted intermediate results back toward the teacher's trajectory |
| truncation tax | series coinage: the unavoidable loss increase from forcing matrices to a fixed rank (the "34% of energy thrown away" of Section 4) |

One final reading note: every experimental result in the series is reported as **two numbers** (e.g. 4.74 / 4.53) — losses on two disjoint stretches of test text, i.e. two independent replications. An improvement counts only when both move together; one-sided gains are usually luck.



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
