---
title: "Low-Rank Compression Series (0): A Primer — Everything You Need to Read This Series"
date: 2026-08-30 08:00:00
mathjax: true
sticky: 60
tags: [primer, LLM, compression, linear-algebra, tutorial]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 低秩压缩系列（〇）：预备知识——读懂本系列需要的一切

这个系列讲的是一件事：**把一个 80 亿参数的语言模型压缩成 23 亿参数，并且尽量不让它变笨**。系列正文假设读者熟悉一些机器学习词汇，这一篇把需要的背景一次性补齐——只要你学过高等数学和线性代数，读完这篇就能顺畅读完全系列。已经熟悉语言模型的读者可以直接跳到正文，遇到不认识的词再回来查第 9 节的速查表。

### 0. 为什么做这件事：本系列的目的与动机

训练一个能力不错的小语言模型，标准做法是从随机初始化开始，烧掉海量数据和 GPU 时间。但世界上已经有很多训练好的大模型，它们的权重里存着花大代价学到的知识。一个自然的问题是：**能不能把大模型"改造"成小模型，把这份知识尽量带过去，让小模型不必从零学起？**

本系列选择的改造手段是**低秩分解**：把大模型里的每个大矩阵换成两个瘦矩阵的乘积（第 5 节详细解释）。市面上更流行的压缩手段是**量化**——把每个数字的存储精度从 16 位压到 4 位。选低秩而不是量化，原因在目的：量化只是把**同一个模型**存得更省，结构没变，也没法在量化后的模型上继续训练；而低秩分解产生的是一个**结构上真正更小的新模型**，参数量通过 rank（保留多少个方向）自由调节，想要多小就有多小，并且可以正常继续训练。换句话说，我们做的不是"部署时省显存"，而是**给小模型的预训练制造一个好起点**。

由此拆出本系列的两个研究问题：

1. **不做任何训练，改造能做到多好？** 只靠解方程（所谓闭式方法）从大模型构造小模型，极限在哪里、被什么卡住。这是第一到五篇的主线；途中发现的"表示坍缩""val loss 假象"等现象本身也有独立价值。
2. **一个好的起点对后续训练值多少？** 同样的训练预算，从闭式改造的模型出发和从随机初始化出发，差距有多大、能维持多久。第三篇给出第一个答案，后续文章会沿着这条线走向更小的 rank 和 scaling law。

Qwen3-8B → 2.29B 只是这个问题的第一个实验场，不是终点。

### 1. 语言模型在做什么：预测下一个词

语言模型的任务简单得出乎意料：**给定前文，猜下一个词**。

输入"今天天气真"，模型输出一张概率表："好" 35%、"不错" 20%、"冷" 8%……模型的全部能力——写文章、答题、写代码——都是把这个"猜下一个词"连续做几千次的结果。

两个词汇：

- **token**：模型处理文本的最小单位。不完全等于"词"——英文里常见词是一个 token，长词会被拆成几段；中文常见字大约一字一个 token。每个模型有自己固定的 token 总表，叫**词表**（vocabulary）；词表大小因模型而异，本系列压缩的对象是阿里开源的 **Qwen3-8B**，它的词表有 **151936** 种 token——后文所有出现 151936 的地方都是这个来历。
- **logits**：模型对每种 token 打的原始分数（151936 个实数），经过 **softmax** 函数（$p\_i = e^{z\_i}/\sum\_j e^{z\_j}$，把任意实数组变成总和为 1 的概率）变成上面那张概率表。

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

### 3. 实验设置（读数字前必看）

系列所有数字都在同一套设置下产生，这里一次讲清楚：

| 项目 | 设置 |
|---|---|
| 被压缩的模型（教师） | Qwen3-8B-Base，80 亿参数，36 层，隐藏维 4096，词表 151936 |
| 压缩后（学生） | 22.9 亿参数：252 个大矩阵全部换成低秩因子，embedding 与 lm_head 保持原样 |
| 数据 | FineWeb-Edu（网页教育类文本）的 100B token 子集，用 Qwen3 分词器预先切好 |
| 校准数据 | 从训练集分片中取若干段（后期用到 512 段 × 8192 token ≈ 420 万 token），闭式方法只用它估计统计量，不做梯度更新 |
| 测试数据 | 独立的验证分割，与校准数据无重叠 |
| 评测口径 | 每段文本逐位置预测下一个 token，累加交叉熵后按 token 数平均 |

评测口径：我们从验证分割中取 800 段（每段 8192 token，合计约 660 万 token），逐位置计算下一个 token 的交叉熵后按 token 平均，得到一个数值；同时把 800 段分成 8 折分别评测，用折间标准差衡量抽样波动（典型值 ±0.002 量级，远小于本系列关心的最小改进 0.02）。一次评测约 9 分钟。系列早期曾用 40 段并分成两个窗口分别报告，所以早期文章里会看到成对出现的两个数值；后来发现评测成本远低于当初的估计，已统一改为上述口径。





### 4. 模型内部长什么样：一条传送带和 36 个加工站

本系列的模型 Qwen3-8B 属于 Transformer 家族（更具体地说是只含解码器的那一类，GPT 系列、Llama 系列都是同一家族）。下面这幅图画的就是 Qwen3-8B 的实际配置——36 层、隐藏维 4096、注意力用 GQA（32 个查询头共享 8 组键值头）、MLP 用 SwiGLU、归一化用 RMSNorm。不同模型在层数、维度、注意力细节上各有不同，但这个骨架是共通的：

<svg viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg" style="max-width:480px;display:block;margin:1em auto;font-family:system-ui,sans-serif">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#666"/>
    </marker>
  </defs>
  <!-- 输入与嵌入 -->
  <rect x="85" y="12" width="130" height="32" rx="6" fill="#eef3fb" stroke="#7a9cc6"/>
  <text x="150" y="33" text-anchor="middle" font-size="14" fill="#333">输入 token</text>
  <line x1="150" y1="44" x2="150" y2="66" stroke="#666" stroke-width="1.5" marker-end="url(#ar)"/>
  <rect x="85" y="68" width="130" height="32" rx="6" fill="#eef3fb" stroke="#7a9cc6"/>
  <text x="150" y="89" text-anchor="middle" font-size="14" fill="#333">嵌入层</text>
  <line x1="150" y1="100" x2="150" y2="128" stroke="#666" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="162" y="119" font-size="12" fill="#888">4096 维向量</text>
  <!-- 重复容器 -->
  <rect x="45" y="130" width="420" height="200" rx="8" fill="none" stroke="#bbb" stroke-dasharray="6 4"/>
  <text x="455" y="150" text-anchor="end" font-size="13" fill="#888">加工站（block）×36</text>
  <!-- 主线（残差流） -->
  <line x1="150" y1="130" x2="150" y2="330" stroke="#666" stroke-width="2"/>
  <text x="60" y="238" font-size="12" fill="#888">残差流（主线）</text>
  <!-- 注意力分支 -->
  <line x1="150" y1="170" x2="268" y2="170" stroke="#666" stroke-width="1.5" marker-end="url(#ar)"/>
  <rect x="270" y="153" width="150" height="34" rx="6" fill="#fdf1e7" stroke="#d99a5b"/>
  <text x="345" y="175" text-anchor="middle" font-size="14" fill="#333">注意力（q k v o）</text>
  <polyline points="345,187 345,207 172,207" fill="none" stroke="#666" stroke-width="1.5" marker-end="url(#ar)"/>
  <circle cx="150" cy="207" r="10" fill="#fff" stroke="#666" stroke-width="1.5"/>
  <text x="150" y="212" text-anchor="middle" font-size="14" fill="#333">+</text>
  <!-- MLP 分支 -->
  <line x1="150" y1="245" x2="268" y2="245" stroke="#666" stroke-width="1.5" marker-end="url(#ar)"/>
  <rect x="270" y="228" width="150" height="34" rx="6" fill="#fdf1e7" stroke="#d99a5b"/>
  <text x="345" y="250" text-anchor="middle" font-size="14" fill="#333">MLP（gate up down）</text>
  <polyline points="345,262 345,282 172,282" fill="none" stroke="#666" stroke-width="1.5" marker-end="url(#ar)"/>
  <circle cx="150" cy="282" r="10" fill="#fff" stroke="#666" stroke-width="1.5"/>
  <text x="150" y="287" text-anchor="middle" font-size="14" fill="#333">+</text>
  <text x="175" y="316" font-size="12" fill="#888">每部分的结果"加"回主线</text>
  <!-- 出口 -->
  <line x1="150" y1="330" x2="150" y2="352" stroke="#666" stroke-width="1.5" marker-end="url(#ar)"/>
  <rect x="85" y="354" width="130" height="32" rx="6" fill="#eef3fb" stroke="#7a9cc6"/>
  <text x="150" y="375" text-anchor="middle" font-size="14" fill="#333">归一化</text>
  <line x1="150" y1="386" x2="150" y2="408" stroke="#666" stroke-width="1.5" marker-end="url(#ar)"/>
  <rect x="85" y="410" width="130" height="32" rx="6" fill="#eaf6ee" stroke="#6fae85"/>
  <text x="150" y="431" text-anchor="middle" font-size="14" fill="#333">lm_head 打分</text>
  <line x1="150" y1="442" x2="150" y2="464" stroke="#666" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="150" y="484" text-anchor="middle" font-size="13" fill="#333">151936 个 logits（词表中每个 token 一个分数）</text>
</svg>

- **残差流（residual stream）**：贯穿全程的那条"主线"。每个 token 对应一个 4096 维向量，从头传到尾；每个加工站（block）不是替换它，而是算出一个修正量**加上去**：$h \leftarrow h + \text{本站的输出}$。这个"只做加法"的设计叫**残差连接**，是理解系列中很多现象的关键——比如误差会沿主线一路累积。
- 每个 block 里有两个部分，共 **7 个大矩阵**（每个矩阵就是一次线性变换）：
  - **注意力（attention）**：让每个位置"回头看"前文。四个矩阵各司其职——**q**（query，我在找什么）、**k**（key，每个位置能提供什么）、**v**（value，实际取回的内容）、**o**（output，把取回的内容整理后写回主线）。q 和 k 只负责算"该看哪里"（它们的点积经 softmax 变成注意力权重），v 和 o 负责"搬运内容"——系列正文把前者叫**模式通路**、后者叫**内容通路**，区别就在这里。
  - **MLP**（多层感知机；Transformer 文献里也常叫 FFN，前馈网络——同一个组件的两种名字，本系列跟随代码命名用 MLP）：一个两层的非线性变换。三个矩阵：**gate** 和 **up** 把 4096 维升到 12288 维（gate 那一路过一个非线性函数后与 up 那一路**逐元素相乘**——这个结构叫 SwiGLU，那个乘法在系列里反复出场），**down** 再降回 4096 维写回主线。
- **RMSNorm**：每个部分入口处的归一化层，把向量除以它自身的均方根（root mean square），防止数值越滚越大。注意它是个**除法**——分母由向量里幅度最大的那些分量主导，这个细节在系列第 4、5 篇会变得重要。
- **lm_head**：最后一站，一个 4096×151936 的矩阵，把 4096 维向量变成词表上每个 token 的得分。

36 个 block × 7 个矩阵 = 252 个大矩阵，占了模型参数的绝大部分。**压缩它们就是压缩模型。**

### 5. 低秩压缩：用两个瘦矩阵代替一个胖矩阵

一个 4096×4096 的矩阵 $W$ 有 1680 万个参数。**低秩近似**的想法：找一个 4096×384 的 $A$ 和一个 384×4096 的 $B$，用乘积 $AB$ 代替 $W$。参数量变成 $384\times(4096+4096) = 315$ 万——**省了 81%**。数字 384 叫这个近似的**秩**（rank）：$AB$ 作为矩阵，秩最多是 384。

代价是什么？$AB$ 的秩最多 384，而 $W$ 的秩可以高达 4096——如果 $W$ 里的信息真需要那么多独立方向，压缩必然丢东西。丢多少、怎么丢得最少，工具是**奇异值分解**（SVD）：任何矩阵都能写成 $W = \sum\_i \sigma\_i u\_i v\_i^T$，即一串秩 1 矩阵之和，权重 $\sigma\_1 \ge \sigma\_2 \ge \cdots \ge 0$ 叫**奇异值**（可以理解为"$W$ 在各个独立方向上的作用强度"，是特征值概念对非方阵的推广）。有一条经典定理（Eckart–Young）：**只保留前 $r$ 项就是最优的秩 $r$ 近似**，丢掉的误差恰好是被扔掉的奇异值的平方和。

系列里常说的"某矩阵保留了 66% 的能量"，意思就是前 384 个奇异值的平方和占全部平方和的 66%——剩下 34% 的信息被硬生生扔掉了。奇异值衰减快的矩阵（信息集中在少数方向）适合低秩压缩；衰减慢的（信息摊在几千个方向上）怎么压都疼，这正是系列后半的核心矛盾。

### 6. 老师和学生：两条压缩路线

压缩里有两个固定角色：**教师**（teacher）= 压缩前的原模型（80 亿参数，loss 2.11）；**学生**（student）= 压缩后的小模型。让学生模仿教师的过程叫**蒸馏**（distillation）。

造出一个好学生有两条路线，理解它们的区别是读懂全系列的前提：

| | 训练路线 | 闭式路线（closed-form） |
|---|---|---|
| 做法 | 梯度下降：让学生猜词、算 loss、按梯度微调参数，重复几千次 | 解方程：把"学生该是什么样"写成数学问题，**直接解出答案** |
| 成本 | 一块 GPU 跑几十天 | 一块 GPU 跑几小时 |
| 效果 | 更好（能到 loss 3 以下） | 有极限（系列就是在探这个极限） |

打个比方：训练像让学生刷一万道题慢慢开窍；闭式像直接给学生抄一份浓缩笔记——便宜、快，但笔记浓缩得再好也有上限。系列的主线问题就是：**这份"笔记"最好能做到什么程度？** 顺带说明：训练路线也需要一个起点（参数初始值，称为**初始化**或 init），闭式方法的产物正好可以当训练的起点用——所以两条路线是接力而非对立。

还有几个训练路线的词会在正文出现：**step**（一次参数更新，我们每步用约 52 万 token 的数据）、**warmup**（训练开头先把更新步长从 0 缓慢升到设定值，防止一开始就把参数改坏）、**学习率**（learning rate，每步微调的幅度）、**checkpoint**（训练中途存档）。

### 7. 闭式方法的数学核心：一次最小二乘

系列里反复出现一个公式，这里推一遍——只用到线性代数。

问题：学生某一层收到输入向量 $x$，我们希望这一层的输出尽量接近某个目标 $y$。找一个矩阵 $M$，最小化在大量样本上的平均误差：

$$\min\_M\ \mathbb{E} \lVert y - Mx \rVert^2$$

这就是**最小二乘回归**。对 $M$ 求导置零，得到解（一行推导，展开平方、逐项求导即可）：

$$M^\* = \Sigma\_{yx} \Sigma\_{xx}^{-1}, \qquad \Sigma\_{yx} = \mathbb{E}[yx^T],\ \ \Sigma\_{xx} = \mathbb{E}[xx^T]$$

$\Sigma\_{xx}$ 叫 $x$ 的**协方差矩阵**（描述输入在各方向上的分布强度），$\Sigma\_{yx}$ 是**互协方差**。实际中 $\Sigma\_{xx}$ 可能接近奇异（不可逆），所以给对角线加一个小量 $\lambda I$ 再求逆——这个稳定化技巧叫**岭回归**（ridge regression），$\lambda$ 是它唯一的超参数。

这里有一个关键的细节，它决定了整个方法的成败：**目标 $y$ 取什么？** 一个自然但错误的选择是"教师这一层在同样输入上的输出"，即 $y = Wx$（$W$ 是教师的原权重）——那样解出来的 $M^\*$ 恰好就是 $W$ 本身，什么信息也没多出来。

实际的选择是：自变量取**学生实际收到的输入** $x\_s$（它已经被前面各层的压缩误差污染），目标取**教师在它自己的干净输入 $x\_t$ 上的输出** $W x\_t$。这时解变成

$$M^\* = W \Sigma\_{ts}\Sigma\_{ss}^{-1} \ne W, \qquad \Sigma\_{ts} = \mathbb{E}[x\_t x\_s^T],\ \Sigma\_{ss} = \mathbb{E}[x\_s x\_s^T]$$

多出来的因子 $\Sigma\_{ts}\Sigma\_{ss}^{-1}$ 是"从被污染的输入线性还原干净输入"的最优算子。所以 $M^\*$ 同时干两件事：**先纠正上游累积的误差，再做教师那一层原本的变换**。只有当学生输入没有漂移（$x\_s = x\_t$）时这个因子才退化成单位矩阵、$M^\*$ 才回到 $W$。系列第 2 篇标题里的"轨迹矫正"就是指这件事，它也是把结果从 8.5 推进到 5.59 的关键——低秩结构完全没变，只换了回归的目标。

但这个 $M^\*$ 是**满秩**的（4096×4096），并没有省参数——它只是"这一层最好能做成什么样"的答案。低秩约束在第二步进来：把 $M^\*$ 压成两个瘦矩阵的乘积，即在

$$\min\_{A,B}\ \mathbb{E} \lVert M^\* x - AB x \rVert^2, \qquad A \in \mathbb{R}^{4096\times 384},\ B \in \mathbb{R}^{384\times 4096}$$

的意义下找最优的 $A, B$。注意这里的误差是在**输入 $x$ 的真实分布下**衡量的，而不是直接比较两个矩阵的元素差——同样大小的矩阵误差，如果落在 $x$ 几乎不出现的方向上就无关紧要。把这个分布因素折进 SVD 的做法就是下面的白化。

所以完整的两步是：**先解回归得到满秩的目标 $M^\*$，再把 $M^\*$ 低秩化得到实际部署的 $A, B$。** 系列各篇改进的其实是这两步里的不同环节——改回归目标（第 2 篇）、改低秩化时的度量（第 5 篇）、给 $AB$ 之外再加一个稀疏项（第 5 篇），等等。

也许你会问：为什么要绕这两步，不直接对 $A, B$ 做优化？因为**这里的两步法给出的就是全局最优解，而且是可证的**。关键在于目标函数可以按 $M^\*$ 干净地拆开（交叉项为零）：

$$\mathbb{E}\lVert y - ABx \rVert^2 = \underbrace{\mathbb{E}\lVert y - M^\*x \rVert^2}\_{\text{与 } A,B \text{ 无关的常数}} + \underbrace{\mathbb{E}\lVert (M^\*-AB)x \rVert^2}\_{\text{只有这项需要优化}}$$

交叉项之所以为零，是因为 $M^\*$ 具有**正交投影**的性质：$M^\*x$ 是 $y$ 在"$x$ 的所有线性函数"构成的子空间上的投影，所以残差 $y-M^\*x$ 与任何形如 $Cx$ 的量都不相关。这里要注意区分两件事：$M^\*$ 本身**不是**带秩约束问题的解（它是满秩的），它的作用是把"带噪声的回归问题"等价改写成"用秩 384 矩阵逼近一个确定矩阵 $M^\*$"——改写之后才轮到 Eckart–Young 出场。第一项是低秩与否都消不掉的固有残差；第二项在白化坐标下就是 $\lVert (M^\*-AB)L\rVert\_F^2$，而"用秩 $r$ 矩阵在 F 范数下最好地逼近给定矩阵"正是上一节 Eckart–Young 定理的场景——SVD 截断直接给出全局最优。相比之下，用梯度下降直接优化 $A, B$ 只能得到局部最优（$A \to AR,\ B \to R^{-1}B$ 的旋转冗余还让优化面上出现平坦方向），而且慢得多。

这个"两步等于全局最优"只在**单层、二次损失**下成立。一旦目标换成整个模型 36 层复合后的最终 loss，闭式解就不存在了——这正是训练路线能超过闭式路线的根本原因。同理，系列后期在 $AB$ 之外再加稀疏项时也破坏了这个结构，那里只能改用交替求解，不再有全局最优的保证。

**与量化方法的关系。** 压缩 LLM 的另一大流派是量化（把权重从 16 位浮点降到 4 位整数），它同样要在校准数据上做逐层拟合，思路和上面很像，但目标的选择停在了不同的地方：

- **AWQ、SmoothQuant** 一类：用激活统计判断哪些通道重要（据此缩放或分配精度），但拟合目标始终是**原权重 $W$ 本身**——它们要的是"量化后的权重接近原权重"，激活只用来加权。
- **GPTQ** 一类：顺序处理各层，拟合时用的是**量化后模型的真实输入**（因此吸收了一部分上游误差），但目标是"教师权重作用在这个脏输入上"的输出 $W x_s$。
- **本系列**：输入同样取脏的 $x_s$，但目标取教师在自己**干净**输入上的输出 $W x_t$。

差别只在最后一个下标，但在极端压缩下结果差距很大——我们实测过三种目标：按漂移分布加权逼近 $W$ 得到 19.40，匹配 $W x_s$（顺着漂移）得到 12.22，匹配 $W x_t$（纠正漂移）得到 5.59。

为什么量化界不太需要这一步：4 位量化的权重误差远小于丢掉 80% 秩的误差，漂移轻微时两种目标几乎等价。压缩越极端，"顺着漂移"和"纠正漂移"的分野才越致命。反过来说，量化领域成熟的那些技巧（用激活重要性加权、逐层顺序补偿）本系列也都用上了，第 5 篇的度量修正与它们思路相通。


两个引申：

- **校准数据（calibration data）**：上面的期望 $\mathbb{E}[\cdot]$ 在实践中用一小批真实文本的样本平均来估计。这批文本就叫校准数据——它是闭式方法唯一"见过"的数据，它的数量和多样性直接决定协方差估得准不准（第 5 篇整整一章在讲这件事）。
- **白化（whitening）**：一次坐标变换，作用是让"衡量误差"变得公平。

  为什么需要它：低秩截断必须丢掉一些方向，而输入 $x$ 的分布是极不均匀的——某些方向上 $x$ 的取值大且常见，某些方向几乎不出现。在后者上犯多大误差都无所谓（乘上去的分量本来接近零），所以不能按"矩阵元素差"来判断丢哪些方向。

  怎么做：$x$ 的协方差矩阵 $\Sigma\_{xx}$ 恰好描述了各方向的活跃程度。对它做 Cholesky 分解 $\Sigma\_{xx}=LL^T$（把对称正定矩阵写成一个三角矩阵与其转置的乘积），再令 $z = L^{-1}x$，则 $z$ 的协方差变成单位矩阵——各方向方差都是 1、互不相关，这就是"白"（借自白噪声：能量在所有方向上均匀分布）。

  为什么这样就对了：在新坐标下，我们真正关心的误差变成一个普通的矩阵范数

  $$\mathbb{E}\lVert (M-AB)x \rVert^2 = \lVert (M-AB)L \rVert\_F^2$$

  于是 Eckart–Young 定理可以直接用在 $(M-AB)L$ 上，SVD 截断即最优。**"在真实输入分布下最优"经过白化变成了"在普通矩阵范数下最优"，后者有解析解。** 算完再乘回 $L^{-1}$ 换回原坐标，得到实际部署的 $A, B$。系列里说的"白化 SVD 截断"就是这三步。

  一个类比：比较两个城市的治安不能直接比案件数，要先除以人口变成发案率。白化就是矩阵版的"除以人口"，先把各方向的重要性归一化，再比较误差。

### 8. 两个常用的诊断指标

系列里反复使用两个诊断指标，先认识它们：

**R²（决定系数）**：衡量"$y$ 能被 $x$ 线性预测的程度"，取值 0 到 1。R²=1 表示完美线性关系；R²=0.3 表示线性回归只能解释 30% 的变化，剩下 70% 是线性工具够不着的。

  系列里它专门回答一个问题：**压缩造成的偏差，还有多少能用线性手段修回来？** 做法是拿学生某处的中间结果去线性预测教师同一处的中间结果，R² 就是"可修比例"。这个数字的用处很实际：loss 只告诉你结果变差了，R² 告诉你**下一步还该不该用线性工具**。正文里三次靠它做决定——发现中段 MLP 处 R² 只有 0.25~0.40（于是判断剩余偏差主要是非线性的）、定位到某一层把 R² 从 0.82 砍到 0.27（于是找到问题层）、验证新加的矫正器把该处 R² 从 0.36 提到 0.66（于是确认机制判断正确）。

**梯度敏感度（Fisher 信息）**：想知道模型里哪个参数重要、哪个可有可无？一个通用办法：算 loss 对它的导数（梯度）在很多样本上的平方平均。直觉：如果轻轻动一下某参数 loss 就剧烈变化（梯度大），它就重要；反之则不重要。这个"梯度平方的平均"叫 Fisher 信息，系列里用它来决定"哪些矩阵该多分一点秩"。

### 9. 术语速查表

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
| 校准数据 | 闭式方法用来估计统计量的那一小批文本（见第 7 节） |
| 矫正器（corrector） | 系列自创词：插在模型某处的一个小变换，负责把跑偏的中间结果拉回教师的轨迹 |
| 截断税 | 系列自创词：把矩阵压到固定秩时不可避免的那部分 loss 损失（见第 5 节的"扔掉 34% 能量"） |

</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Low-Rank Compression Series (0): A Primer — Everything You Need to Read This Series

This series is about one thing: **compressing an 8-billion-parameter language model down to 2.3 billion parameters without making it much dumber**. The main posts assume some machine-learning vocabulary; this page fills in all of it at once. If you know calculus and linear algebra, reading this primer should be enough to follow the whole series. If you already know language models, skip ahead and come back to the glossary (Section 9) as needed.

### 0. Why We Are Doing This: Purpose and Motivation

The standard way to obtain a capable small language model is to train one from random initialization, burning enormous amounts of data and GPU time. Yet the world is full of already-trained large models whose weights store knowledge acquired at great expense. A natural question: **can we "remodel" a large model into a small one, carrying as much of that knowledge over as possible, so the small model does not start from zero?**

Our remodeling tool is **low-rank factorization**: replace every large matrix in the model with a product of two thin matrices (Section 5 explains how). The more popular compression tool is **quantization** — storing each number in 4 bits instead of 16. We choose low-rank over quantization because of the goal: quantization stores **the same model** more cheaply, its structure unchanged, and you cannot continue training a quantized model; low-rank factorization produces a **structurally smaller new model** whose parameter count is freely adjustable through the rank (how many directions to keep) — as small as you want — and which trains normally. In other words, this is not about saving GPU memory at deployment; it is about **manufacturing a good starting point for pretraining small models**.

This splits into the series' two research questions:

1. **With no training at all, how good can the remodeling get?** Building the small model from the large one purely by solving equations (so-called closed-form methods) — where is the limit, and what blocks it? This is the main thread of parts 1–5; the phenomena discovered along the way ("representation collapse," "the val-loss illusion") have standalone value.
2. **How much is a good starting point worth to subsequent training?** Given the same training budget, how large is the gap between starting from the closed-form remodel and starting from random initialization — and how long does it persist? Part 3 gives the first answer; later posts follow this thread toward smaller ranks and scaling laws.

Qwen3-8B → 2.29B is the first testbed for this question, not the destination.

### 1. What a Language Model Does: Predict the Next Word

The task is surprisingly simple: **given the text so far, guess the next word.**

Feed in "The weather today is really", and the model outputs a probability table: "nice" 35%, "cold" 8%, ... Everything a model can do — write essays, answer questions, write code — is this guess repeated thousands of times.

Two words of vocabulary:

- **token**: the smallest unit of text the model handles. Not exactly a word — common English words are one token, long words get split. Each model has its own fixed token inventory, the **vocabulary**, whose size varies by model. The model this series compresses is Alibaba's open-source **Qwen3-8B**, whose vocabulary has **151,936** tokens — every later appearance of 151,936 traces back to this.
- **logits**: the raw scores the model assigns to every token (151,936 real numbers), turned into the probability table by the **softmax** function ($p\_i = e^{z\_i}/\sum\_j e^{z\_j}$, which maps any real vector to probabilities summing to 1).

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

### 3. Experimental Setup (read before the numbers)

Every number in the series comes from one setup, stated here once:

| Item | Setting |
|---|---|
| Compressed model (teacher) | Qwen3-8B-Base: 8B params, 36 layers, hidden dim 4096, vocabulary 151,936 |
| After compression (student) | 2.29B params: all 252 large matrices replaced by low-rank factors; embedding and lm_head left intact |
| Data | FineWeb-Edu (educational web text), the 100B-token subset, pre-tokenized with the Qwen3 tokenizer |
| Calibration data | passages drawn from training shards (up to 512 × 8192 tokens ≈ 4.2M tokens in later work); used only to estimate statistics, never for gradient updates |
| Test data | a separate validation split, disjoint from the calibration data |
| Metric | next-token cross-entropy at every position, summed and averaged over tokens |

Evaluation protocol: we take 800 passages from the validation split (8192 tokens each, ~6.6M tokens total), compute next-token cross-entropy at every position and average over tokens for a single number; the 800 passages are also split into 8 folds so the between-fold standard deviation measures sampling spread (typically 0.002, far below the 0.02 improvements this series cares about). One evaluation takes about nine minutes. Early posts used 40 passages reported as two windows, hence the paired numbers seen there; having found evaluation far cheaper than first assumed, we standardized on the protocol above.





### 4. Inside the Model: a Conveyor Belt and 36 Stations

The model in this series, Qwen3-8B, belongs to the Transformer family (specifically the decoder-only kind, as are the GPT and Llama series). The diagram below depicts Qwen3-8B's actual configuration: 36 layers, hidden dim 4096, GQA attention (32 query heads sharing 8 key-value groups), SwiGLU MLPs, RMSNorm normalization. Models differ in depth, width and attention details, but this skeleton is common to all:

<svg viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg" style="max-width:480px;display:block;margin:1em auto;font-family:system-ui,sans-serif">
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#666"/>
    </marker>
  </defs>
  <rect x="85" y="12" width="130" height="32" rx="6" fill="#eef3fb" stroke="#7a9cc6"/>
  <text x="150" y="33" text-anchor="middle" font-size="14" fill="#333">input token</text>
  <line x1="150" y1="44" x2="150" y2="66" stroke="#666" stroke-width="1.5" marker-end="url(#ar2)"/>
  <rect x="85" y="68" width="130" height="32" rx="6" fill="#eef3fb" stroke="#7a9cc6"/>
  <text x="150" y="89" text-anchor="middle" font-size="14" fill="#333">embedding</text>
  <line x1="150" y1="100" x2="150" y2="128" stroke="#666" stroke-width="1.5" marker-end="url(#ar2)"/>
  <text x="162" y="119" font-size="12" fill="#888">4096-dim vector</text>
  <rect x="45" y="130" width="420" height="200" rx="8" fill="none" stroke="#bbb" stroke-dasharray="6 4"/>
  <text x="455" y="150" text-anchor="end" font-size="13" fill="#888">block ×36</text>
  <line x1="150" y1="130" x2="150" y2="330" stroke="#666" stroke-width="2"/>
  <text x="58" y="238" font-size="12" fill="#888">residual stream</text>
  <line x1="150" y1="170" x2="268" y2="170" stroke="#666" stroke-width="1.5" marker-end="url(#ar2)"/>
  <rect x="270" y="153" width="150" height="34" rx="6" fill="#fdf1e7" stroke="#d99a5b"/>
  <text x="345" y="175" text-anchor="middle" font-size="14" fill="#333">attention (q k v o)</text>
  <polyline points="345,187 345,207 172,207" fill="none" stroke="#666" stroke-width="1.5" marker-end="url(#ar2)"/>
  <circle cx="150" cy="207" r="10" fill="#fff" stroke="#666" stroke-width="1.5"/>
  <text x="150" y="212" text-anchor="middle" font-size="14" fill="#333">+</text>
  <line x1="150" y1="245" x2="268" y2="245" stroke="#666" stroke-width="1.5" marker-end="url(#ar2)"/>
  <rect x="270" y="228" width="150" height="34" rx="6" fill="#fdf1e7" stroke="#d99a5b"/>
  <text x="345" y="250" text-anchor="middle" font-size="14" fill="#333">MLP (gate up down)</text>
  <polyline points="345,262 345,282 172,282" fill="none" stroke="#666" stroke-width="1.5" marker-end="url(#ar2)"/>
  <circle cx="150" cy="282" r="10" fill="#fff" stroke="#666" stroke-width="1.5"/>
  <text x="150" y="287" text-anchor="middle" font-size="14" fill="#333">+</text>
  <text x="175" y="316" font-size="12" fill="#888">each part ADDS its result back</text>
  <line x1="150" y1="330" x2="150" y2="352" stroke="#666" stroke-width="1.5" marker-end="url(#ar2)"/>
  <rect x="85" y="354" width="130" height="32" rx="6" fill="#eef3fb" stroke="#7a9cc6"/>
  <text x="150" y="375" text-anchor="middle" font-size="14" fill="#333">norm</text>
  <line x1="150" y1="386" x2="150" y2="408" stroke="#666" stroke-width="1.5" marker-end="url(#ar2)"/>
  <rect x="85" y="410" width="130" height="32" rx="6" fill="#eaf6ee" stroke="#6fae85"/>
  <text x="150" y="431" text-anchor="middle" font-size="14" fill="#333">lm_head scoring</text>
  <line x1="150" y1="442" x2="150" y2="464" stroke="#666" stroke-width="1.5" marker-end="url(#ar2)"/>
  <text x="150" y="484" text-anchor="middle" font-size="13" fill="#333">151,936 logits (one score per vocabulary token)</text>
</svg>

- **Residual stream**: the "conveyor belt" running end to end — one 4096-dim vector per token. Each station (block) doesn't replace it; it computes a correction and **adds** it: $h \leftarrow h + \text{station output}$. This add-only design (the **residual connection**) explains many phenomena in the series — e.g. errors accumulate along the belt.
- Each block has two parts, totaling **7 large matrices** (each is one linear map):
  - **Attention** lets each position look back at the context. Four matrices: **q** (query: what am I looking for), **k** (key: what each position offers), **v** (value: the content actually fetched), **o** (output: reorganize and write back). q and k only decide *where to look* (their dot products go through softmax into attention weights); v and o *carry the content*. The series calls the former the **pattern pathway** and the latter the **content pathway**.
  - **MLP** (multi-layer perceptron; the Transformer literature often calls it the FFN, feed-forward network — two names for the same component; this series follows the code naming and says MLP): a two-layer nonlinear map. Three matrices: **gate** and **up** lift 4096 → 12288 dims (the gate path passes a nonlinearity and is **elementwise-multiplied** with the up path — the SwiGLU structure; that multiplication recurs throughout the series), then **down** projects back to 4096.
- **RMSNorm**: the normalization at each part's entrance — divide the vector by its own root-mean-square. Note it is a **division**, with a denominator dominated by the largest components; this detail becomes important in parts 4-5.
- **lm_head**: the final 4096×151936 matrix mapping the internal vector to a score per vocabulary token.

36 blocks × 7 matrices = 252 large matrices — the bulk of all parameters. **Compressing them is compressing the model.**

### 5. Low-Rank Compression: Two Thin Matrices for One Fat One

A 4096×4096 matrix $W$ has 16.8M parameters. **Low-rank approximation**: find a 4096×384 matrix $A$ and a 384×4096 matrix $B$, and use $AB$ instead. Parameter count: $384\times(4096+4096) = 3.15$M — **81% saved**. The number 384 is the **rank** of the approximation.

The cost: $AB$ has rank at most 384 while $W$ can have rank 4096 — if $W$ genuinely uses that many independent directions, something must be lost. The tool for losing the least is the **singular value decomposition** (SVD): any matrix can be written $W = \sum\_i \sigma\_i u\_i v\_i^T$, a sum of rank-1 pieces with weights $\sigma\_1 \ge \sigma\_2 \ge \cdots \ge 0$ (the **singular values** — the strengths of $W$'s action along independent directions; the generalization of eigenvalues to non-square matrices). A classical theorem (Eckart–Young): **keeping the first $r$ terms is the optimal rank-$r$ approximation**, and the error is exactly the sum of squares of the discarded singular values.

When the series says a matrix "keeps 66% of its energy at rank 384", it means the first 384 squared singular values are 66% of the total — the other 34% is simply thrown away. Matrices with fast-decaying spectra compress well; those spreading information across thousands of directions hurt no matter what — the central tension of the series' second half.

### 6. Teacher and Student: Two Routes to Compression

Two fixed roles: the **teacher** = the original model (8B params, loss 2.11); the **student** = the compressed one. Making the student imitate the teacher is called **distillation**. There are two routes, and their difference underpins the whole series:

| | Training route | Closed-form route |
|---|---|---|
| How | gradient descent: guess, compute loss, nudge parameters, repeat thousands of times | write "what the student should be" as a math problem and **solve it directly** |
| Cost | one GPU for weeks | one GPU for hours |
| Quality | better (below loss 3) | has a limit (the series maps that limit) |

An analogy: training is a student grinding through ten thousand exercises; closed-form is handing the student a condensed cheat sheet — cheap and fast, but no cheat sheet is perfect. The series' driving question: **how good can the cheat sheet get?** Note the routes relay rather than compete: training needs a starting point (the **initialization**, or init), and the closed-form output serves as exactly that.

Training-route vocabulary appearing in the posts: **step** (one parameter update; ours consumes ~0.5M tokens each), **warmup** (ramping the update size from 0 at the start to avoid early damage), **learning rate** (the nudge size), **checkpoint** (a mid-training save).

### 7. The Mathematical Core of Closed-Form: One Least-Squares Solve

One formula recurs throughout; here is its derivation, using only linear algebra.

Problem: a student layer receives input $x$ and we want its output to approximate a target $y$. Find the matrix $M$ minimizing the average error

$$\min\_M\ \mathbb{E} \lVert y - Mx \rVert^2$$

This is **least-squares regression**. Setting the derivative in $M$ to zero gives

$$M^\* = \Sigma\_{yx} \Sigma\_{xx}^{-1}, \qquad \Sigma\_{yx} = \mathbb{E}[yx^T],\ \ \Sigma\_{xx} = \mathbb{E}[xx^T]$$

$\Sigma\_{xx}$ is the input **covariance matrix**; $\Sigma\_{yx}$ the cross-covariance. In practice $\Sigma\_{xx}$ can be near-singular, so a small $\lambda I$ is added before inverting — the stabilization known as **ridge regression**.

One detail here decides whether the whole method works: **what is the target $y$?** A natural but wrong choice is "the teacher layer's output on the same input", $y = Wx$ with $W$ the teacher's original weight — that would make $M^\* = W$ exactly, adding no information.

The actual choice: the regressor is the input the student **actually receives**, $x\_s$ (already corrupted by upstream compression error), while the target is the teacher's output on its own clean input, $W x\_t$. The solution becomes

$$M^\* = W \Sigma\_{ts}\Sigma\_{ss}^{-1} \ne W, \qquad \Sigma\_{ts} = \mathbb{E}[x\_t x\_s^T],\ \Sigma\_{ss} = \mathbb{E}[x\_s x\_s^T]$$

The extra factor $\Sigma\_{ts}\Sigma\_{ss}^{-1}$ is the optimal linear operator recovering the clean input from the corrupted one. So $M^\*$ does two jobs at once: **undo the accumulated upstream error, then apply the transformation the teacher layer performed.** Only when the student input has not drifted ($x\_s = x\_t$) does the factor collapse to the identity and $M^\*$ return to $W$. This is what "trajectory correction" in part 2's title means, and it is what moved the result from 8.5 to 5.59 — the low-rank structure was unchanged; only the regression target was.

But this $M^\*$ is **full-rank** (4096×4096) and saves no parameters — it merely answers "what is the best this layer could be". The rank constraint enters in a second step: compress $M^\*$ into a product of two thin matrices, i.e. find the optimal $A, B$ under

$$\min\_{A,B}\ \mathbb{E} \lVert M^\* x - AB x \rVert^2, \qquad A \in \mathbb{R}^{4096\times 384},\ B \in \mathbb{R}^{384\times 4096}$$

Note the error is measured **under the true distribution of the input $x$**, not by comparing matrix entries directly — a matrix error of the same size does not matter if it lies along directions $x$ rarely takes. Folding that distributional factor into the SVD is exactly what whitening does below.

So the full procedure is two steps: **solve the regression for a full-rank target $M^\*$, then low-rank-ify $M^\*$ into the $A, B$ actually deployed.** What the posts in this series improve are different parts of these two steps — the regression target (part 2), the metric used when low-rank-ifying (part 5), adding a sparse term alongside $AB$ (part 5), and so on.

You might ask: why the detour — why not optimize $A, B$ directly? Because **the two-step procedure here yields the global optimum, provably.** The objective splits cleanly around $M^\*$ (the cross term vanishes):

$$\mathbb{E}\lVert y - ABx \rVert^2 = \underbrace{\mathbb{E}\lVert y - M^\*x \rVert^2}\_{\text{constant in } A,B} + \underbrace{\mathbb{E}\lVert (M^\*-AB)x \rVert^2}\_{\text{the only part to optimize}}$$

The cross term vanishes because $M^\*$ is an **orthogonal projection**: $M^\*x$ is the projection of $y$ onto the subspace of linear functions of $x$, so the residual $y-M^\*x$ is uncorrelated with anything of the form $Cx$. Note the distinction: $M^\*$ itself is **not** the solution to the rank-constrained problem (it is full-rank); its role is to rewrite "a noisy regression problem" equivalently as "approximate the fixed matrix $M^\*$ with a rank-384 one" — only then does Eckart–Young apply. The first term is residual no rank choice can remove; the second, in whitened coordinates, is $\lVert (M^\*-AB)L\rVert\_F^2$ — and "best rank-$r$ approximation of a given matrix in Frobenius norm" is exactly the Eckart–Young setting from the previous section, so SVD truncation is globally optimal. Gradient descent on $A, B$ directly would only find a local optimum (the rotation redundancy $A \to AR,\ B \to R^{-1}B$ also creates flat directions) and would be far slower.

This equivalence holds only for a **single layer with a quadratic loss**. Once the objective becomes the final loss after composing all 36 layers, no closed form exists — which is precisely why the training route can beat the closed-form one. Likewise, adding a sparse term alongside $AB$ later in the series breaks the structure, so that part falls back to alternating solves with no global guarantee.

**Relation to quantization methods.** The other major family of LLM compression is quantization (weights from 16-bit floats down to 4-bit integers). It also fits layer by layer on calibration data, very much like the above, but stops at a different choice of target:

- **AWQ, SmoothQuant** and kin: use activation statistics to judge which channels matter (then rescale or allocate precision), but the fitting target remains **the original weight $W$** — they want "quantized weight close to original weight", with activations only providing weights.
- **GPTQ** and kin: process layers sequentially and fit using the **quantized model's real input** (thus absorbing some upstream error), but the target is the teacher weight applied to that corrupted input, $W x_s$.
- **This series**: the input is likewise the corrupted $x_s$, but the target is the teacher's output on its own **clean** input, $W x_t$.

The difference is one subscript, yet under extreme compression the outcomes diverge sharply — we measured all three: approximating $W$ weighted by drifted statistics gives 19.40, matching $W x_s$ (following the drift) gives 12.22, matching $W x_t$ (correcting the drift) gives 5.59.

Why quantization rarely needs this: 4-bit weight error is far smaller than discarding 80% of the rank, so with mild drift the two targets are nearly equivalent. The more extreme the compression, the more decisive the split between "follow the drift" and "correct the drift". Conversely, the tricks quantization matured — activation-importance weighting, sequential layerwise compensation — are all used here too; part 5's metric fix is close kin to them.


Two extensions:

- **Calibration data**: the expectations $\mathbb{E}[\cdot]$ are estimated by averaging over a small batch of real text — the only data the closed-form method ever "sees". Its quantity and diversity decide how well the covariances are estimated (part 5 devotes a chapter to this).
- **Whitening**: a change of coordinates whose purpose is to make error measurement fair.

  Why it is needed: low-rank truncation must discard some directions, and the input distribution is highly uneven — along some directions $x$ takes large, frequent values; along others it barely appears. Error along the latter is harmless (the component multiplying it is near zero), so "difference in matrix entries" is the wrong criterion for choosing what to discard.

  How: the covariance $\Sigma\_{xx}$ describes exactly how active each direction is. Factor it as $\Sigma\_{xx}=LL^T$ (Cholesky: a symmetric positive-definite matrix written as a triangular matrix times its transpose) and set $z = L^{-1}x$; then $z$ has identity covariance — unit variance in every direction, no correlations. That is "white" (as in white noise: energy spread evenly across directions).

  Why that works: in the new coordinates the error we actually care about becomes an ordinary matrix norm

  $$\mathbb{E}\lVert (M-AB)x \rVert^2 = \lVert (M-AB)L \rVert\_F^2$$

  so Eckart–Young applies directly to $(M-AB)L$ and SVD truncation is optimal. **"Optimal under the true input distribution" becomes, after whitening, "optimal in an ordinary matrix norm" — and the latter has a closed-form solution.** Multiply back by $L^{-1}$ to return to the original coordinates and obtain the deployed $A, B$. The series' "whitened SVD truncation" is these three steps.

  An analogy: comparing two cities' safety by raw crime counts is wrong; you divide by population to get a rate. Whitening is the matrix version of dividing by population — normalize each direction's importance first, then compare errors.

### 8. Two Diagnostic Measures

**R² (coefficient of determination)**: how much of $y$ is linearly predictable from $x$, from 0 to 1. R²=1 is a perfect linear relationship; R²=0.3 means linear regression explains 30% of the variation and the other 70% is beyond any linear tool.

  In this series it answers one specific question: **how much of the deviation caused by compression can still be repaired linearly?** We take the student's intermediate values at some point and linearly predict the teacher's at the same point; R² is the "repairable fraction". Its practical use: loss only tells you the result got worse, while R² tells you **whether a linear tool is still worth trying**. It drove three decisions in the posts — finding R² of only 0.25-0.40 at the mid-network MLPs (concluding the remaining deviation is mostly nonlinear), locating a single layer that cuts R² from 0.82 to 0.27 (identifying the problem layer), and confirming a new corrector lifted that spot from 0.36 to 0.66 (validating the mechanism).

**Gradient sensitivity (Fisher information)**: which parameters matter? Average the squared gradient of the loss with respect to each over many samples. Intuition: if wiggling a parameter moves the loss a lot, it matters. The series uses this to decide which matrices deserve more rank.

### 9. Glossary

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
| calibration data | the small batch of text used to estimate statistics (Section 7) |
| corrector | series coinage: a small transform inserted somewhere in the model to pull drifted intermediate results back toward the teacher's trajectory |
| truncation tax | series coinage: the unavoidable loss increase from forcing matrices to a fixed rank (the "34% of energy thrown away" of Section 5) |

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
  var postTitles = {zh: '低秩压缩系列（〇）：预备知识——读懂本系列需要的一切', en: 'Low-Rank Compression Series (0): A Primer — Everything You Need to Read This Series'};
  var titleEl = document.querySelector('.post-title');
  if (titleEl) titleEl.textContent = postTitles[lang];
}
switchLang('zh');
</script>
