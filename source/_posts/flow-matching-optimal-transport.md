---
title: "Flow Matching and Optimal Transport"
date: 2025-06-25 10:00:00
tags:
  - flow matching
  - generative models
  - optimal transport
  - machine learning
categories:
  - machine learning
mathjax: true
---

<div class="lang-switch">
  <button id="btn-en" class="lang-btn active" onclick="switchLang('en')">English</button>
  <button id="btn-zh" class="lang-btn" onclick="switchLang('zh')">中文</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh" style="display:none;">

<!-- more -->

{% note info %}
**前情提要**：在 [The Mathematics of Flow Matching](/2025/06/12/flow-matching-three-steps/) 中我们推导了 CFM loss，在 [From Flow Matching to MeanFlow](/2025/06/18/meanflow-one-step-generation/) 中把学习对象从瞬时速度换成了平均速度。两篇都反复出现"最优传输"这个词，但都没说清：**Flow Matching 到底在什么意义上是最优传输？**

**本文回答这个问题**，并解读 OT-CFM（Tong et al., TMLR 2024）。一句话版本：

- Flow Matching 的**条件路径**是最优传输路径，但**边际路径**一般**不是**。
- 动态最优传输（Benamou-Brenier）此前只能靠模拟 ODE 去逼近，代价是训练时要穿过上百次网络求值。
- OT-CFM 的做法：把便宜的**静态** OT 解装进 CFM 的采样分布，从而**免模拟地**得到**动态** OT 的解。

**符号约定**：$q_0, q_1$ 为源/目标分布，$x_0 \sim q_0$，$x_1 \sim q_1$；$z$ 为条件变量；$u_t$ 为真实速度场，$v_t^\theta$ 为神经网络。
{% endnote %}

## 1. 两个最优传输问题：静态与动态

最优传输有两副面孔，而 Flow Matching 只与其中一副天然相容。

### 静态 OT：只问"谁配谁"

Kantorovich 形式的 $2$-Wasserstein 问题在所有**耦合**（joint distribution）上求最小代价：

$$W_2^2(q_0, q_1) = \inf\_{\pi \in \Pi(q_0, q_1)} \int \lVert x_0 - x_1 \rVert^2 \mathrm{d}\pi(x_0, x_1) \tag{1} \label{eq:static-ot-zh}$$

其中 $\Pi(q_0, q_1)$ 是边际分别为 $q_0$、$q_1$ 的所有联合分布。解 $\pi^{\ast}$ 称为 **OT plan**——它只回答"哪个 $x_0$ 应该配哪个 $x_1$"，**不涉及时间**，也不涉及任何 ODE。

它的好处是**可解**：离散情形是标准线性规划（或 Sinkhorn），成熟求解器现成可用。

### 动态 OT：问"怎么走"

Benamou-Brenier (2000) 的等价形式改为在**速度场**上求最小动能：

$$W_2^2(q_0, q_1) = \inf\_{p_t, u_t} \int_0^1 \int p_t(x) \lVert u_t(x) \rVert^2 \mathrm{d}x \mathrm{d}t \tag{2} \label{eq:dynamic-ot-zh}$$

约束是连续性方程 $\partial_t p_t = -\nabla \cdot (p_t u_t)$ 以及边界条件 $p_0 = q_0$、$p_1 = q_1$。

这才是生成模型想要的东西——它直接给出一个可以积分的速度场 $u_t$。但它是一个**关于整条路径的变分问题**，比 $\eqref{eq:static-ot-zh}$ 难得多。

{% note danger %}
**此前的困境**。要让神经 ODE 逼近动态 OT，之前的做法（TrajectoryNet、Finlay et al. 的正则化 CNF）是给 CNF 加 $L^2$ 动能正则。论文 §2.2 原文点出代价：

> "these models required **integrating over and backpropagating through tens to hundreds of function evaluations**, resulting in both numerical and efficiency issues. We aim to avoid these issues by directly regressing to the vector field in a **simulation-free** way."

也就是说：**动态 OT 难，是因为它需要模拟（simulation）**——训练每一步都要真的把 ODE 积出来再反传。

**本文的核心问题**：$\eqref{eq:static-ot-zh}$ 便宜、$\eqref{eq:dynamic-ot-zh}$ 有用，而 Benamou-Brenier 说两者等价。能否把静态解**提升（lift）**为动态解，且训练完全免模拟？
{% endnote %}

## 2. Flow Matching 里的"OT"是哪一个？

在回答上面的问题之前，先澄清一个长期的混淆——这也是读 Lipman et al. 原文时最容易误解的地方。

[前文](/2025/06/12/flow-matching-three-steps/)的构造是：取一对样本，用**直线**连起来

$$x_t = (1-t)x_0 + t x_1$$

一条直线正是从 $x_0$ 到 $x_1$ 的动态 OT 路径（两点之间动能最小的走法就是匀速直线）。所以说 FM 用了"OT 路径"，**在给定一对 $(x_0, x_1)$ 的条件下是对的**。

但 OT-CFM 论文 §3.2.1 特意强调了下一句：

> "We emphasize that although the **conditional** probability path $p_t(x \mid z)$ **is** an optimal transport path from $p_0(x \mid z)$ to $p_1(x \mid z)$, the **marginal** path $p_t(x)$ **is not in general an OT path** from the standard normal $p_0(x)$ to the data distribution $p_1(x)$."

{% note success %}
**关系澄清**（两个层次，别混淆）：

| | 条件层面 $p_t(x \mid z)$ | 边际层面 $p_t(x)$ |
|---|---|---|
| 是什么 | 单条样本连线 | 所有连线叠加后的真实概率流 |
| 是 OT 吗 | **是**（直线 = 两点间动态 OT） | **一般不是** |
| 谁在乎 | 推导的中间构造 | 采样时真正积分的那个场 |

一句话：**每条连线各自最优，叠加起来的流不最优**。因为无数条直线互相交叉，网络在交叉点处只能给出一个速度值——那是各方向的加权平均，不再是任何一条直线的速度。
{% endnote %}

论文 Table 1 把这件事做成了两列 checkbox，一目了然：

| 方法 | $q(z)$ | Cond. OT | Marginal OT |
|---|---|---|---|
| Flow Matching (Lipman et al.) | $q(x_1)$ | ✓ | ✗ |
| Rectified Flow (Liu) | $q(x_0)q(x_1)$ | ✓ | ✗ |
| Stochastic Interpolant | $q(x_0)q(x_1)$ | ✓ | ✗ |
| I-CFM（独立耦合） | $q(x_0)q(x_1)$ | ✓ | ✗ |
| **OT-CFM（本文）** | $\pi(x_0, x_1)$ | ✓ | **✓** |

**所以 OT-CFM 要补的正是右下那一格**：让边际流也成为 OT 流。注意前四行不是"错"——它们都是正确的生成模型，只是边际层面不解动态 OT。

## 3. 第一步：广义 CFM——把耦合变成一个可选项

论文的第一个贡献是一个框架，它的作用是**腾出自由度**。

### 混合视角

设边际概率路径是一族条件路径的混合：

$$p_t(x) = \int p_t(x \mid z) q(z) \mathrm{d}z \tag{3} \label{eq:mixture-zh}$$

其中 $z$ 是**任意**条件变量，$q(z)$ 是它的分布。若每条 $p_t(x \mid z)$ 由 $u_t(x \mid z)$ 生成，则生成混合路径 $p_t(x)$ 的边际速度场是加权平均：

$$u_t(x) = \mathbb{E}\_{q(z)} \left[ \frac{u_t(x \mid z)\enspace p_t(x \mid z)}{p_t(x)} \right] \tag{4} \label{eq:marginal-field-zh}$$

**Theorem 3.1**：$\eqref{eq:marginal-field-zh}$ 确实生成 $\eqref{eq:mixture-zh}$。

问题是 $\eqref{eq:marginal-field-zh}$ 算不出来——分母 $p_t(x)$ 是个积分。于是定义 **CFM loss**，只回归**条件**速度场：

$$\mathcal{L}\_{\mathrm{CFM}}(\theta) = \mathbb{E}\_{t, q(z), p_t(x \mid z)} \lVert v_t^\theta(x) - u_t(x \mid z) \rVert^2 \tag{5} \label{eq:cfm-zh}$$

**Theorem 3.2**：$\mathcal{L}\_{\mathrm{CFM}}$ 与 $\mathcal{L}\_{\mathrm{FM}}$ 相差一个与 $\theta$ 无关的常数，因此

$$\nabla\_\theta \mathcal{L}\_{\mathrm{FM}}(\theta) = \nabla\_\theta \mathcal{L}\_{\mathrm{CFM}}(\theta) \tag{6} \label{eq:grad-eq-zh}$$

这就是[前文](/2025/06/12/flow-matching-three-steps/)用勾股定理证过的那件事，这里推广到了任意 $z$。

### 关键：$q(z)$ 是自由的

取 $z := (x_0, x_1)$，条件路径为两点间的高斯桥：

$$p_t(x \mid z) = \mathcal{N}(x \mid t x_1 + (1-t) x_0, \enspace \sigma^2), \qquad u_t(x \mid z) = x_1 - x_0 \tag{7} \label{eq:icfm-zh}$$

此时 $\eqref{eq:cfm-zh}$ 就是[前文](/2025/06/12/flow-matching-three-steps/)推出的那个 loss。前文取 $q(z) = q(x_0) q(x_1)$——**独立耦合**，论文称之为 **I-CFM**。

{% note info %}
**这是框架的价值所在**：$\eqref{eq:grad-eq-zh}$ 的成立**只要求 $q(z)$ 的边际是 $q_0$ 和 $q_1$**，完全不要求 $x_0 \perp x_1$。

所以独立性不是一个"假设"，而是构造 $q(z)$ 时的一个**默认取法**——前文的推导没有任何漏洞，$Z_1 \sim q_1$ 严格成立。但它意味着 $q(z)$ 是一个此前**未被使用的自由度**。

论文的第一个贡献就是把这个自由度显式化（并顺带说明 FM 只需 $q(x_1)$、要求源为 Gaussian 的限制也随之解除，任意源分布都可以）。第二个贡献是把它用起来。
{% endnote %}

## 4. 第二步：把静态 OT plan 装进 $q(z)$

现在第 1 节的问题有了一个几乎显然的答案。

### OT-CFM

令条件变量的分布就是**静态 OT plan**：

$$\boxed{q(z) := \pi^{\ast}(x_0, x_1)} \tag{8} \label{eq:otcfm-zh}$$

即 $\eqref{eq:static-ot-zh}$ 的解。条件路径仍用 $\eqref{eq:icfm-zh}$，什么都不变——**唯一的改动是 $(x_0, x_1)$ 不再各自独立采样，而是按 $\pi^{\ast}$ 联合采样**。

{% note success %}
**Proposition 3.4**：在 $q_0$、$q_1$、$\pi^{\ast}$ 的正则性条件下，当 $\sigma^2 \to 0$ 时，$\eqref{eq:otcfm-zh}$ 诱导的边际路径 $p_t$ 与边际速度场 $u_t$ **最小化 $\eqref{eq:dynamic-ot-zh}$**，即 $u_t$ 解 $q_0$ 到 $q_1$ 的**动态最优传输问题**。

这就完成了"提升"：输入是**静态** OT 的解（一个配对），输出是**动态** OT 的解（一个速度场），而训练只是 $\eqref{eq:cfm-zh}$ 这个回归——**没有任何 ODE 模拟**。
{% endnote %}

### 为什么装进去就成立

关键在 Brenier 定理：OT plan 是确定性的且**不交叉**，$x_1 = \nabla \Psi^{\ast}(x_0)$（凸函数的梯度）。

于是第 2 节那个"条件是 OT、边际不是 OT"的裂缝被弥合了：既然直线族互不交叉，每个点 $x$ 只被唯一一条连线穿过，$\eqref{eq:marginal-field-zh}$ 的加权平均**退化成单点取值**——

$$u_t(x) = u_t(x \mid z) = \nabla \Psi^{\ast}(x_0) - x_0$$

条件场与边际场重合。既然每条条件路径都是 OT 路径，边际路径也就是 OT 路径了。McCann 位移插值 $p_t = [(1-t)\mathrm{Id} + t \nabla\Psi^{\ast}]\_{\sharp} q_0$ 正是动态 OT 的解。

{% note danger %}
**对比第 2 节**：独立耦合下连线大量交叉 → $\eqref{eq:marginal-field-zh}$ 是真的在平均 → 条件 OT 传不到边际。OT 耦合下不交叉 → 不平均 → 条件 OT 直接就是边际 OT。

**这就是那张 checkbox 表里 `Marginal OT` 一列的全部机制。**
{% endnote %}

### 实际方案：Minibatch OT

精确 OT plan 的时间复杂度是 $O(n^3)$、内存 $O(n^2)$，大数据集不可行。论文按数据规模分两种情形：

1. **小数据集**（如单细胞）：直接算精确 OT plan，网络的作用是把这个映射**外推到未见数据**。
2. **大数据集**（如图像）：**在每个 batch 内解 OT**。采样 $\{x_0^i\}\_{i=1}^B$、$\{x_1^i\}\_{i=1}^B$，解 $B \times B$ 的分配问题，用得到的配对计算 $\eqref{eq:cfm-zh}$。

{% note warning %}
**Minibatch OT 是有偏的**——它不等于全局 OT。但论文指出：

- 当 OT batch size 等于 $(q_0, q_1)$ 支撑集大小时，恢复精确 OT，由 Prop 3.4 学到精确动态 OT。
- 实测 batch size 远小于数据集即可很好地逼近 OT 映射（Fig. D.2），作者推测这来自网络优化本身的泛化。
- 训练开销 $< 1\%$。
{% endnote %}

## 5. 副产品：训练与推理都更快

论文结论段的句式是"lifting static OT to dynamic ... **while also** allowing more efficient training and inference"。以下两条是那个 "while also"——重要，但逻辑上是**推论**，不是出发点。

### 训练更快：回归目标的方差降低

$\eqref{eq:cfm-zh}$ 是个**随机**回归：同一个 $x$ 可能来自不同的 $z$，目标 $u_t(x \mid z)$ 随之抖动。论文称这个抖动为 **objective variance**：

$$\mathrm{OV} := \mathbb{E} \lVert u_t(x \mid z) - u_t(x) \rVert^2 \tag{9} \label{eq:ov-zh}$$

它正是 $\eqref{eq:grad-eq-zh}$ 里那个"与 $\theta$ 无关的常数"，也是 $\mathcal{L}\_{\mathrm{CFM}}$ 的下界——网络再强也降不到零以下。

**Proposition B.2**：若 $\pi$ 是 Monge 映射，则 $\sigma \to 0$ 时 OT-CFM 的 $\mathrm{OV} \to 0$。这是第 4 节"条件场 = 边际场"的直接推论。而独立耦合下 $\mathrm{OV}$ 不趋于零。

实测：OT-CFM 与 SB-CFM 的 OV 比 I-CFM/FM **低至少一个数量级**，相应地验证误差下降更快（Fig. 2 左、Fig. D.8）。

### 推理更快：NFE 更少

推理是数值积分 $\mathrm{d}Z_t = v_t^\theta(Z_t)\mathrm{d}t$，每次求值 $v_t^\theta$ 是一次网络前向。总求值次数记作 **NFE**（Number of Function Evaluations）——它就是推理成本。

$N$ 步 Euler 的截断误差由加速度控制：把精确解展开 $Z_{t+h} = Z_t + h\dot{Z}\_t + \frac{h^2}{2}\ddot{Z}\_t + O(h^3)$，Euler 丢掉的是 $O(h^2 \lVert \ddot{Z}\_t \rVert)$ 那一项，累积后

$$\mathrm{NFE} = N \gtrsim \frac{C}{\varepsilon} \sup\_{t} \lVert \ddot{Z}\_t \rVert \tag{10} \label{eq:nfe-zh}$$

而在 OT 耦合下轨迹是**直线**，$\ddot{Z}\_t \equiv 0$——截断误差为零，原则上一步即精确。

实测（Fig. 2 右、Fig. 3 右）：固定 NFE 时 OT-CFM 的样本质量更好；达到同等质量所需 NFE 更少。CIFAR-10 上 OT-CFM 在 FID 与 NFE 两个维度都优于 I-CFM 和 FM。

### 其他实验结论

- **动态 OT 精度**（NPE，normalized path energy）：OT-CFM 的路径动能非常接近 $W_2^2$，即真的解了动态 OT；$\text{moons} \leftrightarrow \text{8gaussians}$ 这类任务上 I-CFM 差距明显。
- **单细胞轨迹插值**：CITE-seq / EB / Multiome 三个数据集上 OT-CFM 的 EMD 全面优于 DSB、I-CFM、TrajectoryNet、正则化 CNF。

## 6. 一个参数统一三种方法：SB-CFM

论文第三个贡献顺着同一条思路：把 $q(z)$ 换成**熵正则** OT plan $\pi^{2\sigma^2}$，条件路径换成布朗桥

$$p_t(x \mid z) = \mathcal{N}\big(x \mid t x_1 + (1-t) x_0, \enspace t(1-t)\sigma^2\big) \tag{11} \label{eq:sb-zh}$$

**Proposition 3.5**：这样得到的边际速度场生成的概率路径，与 Schrödinger 桥问题 $\pi^{\ast} = \arg\min \mathrm{KL}(\pi \Vert p_{\mathrm{ref}})$ 的解相同。

{% note success %}
**熵正则系数 $\varepsilon$ 把三者串成一条线**：

$$\underbrace{\varepsilon \to 0}\_{\textbf{OT-CFM}} \quad \longleftarrow \quad \underbrace{\varepsilon = 2\sigma^2}\_{\textbf{SB-CFM}} \quad \longrightarrow \quad \underbrace{\varepsilon \to \infty}\_{\textbf{I-CFM（独立耦合）}}$$

独立耦合原来是熵正则趋于无穷的极限——熵最大、结构最少的那个耦合。这给了它一个准确的定位：不是"错的选择"，而是**这条谱线的一个端点**。
{% endnote %}

## 7. 与前两篇的联系

{% note info %}
| | [Flow Matching](/2025/06/12/flow-matching-three-steps/) | [MeanFlow](/2025/06/18/meanflow-one-step-generation/) | **本文 (OT-CFM)** |
|---|---|---|---|
| 核心问题 | 如何免模拟地训速度场？ | 如何一步生成？ | 如何免模拟地解动态 OT？ |
| 改哪一处 | — | 学习对象（$v \to u$） | 采样分布 $q(z)$ |
| 数学工具 | $L^2$ 正交投影 | 微积分基本定理 | 静态/动态 OT 等价（Benamou-Brenier） |
| 推理 NFE | 100-250 | 1 | 显著减少 |

三篇共享同一个起点 $x_t = (1-t)x_0 + tx_1$ 和同一个 CFM loss 结构，改的是不同的部件：MeanFlow 改学什么，OT-CFM 改从哪儿采样。两者正交，可以叠加。
{% endnote %}

## 8. 总结

1. **两个 OT**：静态 OT $\eqref{eq:static-ot-zh}$ 只问配对，可解；动态 OT $\eqref{eq:dynamic-ot-zh}$ 给速度场，有用但此前需要模拟 ODE 才能逼近。

2. **FM 与 OT 的真实关系**：条件路径是 OT，边际路径一般不是——因为无数直线交叉，边际场是平均值。这是 Table 1 中 `Cond. OT ✓ / Marginal OT ✗` 的含义。

3. **广义 CFM**（贡献一）：$\eqref{eq:grad-eq-zh}$ 只要求 $q(z)$ 边际正确，不要求独立。这把耦合变成一个自由度，也解除了源分布必须是 Gaussian 的限制。

4. **OT-CFM**（贡献二）：令 $q(z) := \pi^{\ast}$。OT plan 不交叉 $\Rightarrow$ 边际场 = 条件场 $\Rightarrow$ 条件 OT 传导为边际 OT。Prop 3.4：$\sigma \to 0$ 时解动态 OT。**静态解被免模拟地提升为动态解。**

5. **副产品**：objective variance $\to 0$（训练更快）、轨迹变直（NFE 更少）。

6. **SB-CFM**（贡献三）：熵正则 OT + 布朗桥 $\Rightarrow$ Schrödinger 桥；$\varepsilon$ 从 $0$ 到 $\infty$ 把 OT-CFM 和 I-CFM 串在一条谱线的两端。

---

**参考文献：**

<a id="note-1-zh"></a>**[1]** Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., Bengio, Y. *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*. TMLR, 2024. (arXiv:2302.00482)

<a id="note-2-zh"></a>**[2]** Benamou, J.-D. and Brenier, Y. *A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem*. Numerische Mathematik, 2000.

<a id="note-3-zh"></a>**[3]** Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M. *Flow Matching for Generative Modeling*. ICLR, 2023.

</div>

<!-- English Version -->
<div class="lang-content lang-en">

{% note info %}
**Previously**: [The Mathematics of Flow Matching](/2025/06/12/flow-matching-three-steps/) derived the CFM loss, and [From Flow Matching to MeanFlow](/2025/06/18/meanflow-one-step-generation/) swapped the learning target from instantaneous to average velocity. Both mention "optimal transport" repeatedly without pinning down: **in what sense exactly is Flow Matching optimal transport?**

**This post answers that**, by way of OT-CFM (Tong et al., TMLR 2024). The one-paragraph version:

- Flow Matching's **conditional** path is an optimal transport path, but its **marginal** path generally **is not**.
- Dynamic optimal transport (Benamou-Brenier) could previously only be approximated by simulating the ODE — costing hundreds of network evaluations per training step.
- OT-CFM's move: plug the cheap **static** OT solution into CFM's sampling distribution, thereby obtaining the **dynamic** OT solution **simulation-free**.

**Notation**: $q_0, q_1$ are source/target distributions, $x_0 \sim q_0$, $x_1 \sim q_1$; $z$ is the conditioning variable; $u_t$ is the true velocity field, $v_t^\theta$ the network.
{% endnote %}

## 1. Two Optimal Transport Problems: Static and Dynamic

Optimal transport has two faces, and Flow Matching is naturally compatible with only one of them.

### Static OT: only asks "who pairs with whom"

The Kantorovich form of the $2$-Wasserstein problem minimizes cost over all **couplings**:

$$W_2^2(q_0, q_1) = \inf\_{\pi \in \Pi(q_0, q_1)} \int \lVert x_0 - x_1 \rVert^2 \mathrm{d}\pi(x_0, x_1) \tag{1} \label{eq:static-ot-en}$$

where $\Pi(q_0, q_1)$ is the set of joint distributions with marginals $q_0$ and $q_1$. The solution $\pi^{\ast}$ is the **OT plan** — it only answers "which $x_0$ should pair with which $x_1$", involving **no time** and no ODE.

Its virtue is **solvability**: in the discrete case it is a standard linear program (or Sinkhorn), with mature solvers off the shelf.

### Dynamic OT: asks "how to travel"

Benamou-Brenier's (2000) equivalent form instead minimizes kinetic energy over **velocity fields**:

$$W_2^2(q_0, q_1) = \inf\_{p_t, u_t} \int_0^1 \int p_t(x) \lVert u_t(x) \rVert^2 \mathrm{d}x \mathrm{d}t \tag{2} \label{eq:dynamic-ot-en}$$

subject to the continuity equation $\partial_t p_t = -\nabla \cdot (p_t u_t)$ and boundary conditions $p_0 = q_0$, $p_1 = q_1$.

*This* is what generative modeling wants — it hands you a velocity field you can integrate. But it is a **variational problem over entire paths**, far harder than $\eqref{eq:static-ot-en}$.

{% note danger %}
**The prior difficulty**. To make a neural ODE approximate dynamic OT, earlier work (TrajectoryNet, Finlay et al.'s regularized CNFs) added $L^2$ kinetic regularization to a CNF. §2.2 of the paper names the cost:

> "these models required **integrating over and backpropagating through tens to hundreds of function evaluations**, resulting in both numerical and efficiency issues. We aim to avoid these issues by directly regressing to the vector field in a **simulation-free** way."

In other words: **dynamic OT is hard because it requires simulation** — every training step must actually integrate the ODE and backpropagate through it.

**The paper's core question**: $\eqref{eq:static-ot-en}$ is cheap, $\eqref{eq:dynamic-ot-en}$ is useful, and Benamou-Brenier says they are equivalent. Can we **lift** the static solution to the dynamic one, with fully simulation-free training?
{% endnote %}

## 2. Which "OT" Is the One in Flow Matching?

Before answering, let us clear up a long-standing confusion — the easiest thing to misread in Lipman et al.

The [previous post](/2025/06/12/flow-matching-three-steps/) took a sample pair and connected it with a **straight line**:

$$x_t = (1-t)x_0 + t x_1$$

A straight line *is* the dynamic OT path from $x_0$ to $x_1$ (constant-speed straight travel minimizes kinetic energy between two points). So saying FM uses "OT paths" **is correct — conditionally on a given pair $(x_0, x_1)$**.

But §3.2.1 of the OT-CFM paper deliberately adds the next sentence:

> "We emphasize that although the **conditional** probability path $p_t(x \mid z)$ **is** an optimal transport path from $p_0(x \mid z)$ to $p_1(x \mid z)$, the **marginal** path $p_t(x)$ **is not in general an OT path** from the standard normal $p_0(x)$ to the data distribution $p_1(x)$."

{% note success %}
**The relationship, disambiguated** (two levels, do not conflate):

| | Conditional level $p_t(x \mid z)$ | Marginal level $p_t(x)$ |
|---|---|---|
| What it is | A single sample's connecting line | The actual probability flow after superposing all lines |
| Is it OT? | **Yes** (a line = dynamic OT between two points) | **Generally no** |
| Who cares | An intermediate construction in the derivation | The field you actually integrate at sampling time |

In one sentence: **each line is individually optimal; their superposition is not.** Countless straight lines cross, and at a crossing point the network can only output one velocity — a weighted average of directions, no longer the velocity of any single line.
{% endnote %}

Table 1 of the paper turns this into two checkbox columns:

| Method | $q(z)$ | Cond. OT | Marginal OT |
|---|---|---|---|
| Flow Matching (Lipman et al.) | $q(x_1)$ | ✓ | ✗ |
| Rectified Flow (Liu) | $q(x_0)q(x_1)$ | ✓ | ✗ |
| Stochastic Interpolant | $q(x_0)q(x_1)$ | ✓ | ✗ |
| I-CFM (independent coupling) | $q(x_0)q(x_1)$ | ✓ | ✗ |
| **OT-CFM (theirs)** | $\pi(x_0, x_1)$ | ✓ | **✓** |

**OT-CFM exists to fill in that bottom-right cell**: make the marginal flow an OT flow too. Note that the first four rows are not "wrong" — they are all correct generative models; they just do not solve dynamic OT at the marginal level.

## 3. Step One: Generalized CFM — Making the Coupling a Free Parameter

The paper's first contribution is a framework whose purpose is to **open up a degree of freedom**.

### The mixture view

Let the marginal probability path be a mixture of conditional paths:

$$p_t(x) = \int p_t(x \mid z) q(z) \mathrm{d}z \tag{3} \label{eq:mixture-en}$$

where $z$ is an **arbitrary** conditioning variable with distribution $q(z)$. If each $p_t(x \mid z)$ is generated by $u_t(x \mid z)$, then the marginal velocity field generating the mixture $p_t(x)$ is a weighted average:

$$u_t(x) = \mathbb{E}\_{q(z)} \left[ \frac{u_t(x \mid z)\enspace p_t(x \mid z)}{p_t(x)} \right] \tag{4} \label{eq:marginal-field-en}$$

**Theorem 3.1**: $\eqref{eq:marginal-field-en}$ indeed generates $\eqref{eq:mixture-en}$.

The trouble is that $\eqref{eq:marginal-field-en}$ is intractable — the denominator $p_t(x)$ is an integral. Hence the **CFM loss**, which regresses only against the **conditional** field:

$$\mathcal{L}\_{\mathrm{CFM}}(\theta) = \mathbb{E}\_{t, q(z), p_t(x \mid z)} \lVert v_t^\theta(x) - u_t(x \mid z) \rVert^2 \tag{5} \label{eq:cfm-en}$$

**Theorem 3.2**: $\mathcal{L}\_{\mathrm{CFM}}$ and $\mathcal{L}\_{\mathrm{FM}}$ differ by a constant independent of $\theta$, hence

$$\nabla\_\theta \mathcal{L}\_{\mathrm{FM}}(\theta) = \nabla\_\theta \mathcal{L}\_{\mathrm{CFM}}(\theta) \tag{6} \label{eq:grad-eq-en}$$

This is exactly what the [previous post](/2025/06/12/flow-matching-three-steps/) proved via the Pythagorean theorem, here generalized to arbitrary $z$.

### The key point: $q(z)$ is free

Take $z := (x_0, x_1)$ with the conditional path a Gaussian bridge between the two points:

$$p_t(x \mid z) = \mathcal{N}(x \mid t x_1 + (1-t) x_0, \enspace \sigma^2), \qquad u_t(x \mid z) = x_1 - x_0 \tag{7} \label{eq:icfm-en}$$

Then $\eqref{eq:cfm-en}$ is precisely the loss derived in the [previous post](/2025/06/12/flow-matching-three-steps/), which took $q(z) = q(x_0) q(x_1)$ — the **independent coupling**, which the paper names **I-CFM**.

{% note info %}
**This is where the framework earns its keep**: $\eqref{eq:grad-eq-en}$ holds **as long as $q(z)$ has marginals $q_0$ and $q_1$**, with no requirement that $x_0 \perp x_1$.

So independence is not an "assumption" but a **default** taken when constructing $q(z)$ — the previous derivation has no gap, and $Z_1 \sim q_1$ holds exactly. What it does mean is that $q(z)$ is a **previously unused degree of freedom**.

The paper's first contribution is to make that freedom explicit (which incidentally also removes FM's restriction to a Gaussian source, since FM conditions only on $q(x_1)$ — any source distribution now works). The second contribution is to use it.
{% endnote %}

## 4. Step Two: Plugging the Static OT Plan into $q(z)$

Now §1's question has an almost obvious answer.

### OT-CFM

Let the conditioning distribution be the **static OT plan**:

$$\boxed{q(z) := \pi^{\ast}(x_0, x_1)} \tag{8} \label{eq:otcfm-en}$$

i.e., the solution of $\eqref{eq:static-ot-en}$. The conditional path stays exactly as in $\eqref{eq:icfm-en}$ — **the only change is that $(x_0, x_1)$ is no longer sampled independently but jointly according to $\pi^{\ast}$**.

{% note success %}
**Proposition 3.4**: under regularity conditions on $q_0$, $q_1$, and $\pi^{\ast}$, as $\sigma^2 \to 0$ the marginal path $p_t$ and field $u_t$ induced by $\eqref{eq:otcfm-en}$ **minimize $\eqref{eq:dynamic-ot-en}$**, i.e., $u_t$ solves the **dynamic optimal transport** problem between $q_0$ and $q_1$.

This completes the lift: the input is a solution to **static** OT (a pairing), the output is a solution to **dynamic** OT (a velocity field), and training is just the regression $\eqref{eq:cfm-en}$ — **no ODE simulation anywhere**.
{% endnote %}

### Why plugging it in works

The key is Brenier's theorem: the OT plan is deterministic and **non-crossing**, $x_1 = \nabla \Psi^{\ast}(x_0)$ (the gradient of a convex function).

This closes the gap from §2. Since the family of straight lines does not cross, each point $x$ lies on exactly one line, and the weighted average in $\eqref{eq:marginal-field-en}$ **collapses to a single value**:

$$u_t(x) = u_t(x \mid z) = \nabla \Psi^{\ast}(x_0) - x_0$$

The conditional field and the marginal field coincide. And since every conditional path is an OT path, so is the marginal path. McCann's displacement interpolation $p_t = [(1-t)\mathrm{Id} + t \nabla\Psi^{\ast}]\_{\sharp} q_0$ is exactly the dynamic OT solution.

{% note danger %}
**Contrast with §2**: under independent coupling the lines cross heavily $\to$ $\eqref{eq:marginal-field-en}$ really is averaging $\to$ conditional OT fails to reach the marginal. Under OT coupling there is no crossing $\to$ no averaging $\to$ conditional OT *is* marginal OT.

**That is the entire mechanism behind the `Marginal OT` column of the checkbox table.**
{% endnote %}

### Practical recipe: minibatch OT

An exact OT plan costs $O(n^3)$ time and $O(n^2)$ memory, infeasible for large datasets. The paper splits by data scale:

1. **Small datasets** (e.g. single-cell): compute the exact OT plan; the network's job is to **extrapolate that map to unseen data**.
2. **Large datasets** (e.g. images): **solve OT within each batch**. Sample $\{x_0^i\}\_{i=1}^B$ and $\{x_1^i\}\_{i=1}^B$, solve the $B \times B$ assignment problem, and use the resulting pairs in $\eqref{eq:cfm-en}$.

{% note warning %}
**Minibatch OT is biased** — it is not global OT. But the paper notes:

- When the OT batch size equals the support size of $(q_0, q_1)$, exact OT is recovered, and by Prop 3.4 exact dynamic OT is learned.
- Empirically, batch sizes far below the dataset size already approximate the OT map well (Fig. D.2); the authors attribute this to generalization from the network optimization itself.
- Training overhead is $< 1\%$.
{% endnote %}

## 5. By-products: Both Training and Inference Get Faster

The paper's conclusion reads "lifting static OT to dynamic ... **while also** allowing more efficient training and inference". The following two are that "while also" — important, but logically **consequences**, not the starting point.

### Faster training: lower variance in the regression target

$\eqref{eq:cfm-en}$ is a **stochastic** regression: the same $x$ can arise from different $z$, so the target $u_t(x \mid z)$ jitters. The paper calls this jitter the **objective variance**:

$$\mathrm{OV} := \mathbb{E} \lVert u_t(x \mid z) - u_t(x) \rVert^2 \tag{9} \label{eq:ov-en}$$

It is precisely the "constant independent of $\theta$" in $\eqref{eq:grad-eq-en}$, and also the floor of $\mathcal{L}\_{\mathrm{CFM}}$ — no network, however strong, gets below it.

**Proposition B.2**: if $\pi$ is a Monge map, then $\mathrm{OV} \to 0$ as $\sigma \to 0$ for OT-CFM. This follows directly from §4's "conditional field = marginal field". Under independent coupling $\mathrm{OV}$ does not vanish.

Measured: OT-CFM and SB-CFM have $\mathrm{OV}$ at least **an order of magnitude below** I-CFM/FM, with correspondingly faster validation-error decay (Fig. 2 left, Fig. D.8).

### Faster inference: fewer NFE

Inference numerically integrates $\mathrm{d}Z_t = v_t^\theta(Z_t)\mathrm{d}t$, and each evaluation of $v_t^\theta$ is one forward pass. The total count is the **NFE** (Number of Function Evaluations) — the inference cost itself.

The truncation error of $N$-step Euler is governed by acceleration: expanding $Z_{t+h} = Z_t + h\dot{Z}\_t + \frac{h^2}{2}\ddot{Z}\_t + O(h^3)$, Euler drops the $O(h^2 \lVert \ddot{Z}\_t \rVert)$ term, and after accumulation

$$\mathrm{NFE} = N \gtrsim \frac{C}{\varepsilon} \sup\_{t} \lVert \ddot{Z}\_t \rVert \tag{10} \label{eq:nfe-en}$$

Under OT coupling the trajectories are **straight**, so $\ddot{Z}\_t \equiv 0$ — zero truncation error, exact in one step in principle.

Measured (Fig. 2 right, Fig. 3 right): at fixed NFE, OT-CFM produces better samples; to reach the same quality it needs fewer NFE. On CIFAR-10 OT-CFM beats I-CFM and FM on both FID and NFE.

### Other experimental findings

- **Dynamic OT accuracy** (NPE, normalized path energy): OT-CFM's path energy is very close to $W_2^2$, i.e. it genuinely solves dynamic OT; on tasks like $\text{moons} \leftrightarrow \text{8gaussians}$ the gap to I-CFM is stark.
- **Single-cell trajectory interpolation**: on CITE-seq / EB / Multiome, OT-CFM's EMD beats DSB, I-CFM, TrajectoryNet, and regularized CNFs across the board.

## 6. One Parameter Unifies Three Methods: SB-CFM

The third contribution follows the same thread: replace $q(z)$ with the **entropically regularized** OT plan $\pi^{2\sigma^2}$ and the conditional path with a Brownian bridge

$$p_t(x \mid z) = \mathcal{N}\big(x \mid t x_1 + (1-t) x_0, \enspace t(1-t)\sigma^2\big) \tag{11} \label{eq:sb-en}$$

**Proposition 3.5**: the resulting marginal velocity field generates the same probability path as the solution of the Schrödinger bridge problem $\pi^{\ast} = \arg\min \mathrm{KL}(\pi \Vert p_{\mathrm{ref}})$.

{% note success %}
**The entropic coefficient $\varepsilon$ strings all three together**:

$$\underbrace{\varepsilon \to 0}\_{\textbf{OT-CFM}} \quad \longleftarrow \quad \underbrace{\varepsilon = 2\sigma^2}\_{\textbf{SB-CFM}} \quad \longrightarrow \quad \underbrace{\varepsilon \to \infty}\_{\textbf{I-CFM (independent)}}$$

The independent coupling turns out to be the infinite-regularization limit — the maximum-entropy, least-structured coupling. That gives it a precise place: not a "wrong choice", but **one endpoint of this spectrum**.
{% endnote %}

## 7. Connection to the Previous Two Posts

{% note info %}
| | [Flow Matching](/2025/06/12/flow-matching-three-steps/) | [MeanFlow](/2025/06/18/meanflow-one-step-generation/) | **This post (OT-CFM)** |
|---|---|---|---|
| Core question | How to train a velocity field simulation-free? | How to generate in one step? | How to solve dynamic OT simulation-free? |
| What changes | — | The learning target ($v \to u$) | The sampling distribution $q(z)$ |
| Mathematical tool | $L^2$ orthogonal projection | Fundamental Theorem of Calculus | Static/dynamic OT equivalence (Benamou-Brenier) |
| Inference NFE | 100-250 | 1 | Substantially fewer |

All three share the same starting point $x_t = (1-t)x_0 + tx_1$ and the same CFM loss structure, changing different components: MeanFlow changes *what* is learned, OT-CFM changes *where samples come from*. The two are orthogonal and composable.
{% endnote %}

## 8. Summary

1. **Two OTs**: static OT $\eqref{eq:static-ot-en}$ only asks about pairing and is solvable; dynamic OT $\eqref{eq:dynamic-ot-en}$ hands you a velocity field — useful, but previously approximable only by simulating the ODE.

2. **The real FM-OT relationship**: the conditional path is OT, the marginal path generally is not — because countless straight lines cross and the marginal field is an average. This is the meaning of `Cond. OT ✓ / Marginal OT ✗` in Table 1.

3. **Generalized CFM** (contribution 1): $\eqref{eq:grad-eq-en}$ only requires $q(z)$ to have the right marginals, not independence. This turns the coupling into a degree of freedom, and also lifts the requirement that the source be Gaussian.

4. **OT-CFM** (contribution 2): set $q(z) := \pi^{\ast}$. The OT plan does not cross $\Rightarrow$ marginal field = conditional field $\Rightarrow$ conditional OT propagates to marginal OT. Prop 3.4: as $\sigma \to 0$ it solves dynamic OT. **The static solution is lifted to a dynamic one, simulation-free.**

5. **By-products**: objective variance $\to 0$ (faster training), straighter trajectories (fewer NFE).

6. **SB-CFM** (contribution 3): entropic OT + Brownian bridge $\Rightarrow$ Schrödinger bridge; sweeping $\varepsilon$ from $0$ to $\infty$ places OT-CFM and I-CFM at the two ends of one spectrum.

---

**References:**

<a id="note-1-en"></a>**[1]** Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., Bengio, Y. *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*. TMLR, 2024. (arXiv:2302.00482)

<a id="note-2-en"></a>**[2]** Benamou, J.-D. and Brenier, Y. *A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem*. Numerische Mathematik, 2000.

<a id="note-3-en"></a>**[3]** Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M. *Flow Matching for Generative Modeling*. ICLR, 2023.

</div>

<script>
function switchLang(lang) {
  // Switch content
  document.querySelectorAll('.lang-content').forEach(function(el) {
    el.style.display = 'none';
  });
  document.querySelectorAll('.lang-btn').forEach(function(el) {
    el.classList.remove('active');
  });
  document.querySelector('.lang-' + lang).style.display = 'block';
  document.getElementById('btn-' + lang).classList.add('active');

  // Switch title
  var titleEl = document.querySelector('.post-title');
  if (titleEl) {
    if (lang === 'zh') {
      titleEl.textContent = 'Flow Matching 与最优传输：把静态 OT 提升为动态 OT';
    } else {
      titleEl.textContent = 'Flow Matching and Optimal Transport: Lifting Static OT to Dynamic OT';
    }
  }

  // Switch TOC: hide headings from the other language
  var tocLinks = document.querySelectorAll('.post-toc a');
  tocLinks.forEach(function(link) {
    var li = link.closest('li');
    if (!li) return;
    var isChinese = /[一-鿿]/.test(link.textContent);
    if (lang === 'zh') {
      li.style.display = isChinese ? '' : 'none';
    } else {
      li.style.display = isChinese ? 'none' : '';
    }
  });
}

// Run on page load to set initial TOC state
document.addEventListener('DOMContentLoaded', function() {
  switchLang('en');
});
</script>
