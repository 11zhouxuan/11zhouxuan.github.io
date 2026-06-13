---
title: "The Mathematics of Flow Matching"
date: 2025-06-12 10:00:00
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

Flow Matching——连续生成模型中"无需模拟"训练目标的极简推导，三步到位。

<!-- more -->

{% note info %}
**目标**：给定源分布 $p\_0$（如标准高斯）和目标分布 $p\_1$（如真实数据），学习一个速度场 $v\_t$，使得由 $v\_t$ 驱动的 ODE 能将 $X\_0 \sim p\_0$ 的样本搬运到接近 $X\_1 \sim p\_1$ 的分布。

**本文逻辑**：**设定** → **构造辅助随机过程** → **推出目标速度场** → **化为可计算损失**。全文三步到位。
{% endnote %}

## 1. 设定：输入与目标

我们有三个基本对象：

1. **两个随机变量**：$X\_0 \sim p\_0$（源分布，如 $\mathcal{N}(0, I)$），$X\_1 \sim p\_1$（目标分布，如自然图像）。

2. **一个 ODE**：由速度场 $v\_t : \mathbb{R}^d \to \mathbb{R}^d$（$t \in [0,1]$）驱动的常微分方程。给定初始点 $x\_0 \in \mathbb{R}^d$，定义轨迹 $Z : [0,1] \to \mathbb{R}^d$ 为

$$\frac{\mathrm{d}}{\mathrm{d}t}Z(t) = v\_t\big(Z(t)\big), \qquad Z(0) = x\_0. \tag{1}$$

即 $Z(t) \in \mathbb{R}^d$ 是从 $x\_0$ 出发、沿速度场运动到时刻 $t$ 的位置。当初始点取为随机变量 $X\_0$ 时，我们把对应的轨迹简记为 $Z\_t$（即 $Z\_t := Z(t)$，初值 $Z\_0 = X\_0$），终点为 $Z\_1$。

3. **目标**：找到 $v\_t$，使得 $Z\_1$ 的分布逼近 $p\_1$——即 $X\_0$ 在 ODE 驱动下到 $t=1$ 时的分布接近目标分布。

### 为什么用 ODE？——最优控制的视角

我们的核心问题是：寻找一个未知函数 $v\_t$，使得它驱动的动力系统能将源分布搬运到目标分布。这本质上是一个**最优控制问题**——$v\_t$ 是"控制量"，ODE (1) 是"状态方程"，目标是终态分布逼近 $p\_1$。ODE 提供了一个连续的、可微的演化框架，使得我们可以将"搬运分布"这个困难的静态问题，转化为"学一个速度场"这个可以用梯度方法求解的优化问题。<a href="#note-1" style="text-decoration:none"><sup>[1]</sup></a>

## 2. 第一步：构造辅助随机过程

### 核心假设

为了给 $v\_t$ 的学习提供"参照"，我们构造一个**辅助随机过程**：对独立采样的 $X\_0 \sim p\_0$、$X\_1 \sim p\_1$，定义

$$X\_t := (1-t)\,X\_0 + t\,X\_1, \qquad t \in [0,1]. \tag{2}$$

这个随机过程具有以下性质：
- **端点正确**：$t=0$ 时 $X\_0 = X\_0 \sim p\_0$，$t=1$ 时 $X\_1 = X\_1 \sim p\_1$；
- 对每一对固定的 $(X\_0, X\_1)$，$t \mapsto X\_t$ 的速度恒为 $\frac{\mathrm{d}}{\mathrm{d}t}X\_t = X\_1 - X\_0$。

### 关键区分：$X\_t$ 不能用来采样 $X\_1$

注意 $X\_t$ 的定义**本身依赖于 $X\_1$**——要计算 $X\_t$，你必须已经知道终点 $X\_1$ 是什么。因此，这个随机过程**不能**作为从 $p\_1$ 中生成样本的方法（它是一个分析工具，不是生成工具）。

真正的生成机制是 ODE (1)：它只需要初始点 $X\_0 \sim p\_0$ 和速度场 $v\_t$，不依赖 $X\_1$。我们的目标是找到 $v\_t$，使得 ODE 驱动产生的随机变量 $Z\_t$ 与辅助过程 $X\_t$ **同分布**。特别地，当 $t=1$ 时，$Z\_1 \overset{d}{=} X\_1$——这才是生成。

### 为什么要构造 $X\_t$？

$X\_t$ 提供了一个"理想参照"：它的边际分布从 $p\_0$ 平滑过渡到 $p\_1$。如果我们能找到一个 $v\_t$ 使得 $Z\_t$ 在每个时刻都与 $X\_t$ 同分布，那么终态自然有 $Z\_1 \sim p\_1$。这样，"搬运分布到 $p\_1$"就转化为"让 $v\_t$ 驱动出的随机变量与 $X\_t$ 同分布"。<a href="#note-2" style="text-decoration:none"><sup>[2]</sup></a>

## 3. 第二步：推出目标速度场

### 问题

我们已构造辅助过程 $X\_t = (1-t)X\_0 + tX\_1$。现在问：**哪个速度场 $v\_t$ 能使 $Z\_t$ 与 $X\_t$ 同分布？**

### 同分布的等价条件

两个随机变量同分布，当且仅当它们在所有测试函数下的期望相同。即要求对任意光滑测试函数 $f$，

$$\mathbb{E}[f(Z\_t)] = \mathbb{E}[f(X\_t)], \qquad \forall\, t \in [0,1]. \tag{3}$$

对 (3) 两边关于 $t$ 求导，就能推出 $v\_t$ 必须满足的条件。

### 左边对 $t$ 求导

由 ODE (1)，$\frac{\mathrm{d}}{\mathrm{d}t}Z\_t = v\_t(Z\_t)$。对 $f(Z\_t)$ 用链式法则：

$$\frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}[f(Z\_t)] = \mathbb{E}[\nabla f(Z\_t) \cdot v\_t(Z\_t)].$$

若 (3) 成立（即 $Z\_t \overset{d}{=} X\_t$），则可将 $Z\_t$ 替换为 $X\_t$：

$$\frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}[f(X\_t)] = \mathbb{E}[\nabla f(X\_t) \cdot v\_t(X\_t)]. \tag{4}$$

### 右边对 $t$ 求导

由 $X\_t = (1-t)X\_0 + tX\_1$，直接求导得 $\frac{\mathrm{d}}{\mathrm{d}t}X\_t = X\_1 - X\_0$，链式法则给出：

$$\frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}[f(X\_t)] = \mathbb{E}[\nabla f(X\_t) \cdot (X\_1 - X\_0)]. \tag{5}$$

### 联立两边

由 (4) = (5)，得对一切 $f$ 成立：

$$\mathbb{E}[\nabla f(X\_t) \cdot v\_t(X\_t)] = \mathbb{E}[\nabla f(X\_t) \cdot (X\_1 - X\_0)].$$

注意 $v\_t(X\_t)$ 是 $X\_t$ 的确定性函数，而 $X\_1 - X\_0$ 不仅仅依赖 $X\_t$——它还依赖具体的 $(X\_0, X\_1)$ 对。对右边使用**条件期望的塔性质**（先对 $X\_t$ 取条件，$\nabla f(X\_t)$ 在内层可视为常数）：

$$\mathbb{E}[\nabla f(X\_t) \cdot (X\_1 - X\_0)] = \mathbb{E}[\nabla f(X\_t) \cdot \mathbb{E}[X\_1 - X\_0 \mid X\_t]].$$

由 $f$ 的任意性，两边的"系数"在几乎处处必须相等，得到：

$$\boxed{v\_t(x) = \mathbb{E}[X\_1 - X\_0 \mid X\_t = x]} \tag{6}$$

<a href="#note-3" style="text-decoration:none"><sup>[3]</sup></a>

<a href="#note-4" style="text-decoration:none"><sup>[4]</sup></a>

{% note danger %}
**障碍**：公式 (6) 给出了 $v\_t(x)$ 的解析表达，但**不好直接计算**——计算条件期望需要知道 $X\_1$ 的密度函数 $p\_1$（从而得到联合密度与 $X\_t$ 的边际密度），而 $p\_1$ 正是我们没有的——我们只有从 $p\_1$ 中抽出的有限样本。
{% endnote %}

## 4. 第三步：化为可计算的损失

$v\_t$ 有解析解（公式 (6)），只是不好计算。一个自然的想法是直接用参数化的 $v\_t^\theta$（如神经网络）去逼近 $v\_t$，写成 **Flow Matching (FM) 损失**：

$$\mathcal{L}\_{\mathrm{FM}}(\theta) = \mathbb{E}\_{t \sim \mathcal{U}[0,1]} \int \lVert v\_t^\theta(x) - v\_t(x)\rVert^2 \, p\_t(x)\,\mathrm{d}x \tag{7}$$

其中 $p\_t$ 是 $X\_t$ 的边际密度。但这同样不可算——它需要 $v\_t(x)$ 和 $p\_t(x)$ 的值，两者都依赖于未知的 $p\_1$。

关键在于：我们**不需要计算** $v\_t$ 和 $p\_t$ 的值，就能得到一个与 $\mathcal{L}\_{\mathrm{FM}}$ 梯度相同的可计算损失。

### 关键事实

$v\_t(X\_t) = \mathbb{E}[X\_1 - X\_0 \mid X\_t]$ 是 $X\_1 - X\_0$ 在 $\sigma(X\_t)$（$X\_t$ 生成的 $\sigma$-代数）上的正交投影。考察以 $X\_1 - X\_0$ 为标签的回归损失：

### 推导

将 $v\_t^\theta(X\_t)$ 与标签 $X\_1 - X\_0$ 之间的误差展开——插入 $v\_t(X\_t)$：

$$\mathbb{E}\lVert v\_t^\theta(X\_t) - (X\_1 - X\_0)\rVert^2 = \mathbb{E}\lVert v\_t^\theta(X\_t) - v\_t(X\_t)\rVert^2 + 2\,\text{Cross} + \mathbb{E}\lVert v\_t(X\_t) - (X\_1 - X\_0)\rVert^2 \tag{8}$$

交叉项为零：记 $h(X\_t) := v\_t^\theta(X\_t) - v\_t(X\_t)$（$X\_t$ 的函数），由塔性质：

$$\text{Cross} = \mathbb{E}[h(X\_t) \cdot (v\_t(X\_t) - (X\_1 - X\_0))] = \mathbb{E}[h(X\_t) \cdot \mathbb{E}[v\_t(X\_t) - (X\_1 - X\_0) \mid X\_t]] = 0 \tag{9}$$

因为 $v\_t(X\_t) = \mathbb{E}[X\_1-X\_0 \mid X\_t]$，条件期望内部恰好为零。

因此 (8) 化简为：

$$\mathbb{E}\lVert v\_t^\theta(X\_t) - (X\_1 - X\_0)\rVert^2 = \underbrace{\mathbb{E}\lVert v\_t(X\_t) - (X\_1 - X\_0)\rVert^2}\_{\text{常数}} + \mathbb{E}\lVert v\_t^\theta(X\_t) - v\_t(X\_t)\rVert^2 \tag{10}$$

### 结论

由 (10)，最小化左边（标签为可采样的 $X\_1-X\_0$）等价于最小化右边第二项（逼近 $v\_t$）：

$$\arg\min\_\theta \, \mathbb{E}\lVert v\_t^\theta(X\_t) - (X\_1 - X\_0)\rVert^2 = \arg\min\_\theta \, \mathbb{E}\lVert v\_t^\theta(X\_t) - v\_t(X\_t)\rVert^2$$

### Conditional Flow Matching (CFM) 损失

对时间也取期望，得到最终的训练目标：

$$\boxed{\mathcal{L}\_{\mathrm{CFM}}(\theta) = \mathbb{E}\_{t \sim \mathcal{U}[0,1]} \, \mathbb{E}\_{X\_0 \sim p\_0,\, X\_1 \sim p\_1}\, \lVert v\_t^\theta((1-t)X\_0 + tX\_1) - (X\_1 - X\_0)\rVert^2} \tag{11}$$

{% note success %}
**为什么这个损失完全可算？**
- 不需要 $v\_t$ 的值——由正交性（公式 (10)），回归标签等价替换为可采样的 $X\_1 - X\_0$；
- 不需要 $X\_t$ 的边际密度公式——输入点 $(1-t)X\_0 + tX\_1$ 就是 $X\_t$ 的一个样本；
- 蒙特卡洛估计只需：均匀抽 $t$，抽噪声 $X\_0 \sim p\_0$，抽数据 $X\_1 \sim p\_1$。

这就是 Flow Matching "simulation-free"（训练中无需积分 ODE）的根本原因。
{% endnote %}

<a href="#note-5" style="text-decoration:none"><sup>[5]</sup></a>

## 5. 总结：三步逻辑链

1. **构造辅助过程**：定义 $X\_t = (1-t)X\_0 + tX\_1$。这个随机过程本身依赖 $X\_1$，不能用于生成，但它从 $p\_0$ 平滑过渡到 $p\_1$，为学习 $v\_t$ 提供了参照。

2. **推出目标速度场**：要求 ODE 的流 $Z\_t$ 与 $X\_t$ 同分布。对"同分布"条件两边求导、用塔性质，推出：$v\_t(x) = \mathbb{E}[X\_1 - X\_0 \mid X\_t = x]$。

3. **化为可计算的回归**：由 $L^2$ 正交投影的勾股定理，回归 $v\_t$（不可算）与回归 $X\_1 - X\_0$（可采样）等价，得到 Flow Matching 损失 (11)。

---

**注释：**

<a id="note-1"></a>**[1]** **推理**：训练完成后，从 $X\_0 \sim p\_0$ 出发积分 ODE 到 $t=1$ 即得生成样本 $Z\_1$。此时不需要 $X\_1$。

<a id="note-2"></a>**[2]** 线性插值 (2) 在 Benamou-Brenier (2000) 意义下是最优的：它在连接 $p\_0$ 到 $p\_1$ 的所有路径中最小化动能。参见 McCann (1997)。

<a id="note-3"></a>**[3]** **记号**：$\mathbb{E}[\cdot \mid X\_t = x]$ 给出 $x$ 的确定性函数，而非随机变量。等式 (6) 两边都是 $x$ 的函数。

<a id="note-4"></a>**[4]** **直觉**：多条轨迹穿过同一点 $x$，来自不同的 $(X\_0,X\_1)$ 对。目标速度取其条件平均。

<a id="note-5"></a>**[5]** Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023. Liu et al., *Flow Straight and Fast*, ICLR 2023 (Rectified Flow).

</div>

<!-- English Version -->
<div class="lang-content lang-en">

A minimal derivation of Flow Matching — the "simulation-free" training objective for continuous generative models — in three clean steps.

{% note info %}
**Goal**: Given a source distribution $p\_0$ (e.g., standard Gaussian) and a target distribution $p\_1$ (e.g., real data), learn a velocity field $v\_t$ so that the ODE driven by $v\_t$ transports samples $X\_0 \sim p\_0$ to approximately follow $X\_1 \sim p\_1$.

**Logic**: **Setting** → **Construct auxiliary process** → **Derive target velocity field** → **Reduce to computable loss**. Three steps, done.
{% endnote %}

## 1. Setting: Inputs and Objective

We have three basic objects:

1. **Two random variables**: $X\_0 \sim p\_0$ (source, e.g., $\mathcal{N}(0, I)$), $X\_1 \sim p\_1$ (target, e.g., natural images).

2. **An ODE**: driven by a velocity field $v\_t : \mathbb{R}^d \to \mathbb{R}^d$ ($t \in [0,1]$). Given an initial point $x\_0 \in \mathbb{R}^d$, define the trajectory $Z: [0,1] \to \mathbb{R}^d$ by

$$\frac{\mathrm{d}}{\mathrm{d}t}Z(t) = v\_t(Z(t)), \qquad Z(0) = x\_0. \tag{1}$$

That is, $Z(t)$ is the position reached at time $t$ starting from $x\_0$ following the velocity field. When the initial point is the random variable $X\_0$, we write $Z\_t := Z(t)$ with $Z\_0 = X\_0$, and the endpoint is $Z\_1$.

3. **Objective**: Find $v\_t$ such that $Z\_1 \sim p\_1$ — i.e., the distribution of the ODE endpoint approximates the target.

### Why an ODE? — The Optimal Control Viewpoint

Our core problem is: find an unknown function $v\_t$ such that the dynamical system it drives transports the source distribution to the target. This is essentially an **optimal control problem** — $v\_t$ is the "control", the ODE (1) is the "state equation", and the goal is for the terminal distribution to approximate $p\_1$. The ODE provides a continuous, differentiable evolution framework that turns the hard static problem of "transporting a distribution" into the tractable optimization problem of "learning a velocity field" via gradient methods.<a href="#note-1" style="text-decoration:none"><sup>[1]</sup></a>

## 2. Step One: Construct the Auxiliary Random Process

### Core Assumption

To provide a "reference" for learning $v\_t$, we construct an **auxiliary random process**: for independently sampled $X\_0 \sim p\_0$, $X\_1 \sim p\_1$, define

$$X\_t := (1-t)\,X\_0 + t\,X\_1, \qquad t \in [0,1]. \tag{2}$$

This process has the following properties:
- **Correct endpoints**: $X\_0 = X\_0 \sim p\_0$ at $t=0$, and $X\_1 = X\_1 \sim p\_1$ at $t=1$;
- For each fixed pair $(X\_0, X\_1)$, the velocity along $t \mapsto X\_t$ is constant: $\frac{\mathrm{d}}{\mathrm{d}t}X\_t = X\_1 - X\_0$.

### Key Distinction: $X\_t$ Cannot Be Used for Sampling

Note that the definition of $X\_t$ **depends on $X\_1$ itself** — to compute $X\_t$, you must already know the endpoint $X\_1$. Therefore, this process **cannot** serve as a method for generating samples from $p\_1$ (it's an analytical tool, not a generative one).

The actual generation mechanism is the ODE (1): it only needs an initial point $X\_0 \sim p\_0$ and the velocity field $v\_t$, without depending on $X\_1$. Our goal is to find $v\_t$ such that the ODE-driven random variable $Z\_t$ has the **same distribution** as $X\_t$. In particular, at $t=1$: $Z\_1 \overset{d}{=} X\_1$ — that's generation.

### Why Construct $X\_t$?

$X\_t$ provides an "ideal reference": its marginal distribution transitions smoothly from $p\_0$ to $p\_1$. If we can find a $v\_t$ making $Z\_t$ equal in distribution to $X\_t$ at every time, then the endpoint automatically satisfies $Z\_1 \sim p\_1$. Thus, "transport the distribution to $p\_1$" becomes "make the ODE-driven random variable match $X\_t$ in distribution".<a href="#note-2" style="text-decoration:none"><sup>[2]</sup></a>

## 3. Step Two: Derive the Target Velocity Field

### The Question

We've constructed the auxiliary process $X\_t = (1-t)X\_0 + tX\_1$. Now ask: **which velocity field $v\_t$ makes $Z\_t$ equal in distribution to $X\_t$?**

### Equivalent Condition for Equal Distribution

Two random variables have the same distribution if and only if they yield the same expectation under all test functions. We require: for all smooth test functions $f$,

$$\mathbb{E}[f(Z\_t)] = \mathbb{E}[f(X\_t)], \qquad \forall\, t \in [0,1]. \tag{3}$$

Differentiating both sides of (3) with respect to $t$ yields the condition that $v\_t$ must satisfy.

### Differentiating the Left Side

By ODE (1), $\frac{\mathrm{d}}{\mathrm{d}t}Z\_t = v\_t(Z\_t)$. Applying the chain rule to $f(Z\_t)$:

$$\frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}[f(Z\_t)] = \mathbb{E}[\nabla f(Z\_t) \cdot v\_t(Z\_t)].$$

If (3) holds (i.e., $Z\_t \overset{d}{=} X\_t$), we can replace $Z\_t$ with $X\_t$:

$$\frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}[f(X\_t)] = \mathbb{E}[\nabla f(X\_t) \cdot v\_t(X\_t)]. \tag{4}$$

### Differentiating the Right Side

From $X\_t = (1-t)X\_0 + tX\_1$, direct differentiation gives $\frac{\mathrm{d}}{\mathrm{d}t}X\_t = X\_1 - X\_0$, and the chain rule yields:

$$\frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}[f(X\_t)] = \mathbb{E}[\nabla f(X\_t) \cdot (X\_1 - X\_0)]. \tag{5}$$

### Equating Both Sides

From (4) = (5), for all $f$:

$$\mathbb{E}[\nabla f(X\_t) \cdot v\_t(X\_t)] = \mathbb{E}[\nabla f(X\_t) \cdot (X\_1 - X\_0)].$$

Note that $v\_t(X\_t)$ is a deterministic function of $X\_t$, while $X\_1 - X\_0$ depends on the specific $(X\_0, X\_1)$ pair, not just $X\_t$. Applying the **tower property of conditional expectation** to the right side (conditioning on $X\_t$, where $\nabla f(X\_t)$ becomes a constant in the inner expectation):

$$\mathbb{E}[\nabla f(X\_t) \cdot (X\_1 - X\_0)] = \mathbb{E}[\nabla f(X\_t) \cdot \mathbb{E}[X\_1 - X\_0 \mid X\_t]].$$

By the arbitrariness of $f$, the "coefficients" must be equal almost everywhere:

$$\boxed{v\_t(x) = \mathbb{E}[X\_1 - X\_0 \mid X\_t = x]} \tag{6}$$

<a href="#note-3" style="text-decoration:none"><sup>[3]</sup></a>

<a href="#note-4" style="text-decoration:none"><sup>[4]</sup></a>

{% note danger %}
**The obstacle**: Formula (6) gives an analytic expression for $v\_t(x)$, but it's **hard to compute directly** — evaluating the conditional expectation requires knowing $p\_1$ (to obtain joint and marginal densities), and $p\_1$ is precisely what we don't have — we only have finite samples from $p\_1$.
{% endnote %}

## 4. Step Three: Reduce to a Computable Loss

$v\_t$ has an analytic solution (formula (6)), it's just hard to compute. A natural idea is to directly approximate $v\_t$ with a parameterized $v\_t^\theta$ (e.g., a neural network), writing the **Flow Matching (FM) loss**:

$$\mathcal{L}\_{\mathrm{FM}}(\theta) = \mathbb{E}\_{t \sim \mathcal{U}[0,1]} \int \lVert v\_t^\theta(x) - v\_t(x)\rVert^2 \, p\_t(x)\,\mathrm{d}x \tag{7}$$

where $p\_t$ is the marginal density of $X\_t$. But this is also intractable — it requires the values of $v\_t(x)$ and $p\_t(x)$, both depending on the unknown $p\_1$.

The key insight: we **don't need to compute** $v\_t$ or $p\_t$ to obtain a loss with the same gradient as $\mathcal{L}\_{\mathrm{FM}}$.

### The Key Fact

$v\_t(X\_t) = \mathbb{E}[X\_1 - X\_0 \mid X\_t]$ is the orthogonal projection of $X\_1 - X\_0$ onto $\sigma(X\_t)$ (the $\sigma$-algebra generated by $X\_t$). Consider the regression loss with $X\_1 - X\_0$ as the label:

### Derivation

Expand the error between $v\_t^\theta(X\_t)$ and the label $X\_1 - X\_0$ — inserting $v\_t(X\_t)$:

$$\mathbb{E}\lVert v\_t^\theta(X\_t) - (X\_1 - X\_0)\rVert^2 = \mathbb{E}\lVert v\_t^\theta(X\_t) - v\_t(X\_t)\rVert^2 + 2\,\text{Cross} + \mathbb{E}\lVert v\_t(X\_t) - (X\_1 - X\_0)\rVert^2 \tag{8}$$

The cross term vanishes: let $h(X\_t) := v\_t^\theta(X\_t) - v\_t(X\_t)$ (a function of $X\_t$), by tower property:

$$\text{Cross} = \mathbb{E}[h(X\_t) \cdot (v\_t(X\_t) - (X\_1 - X\_0))] = \mathbb{E}[h(X\_t) \cdot \mathbb{E}[v\_t(X\_t) - (X\_1 - X\_0) \mid X\_t]] = 0 \tag{9}$$

since $v\_t(X\_t) = \mathbb{E}[X\_1-X\_0 \mid X\_t]$, the conditional expectation inside is exactly zero.

Therefore (8) simplifies to:

$$\mathbb{E}\lVert v\_t^\theta(X\_t) - (X\_1 - X\_0)\rVert^2 = \underbrace{\mathbb{E}\lVert v\_t(X\_t) - (X\_1 - X\_0)\rVert^2}\_{\text{constant}} + \mathbb{E}\lVert v\_t^\theta(X\_t) - v\_t(X\_t)\rVert^2 \tag{10}$$

### Conclusion

By (10), minimizing the left side (label is the sampleable $X\_1-X\_0$) is equivalent to minimizing the second term on the right (approximating $v\_t$):

$$\arg\min\_\theta \, \mathbb{E}\lVert v\_t^\theta(X\_t) - (X\_1 - X\_0)\rVert^2 = \arg\min\_\theta \, \mathbb{E}\lVert v\_t^\theta(X\_t) - v\_t(X\_t)\rVert^2$$

### Conditional Flow Matching (CFM) Loss

Taking expectation over time as well, we obtain the final training objective:

$$\boxed{\mathcal{L}\_{\mathrm{CFM}}(\theta) = \mathbb{E}\_{t \sim \mathcal{U}[0,1]} \, \mathbb{E}\_{X\_0 \sim p\_0,\, X\_1 \sim p\_1}\, \lVert v\_t^\theta((1-t)X\_0 + tX\_1) - (X\_1 - X\_0)\rVert^2} \tag{11}$$

{% note success %}
**Why is this loss fully computable?**
- No need for the value of $v\_t$ — by orthogonality (formula (10)), the regression label is equivalently replaced by the sampleable $X\_1 - X\_0$;
- No need for the marginal density formula of $X\_t$ — the input point $(1-t)X\_0 + tX\_1$ is simply a sample of $X\_t$;
- Monte Carlo estimation only requires: sample $t$ uniformly, sample noise $X\_0 \sim p\_0$, sample data $X\_1 \sim p\_1$.

This is the fundamental reason Flow Matching is "simulation-free" (no ODE integration needed during training).
{% endnote %}

<a href="#note-5" style="text-decoration:none"><sup>[5]</sup></a>

## 5. Summary: Three-Step Logic Chain

1. **Construct auxiliary process**: Define $X\_t = (1-t)X\_0 + tX\_1$. This process depends on $X\_1$ and cannot be used for generation, but it transitions smoothly from $p\_0$ to $p\_1$, providing a reference for learning $v\_t$.

2. **Derive target velocity field**: Require the ODE flow $Z\_t$ to match $X\_t$ in distribution. Differentiate the "equal distribution" condition, apply the tower property, and obtain: $v\_t(x) = \mathbb{E}[X\_1 - X\_0 \mid X\_t = x]$.

3. **Reduce to computable regression**: By the Pythagorean theorem of $L^2$ orthogonal projection, regressing against $v\_t$ (intractable) is equivalent to regressing against $X\_1 - X\_0$ (sampleable), yielding the Flow Matching loss (11).

---

**Notes:**

<a id="note-1"></a>**[1]** **Inference**: Once trained, sample $X\_0 \sim p\_0$, integrate ODE to $t=1$, get $Z\_1$ as output. No $X\_1$ needed.

<a id="note-2"></a>**[2]** The linear interpolation (2) is optimal in the sense of Benamou-Brenier (2000): it minimizes kinetic energy among all paths from $p\_0$ to $p\_1$. See also McCann (1997).

<a id="note-3"></a>**[3]** **Notation**: $\mathbb{E}[\cdot \mid X\_t = x]$ yields a deterministic function of $x$, not a random variable. Both sides of (6) are functions of $x$.

<a id="note-4"></a>**[4]** **Intuition**: Multiple trajectories pass through the same $x$ from different $(X\_0, X\_1)$ pairs. The target velocity at $x$ is their conditional average.

<a id="note-5"></a>**[5]** Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023. Liu et al., *Flow Straight and Fast*, ICLR 2023 (Rectified Flow).

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
      titleEl.textContent = 'Flow Matching 的数学原理：极简介绍';
    } else {
      titleEl.textContent = 'The Mathematics of Flow Matching: A Minimal Introduction';
    }
  }

  // Switch TOC: hide headings from the other language
  var tocLinks = document.querySelectorAll('.post-toc a');
  tocLinks.forEach(function(link) {
    var li = link.closest('li');
    if (!li) return;
    var href = link.getAttribute('href') || '';
    // Chinese headings have Chinese characters or specific patterns
    var isChinese = /[\u4e00-\u9fff]/.test(link.textContent);
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
