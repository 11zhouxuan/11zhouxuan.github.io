---
title: "From Flow Matching to MeanFlow: One-Step Generation"
date: 2025-06-18 10:00:00
tags:
  - flow matching
  - generative models
  - one-step generation
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

从 Flow Matching 到 MeanFlow——用同一分析框架推导一步生成的训练目标。

<!-- more -->

{% note info %}
**前情提要**：在前文中，我们用三步推导了 Flow Matching 的训练目标：(1) 构造辅助过程 $X_t = (1-t)X_0 + tX_1$；(2) 推出目标速度场 $v_t(x) = \mathbb{E}[X_1 - X_0 \mid X_t = x]$；(3) 由 $L^2$ 正交投影得到可计算的 CFM 损失。

**本文问题**：Flow Matching 训练的是**瞬时速度场** $v_t$——采样时需要多步积分 ODE。能否直接学习一个场，使得**一步**就能精确从噪声到达数据？

**本文逻辑**：**定义平均速度** → **推导 MeanFlow 恒等式** → **构造可训练损失**。仍然三步到位。

**符号约定**：与前文完全一致——$X_0 \sim p_0$ 为源分布（噪声），$X_1 \sim p_1$ 为目标分布（数据），$X_t = (1-t)X_0 + tX_1$，ODE 从 $t=0$（噪声端）流向 $t=1$（数据端）。
{% endnote %}

## 1. 设定：与 Flow Matching 完全相同

MeanFlow 的基本设定与 Flow Matching **完全相同**，不引入任何新的对象：

- **两个随机变量**：$X_0 \sim p_0$（源/噪声，如 $\mathcal{N}(0,I)$），$X_1 \sim p_1$（目标/数据）。
- **辅助过程**：$X_t = (1-t)X_0 + tX_1$，$t \in [0,1]$。条件速度 $\frac{\mathrm{d}}{\mathrm{d}t}X_t = X_1 - X_0$。
- **ODE**：$\frac{\mathrm{d}}{\mathrm{d}t}Z_t = v_t(Z_t)$，$Z_0 = X_0$。
- **边际速度场**：$v_t(x) = \mathbb{E}[X_1 - X_0 \mid X_t = x]$（前文已推导）。

**唯一的新概念**是下面将定义的"平均速度场" $u$——它是从已有的瞬时速度场 $v_t$ 通过积分自然导出的量，**不需要**任何额外假设。MeanFlow 的全部工作，是围绕 $u$ 展开一套新的训练方案，使得一步即可完成推理。

## 2. 动机：为什么需要 MeanFlow？

### 2.1 Flow Matching 的瓶颈：多步采样

回顾 Flow Matching 的推理过程：给定训练好的速度场 $v_t^\theta$，从 $Z_0 = X_0 \sim p_0$ 出发，数值积分 ODE

$$\frac{\mathrm{d}}{\mathrm{d}t} Z_t = v_t(Z_t), \qquad Z_0 = X_0 \tag{1}$$

到 $t=1$ 得到生成样本 $Z_1$。

即使每条 conditional flow（给定 $(X_0, X_1)$ 对的轨迹 $X_t = (1-t)X_0 + tX_1$）是直线，**边际速度场**

$$v_t(x) = \mathbb{E}[X_1 - X_0 \mid X_t = x]$$

（对所有可能的 $(X_0, X_1)$ 取条件平均后的速度场）通常是**弯曲**的。这意味着粗糙的欧拉法——比如一步估计 $Z_1 \approx Z_0 + v_0(Z_0)$——会产生很大的误差。实践中需要 100–250 步数值积分才能得到高质量的生成结果。

**核心矛盾**：Flow Matching 的**训练**是高效的（simulation-free），但**推理**是昂贵的（多步 ODE 积分）。

### 2.2 已有方法的不足：Consistency Models

为了实现一步生成，Consistency Models (CM) 引入了一个**一致性约束**：要求网络在同一 ODE 轨迹上的不同时间点输出相同的终点。这是一个施加在**网络行为**上的约束，而非基于底层场的数学性质。

{% note warning %}
**CM 的困难**：
- 一致性约束对应的 ground-truth field 性质**未知**——我们不清楚满足该约束的"理想解"长什么样；
- 训练不稳定，需要精心设计的"离散化课程"（discretization curriculum）来逐步收紧约束；
- 依赖于网络自身的输出作为目标（self-referential），容易产生误差累积。
{% endnote %}

### 2.3 MeanFlow 的出发点

MeanFlow 的设计动机可以概括为一个问题：

{% note info %}
**核心问题**：能否找到一个有**明确数学定义**的 ground-truth field，使得：
1. 它天然支持一步生成（知道该场在一个点的值，就能一步到位）；
2. 它与瞬时速度场 $v_t$ 之间有**精确的数学关系**（不是启发式约束）；
3. 这个关系可以转化为**可训练的损失函数**（不需要积分或模拟）。
{% endnote %}

MeanFlow 的回答是：**平均速度场**（average velocity field）。

回想物理中最基本的概念：位移 = 平均速度 × 时间。如果我们定义"平均速度"为 ODE 轨迹上一段时间内的平均运动速率，那么：

$$Z_1 = Z_0 + u(Z_0, 0, 1) \cdot (1 - 0) = Z_0 + u(Z_0, 0, 1)$$

**精确**成立（而非近似）——这只是对 ODE 积分的改写。知道了 $u(Z_0, 0, 1)$，一步即可完成生成。

更关键的是，这个平均速度场 $u$ 与瞬时速度场 $v$ 之间存在一个由**微积分基本定理**导出的精确恒等式（MeanFlow Identity），这个恒等式不依赖于任何神经网络——它为训练提供了原则性的数学基础。

## 3. 第一步：定义平均速度

对 ODE (1)，从时刻 $s$ 到 $t$（$s < t$）的位移可写为

$$Z_t - Z_s = \int_s^t v_\tau(Z_\tau)\,\mathrm{d}\tau.$$

我们定义**平均速度**（average velocity）为位移除以时间间隔：

$$\boxed{u(Z_s, s, t) \;:=\; \frac{1}{t - s} \int_s^t v_\tau(Z_\tau)\,\mathrm{d}\tau} \tag{2}$$

于是位移关系可简洁地写为：

$$Z_t = Z_s + (t-s)\,u(Z_s, s, t) \tag{3}$$

{% note success %}
**关键性质**：
- 公式 (3) 是**精确**的，不是近似——它只是对 ODE 积分的改写。
- 当 $t \to s$ 时，$u(Z_s, s, t) \to v_s(Z_s)$——平均速度退化为瞬时速度。
- $u$ 是一个关于 $(Z_s, s, t)$ 的场，由底层瞬时速度场 $v$ 唯一确定。
- 一步生成：令 $s=0, t=1$，则 $Z_1 = Z_0 + u(Z_0, 0, 1)$。
{% endnote %}

**记号说明**：$u(Z_s, s, t)$ 中三个变量的含义为：第一个变量 $Z_s$ 是出发点（空间位置）；第二个变量 $s$ 是出发时刻；第三个变量 $t$ 是到达时刻。平均速度描述的是"从 $Z_s$ 出发，经过时段 $[s, t]$，沿 ODE 轨迹的平均运动速度"。

**问题转化**：如果我们能用神经网络 $u_\theta(z, s, t)$ 准确逼近真实的平均速度场 $u$，就能实现一步（或少步）生成。但直接计算 $u$ 需要沿轨迹做积分——这正是我们想避免的。**核心问题**：能否找到一个**不需要积分**的训练目标？

## 4. 第二步：推导 MeanFlow 恒等式

为了推导方便，我们等价地把平均速度写成"以终点为参数"的形式<a href="#note-1" style="text-decoration:none"><sup>[1]</sup></a>：

$$(t - s)\,u(Z_t, s, t) = \int_s^t v_\tau(Z_\tau)\,\mathrm{d}\tau \tag{4}$$

对 (4) 两边关于 $t$ 求导（$s$ 视为与 $t$ 无关的常数）：

**左边**——乘法法则：

$$\frac{\mathrm{d}}{\mathrm{d}t}\Big[(t-s)\,u(Z_t, s, t)\Big] = u(Z_t, s, t) + (t-s)\,\frac{\mathrm{d}}{\mathrm{d}t}\,u(Z_t, s, t)$$

**右边**——微积分基本定理（上限对 $t$ 求导）：

$$\frac{\mathrm{d}}{\mathrm{d}t}\int_s^t v_\tau(Z_\tau)\,\mathrm{d}\tau = v_t(Z_t)$$

**联立**，令左 = 右，整理得到 **MeanFlow 恒等式**：

$$\boxed{u(Z_t, s, t) = v_t(Z_t) - (t-s)\,\frac{\mathrm{d}}{\mathrm{d}t}\,u(Z_t, s, t)} \tag{5}$$

{% note warning %}
**验证**：当 $s = t$ 时，$(t-s) = 0$，恒等式退化为 $u = v_t$——平均速度等于瞬时速度。这与定义一致。
{% endnote %}

### 展开全导数

注意 $\frac{\mathrm{d}}{\mathrm{d}t}$ 是沿 ODE 轨迹的**全导数**（total derivative）。$u$ 的变量为 $(Z_t, s, t)$，其中 $Z_t$ 本身依赖 $t$（通过 ODE），而 $s$ 不依赖 $t$。用链式法则：

$$\frac{\mathrm{d}}{\mathrm{d}t}\,u(Z_t, s, t) = \underbrace{\frac{\mathrm{d}Z_t}{\mathrm{d}t}}\_{= v_t(Z_t)} \cdot \partial_z u + \underbrace{\frac{\mathrm{d}s}{\mathrm{d}t}}\_{= 0} \cdot \partial_s u + \underbrace{\frac{\mathrm{d}t}{\mathrm{d}t}}\_{= 1} \cdot \partial_t u = v_t(Z_t) \cdot \partial_z u + \partial_t u \tag{6}$$

将 (6) 代入 (5)，得到完全展开的形式：

$$u(Z_t, s, t) = v_t(Z_t) - (t-s)\big[v_t(Z_t) \cdot \partial_z u + \partial_t u\big] \tag{7}$$

{% note info %}
**MeanFlow 恒等式的本质**：它是微积分基本定理的直接推论。左边是"平均速度"，右边用"瞬时速度"加上"平均速度随时间的变化率"来表达。这个恒等式**不依赖**任何神经网络——它是 $u$ 和 $v$ 之间的内在数学关系。
{% endnote %}

## 5. 第三步：构造可训练的损失函数

### 构造回归目标

将恒等式 (7) 视为"$u$ 等于某个目标"的形式，用神经网络 $u_\theta$ 替代右边的 $u$（的导数部分），构造**有效回归目标**：

$$u_{\mathrm{tgt}} := v_t(Z_t) - (t-s)\,\big[v_t(Z_t) \cdot \partial_z u_\theta + \partial_t u_\theta\big] \tag{8}$$

### 损失函数

$$\boxed{\mathcal{L}(\theta) = \mathbb{E}\_{s,t}\;\mathbb{E}\_{X_0, X_1}\;\Big\|u_\theta(X_t, s, t) - \mathrm{sg}\big(u_{\mathrm{tgt}}\big)\Big\|^2} \tag{9}$$

其中 $\mathrm{sg}(\cdot)$ 表示 stop-gradient（将目标视为常数，不对其反向传播）。

### 关键：瞬时速度的替换

恒等式中的 $v_t(Z_t)$ 是**边际速度场**——即 $\mathbb{E}[X_1 - X_0 \mid X_t = Z_t]$（回顾前文第二步的结论）。与标准 Flow Matching 完全相同的技巧：我们用**条件速度**（可采样的 $X_1 - X_0$，对应一个具体的 $(X_0, X_1)$ 对）代替边际速度。

在默认的线性插值下 $X_t = (1-t)X_0 + tX_1$，条件速度为 $X_1 - X_0$。于是最终的回归目标为：

$$u_{\mathrm{tgt}} = (X_1 - X_0) - (t-s)\,\big[(X_1 - X_0) \cdot \partial_z u_\theta + \partial_t u_\theta\big] \tag{10}$$

{% note success %}
**为什么可以用条件速度代替边际速度？**

这与前文 Flow Matching 的推导完全一致：给定 $X_t$，条件速度 $X_1 - X_0$ 的条件期望就是边际速度 $v_t(X_t) = \mathbb{E}[X_1 - X_0 \mid X_t]$。由 $L^2$ 正交性，用条件速度作为目标与用边际速度作为目标，在最小化意义下等价。

更直观地说：对每个训练样本 $(X_0, X_1)$，我们知道通过 $X_t$ 的这条具体轨迹的速度是 $X_1 - X_0$。虽然每次只看到一个方向，但在大量样本上取平均后，效果等价于对边际速度做回归。
{% endnote %}

### JVP 的含义与高效计算

目标 (10) 中的关键计算量是全导数

$$\frac{\mathrm{d}}{\mathrm{d}t}\,u_\theta = (X_1 - X_0) \cdot \partial_z u_\theta + \partial_t u_\theta$$

**什么是 Jacobian-Vector Product (JVP)？** 设 $F: \mathbb{R}^n \to \mathbb{R}^m$ 是一个可微映射，它在输入 $x$ 处的 Jacobian 矩阵为 $J_F(x) \in \mathbb{R}^{m \times n}$。给定一个"切向量" $v \in \mathbb{R}^n$，**JVP** 定义为矩阵-向量乘积 $J_F(x) \cdot v \in \mathbb{R}^m$。直观上，JVP 回答的问题是：*当输入沿方向 $v$ 微扰时，输出如何变化？*

在我们的场景中，$u_\theta$ 的输入为 $(z, s, t) \in \mathbb{R}^{d+2}$，它的 Jacobian 为

$$J_{u_\theta} = \big[\,\partial_z u_\theta \;\big|\; \partial_s u_\theta \;\big|\; \partial_t u_\theta\,\big] \in \mathbb{R}^{d \times (d+2)}$$

我们需要计算的全导数恰好是这个 Jacobian 与切向量 $(X_1 - X_0,\; 0,\; 1) \in \mathbb{R}^{d+2}$ 的乘积：

$$J_{u_\theta} \cdot \begin{pmatrix} X_1 - X_0 \\\ 0 \\\ 1 \end{pmatrix} = (X_1 - X_0) \cdot \partial_z u_\theta + 0 \cdot \partial_s u_\theta + 1 \cdot \partial_t u_\theta = \frac{\mathrm{d}}{\mathrm{d}t}\,u_\theta$$

这正是"沿 ODE 轨迹方向的方向导数"——切向量的三个分量分别对应 $\frac{\mathrm{d}z}{\mathrm{d}t} = v_t = X_1 - X_0$、$\frac{\mathrm{d}s}{\mathrm{d}t} = 0$、$\frac{\mathrm{d}t}{\mathrm{d}t} = 1$。

**计算效率**：在现代自动微分框架中（如 PyTorch 的 `torch.func.jvp` 或 JAX 的 `jax.jvp`），JVP 通过**前向模式自动微分**实现——只需一次额外的前向传播，代价与标准的反向传播相当。无需显式构造整个 Jacobian 矩阵（$d$ 可能高达数万维）。

由于 $u_{\mathrm{tgt}}$ 被 stop-gradient 包裹，这次 JVP 计算不会引入高阶导数——神经网络参数 $\theta$ 的梯度更新只涉及标准的一阶反向传播。

## 6. 采样：一步生成

训练完成后，一步生成极其简单：

1. 采样 $X_0 \sim p_0$（如标准高斯）；
2. 计算 $Z_1 = X_0 + u_\theta(X_0, 0, 1)$。

这就是全部。不需要任何 ODE 积分。

{% note success %}
**对比 Flow Matching 采样**：FM 需要从 $Z_0 = X_0$ 出发，多步积分 ODE（如 100–250 步欧拉法）才能到达 $Z_1$。MeanFlow 只需**一次**前向传播：$Z_1 = X_0 + u_\theta(X_0, 0, 1)$。
{% endnote %}

## 7. 与 Flow Matching 的统一视角

{% note info %}
**MeanFlow 是 Flow Matching 的推广**：
- 当 $s = t$ 时：$(t-s) = 0$，MeanFlow Identity 退化为 $u = v_t$，损失退化为标准 CFM。
- 当 $s \neq t$ 时：额外的 $(t-s) \cdot \frac{\mathrm{d}}{\mathrm{d}t}u$ 项提供了"跨时间"的信息传播，使网络学习到的不仅是局部切方向，还有全局位移方向。

因此，MeanFlow 的训练可以看作**标准 FM 加上一个 JVP 修正项**：

$$u_{\mathrm{tgt}} = \underbrace{(X_1 - X_0)}\_{\text{FM 目标}} - \underbrace{(t-s)\big[(X_1 - X_0) \cdot \partial_z u_\theta + \partial_t u_\theta\big]}\_{\text{MeanFlow 修正}}$$

实践中，训练时随机采样一定比例的 $s \neq t$ 样本（如 25%），其余 $s = t$ 的样本相当于做标准 FM 训练。
{% endnote %}

## 8. 总结：三步逻辑链

1. **定义平均速度**：$u(Z_t, s, t) := \frac{1}{t-s}\int_s^t v_\tau(Z_\tau)\,\mathrm{d}\tau$。它将 ODE 积分的位移浓缩为一个场——知道了 $u(X_0, 0, 1)$，一步就能从噪声到数据。

2. **推导恒等式**：对定义式两边关于 $t$ 求导（微积分基本定理），得到 $u = v_t - (t-s)\,\frac{\mathrm{d}}{\mathrm{d}t}u$。这是 $u$ 和 $v$ 之间的内在关系，不依赖任何神经网络。

3. **构造损失**：将恒等式转为回归目标，用条件速度 $X_1 - X_0$ 代替边际速度（与 FM 相同的 $L^2$ 正交性技巧），并用 stop-gradient + JVP 实现高效训练。

{% note warning %}
**与 Consistency Models 的区别**：Consistency Models 通过**约束网络行为**（不同时间步的输出一致）来实现一步生成，需要精心设计的离散化课程。MeanFlow 则基于**底层场的数学恒等式**——$u$ 和 $v$ 的关系由微积分基本定理自然给出，不依赖于网络。这使得训练更稳定，不需要课程学习。
{% endnote %}

---

**注释：**

<a id="note-1"></a>**[1]** 以起点或终点为参照是等价的：知道 $Z_s$ 和 $u$，可以由 $Z_t = Z_s + (t-s)u$ 得到 $Z_t$；反之亦然。MeanFlow 原文采用以终点为参照的写法。两种参数化描述的是同一个物理量。

**参考**：Geng et al., *Mean Flows for One-step Generative Modeling*, arXiv:2505.13447, 2025.

</div>

<!-- English Version -->
<div class="lang-content lang-en">

From Flow Matching to MeanFlow — deriving the training objective for one-step generation using the same analytical framework.

{% note info %}
**Previously**: We derived the Flow Matching training objective in three steps: (1) construct auxiliary process $X_t = (1-t)X_0 + tX_1$; (2) derive target velocity field $v_t(x) = \mathbb{E}[X_1 - X_0 \mid X_t = x]$; (3) reduce to computable CFM loss via $L^2$ orthogonal projection.

**This post**: Flow Matching learns the **instantaneous velocity field** $v_t$ — sampling requires multi-step ODE integration. Can we learn a field that enables **one-step** exact generation from noise to data?

**Logic**: **Define average velocity** → **Derive MeanFlow Identity** → **Construct trainable loss**. Still three steps.

**Notation**: Same as before — $X_0 \sim p_0$ (source/noise), $X_1 \sim p_1$ (target/data), $X_t = (1-t)X_0 + tX_1$, ODE flows from $t=0$ (noise) to $t=1$ (data).
{% endnote %}

## 1. Setting: Identical to Flow Matching

MeanFlow's basic setting is **identical** to Flow Matching — no new objects are introduced:

- **Two random variables**: $X_0 \sim p_0$ (source/noise, e.g., $\mathcal{N}(0,I)$), $X_1 \sim p_1$ (target/data).
- **Auxiliary process**: $X_t = (1-t)X_0 + tX_1$, $t \in [0,1]$. Conditional velocity $\frac{\mathrm{d}}{\mathrm{d}t}X_t = X_1 - X_0$.
- **ODE**: $\frac{\mathrm{d}}{\mathrm{d}t}Z_t = v_t(Z_t)$, $Z_0 = X_0$.
- **Marginal velocity field**: $v_t(x) = \mathbb{E}[X_1 - X_0 \mid X_t = x]$ (derived in previous post).

The **only new concept** is the "average velocity field" $u$ defined below — naturally derived from the existing instantaneous velocity $v_t$ via integration, requiring **no** additional assumptions. MeanFlow's entire contribution is building a new training scheme around $u$ that enables one-step inference.

## 2. Motivation: Why Do We Need MeanFlow?

### 2.1 Flow Matching Bottleneck: Multi-Step Sampling

Recall Flow Matching inference: given trained $v_t^\theta$, start from $Z_0 = X_0 \sim p_0$, numerically integrate ODE

$$\frac{\mathrm{d}}{\mathrm{d}t} Z_t = v_t(Z_t), \qquad Z_0 = X_0 \tag{1}$$

to $t=1$ to obtain generated sample $Z_1$.

Even though each conditional flow (trajectory for a given $(X_0, X_1)$ pair) is a straight line, the **marginal velocity field**

$$v_t(x) = \mathbb{E}[X_1 - X_0 \mid X_t = x]$$

is typically **curved**. This means coarse Euler methods — e.g., one-step estimate $Z_1 \approx Z_0 + v_0(Z_0)$ — incur large errors. In practice, 100–250 integration steps are needed for high-quality generation.

**Core tension**: Flow Matching **training** is efficient (simulation-free), but **inference** is expensive (multi-step ODE integration).

### 2.2 Limitations of Existing Approaches: Consistency Models

To achieve one-step generation, Consistency Models (CM) introduce a **consistency constraint**: requiring the network to output the same endpoint for inputs at different time steps along the same ODE trajectory. This is a constraint imposed on **network behavior**, not based on mathematical properties of the underlying field.

{% note warning %}
**CM difficulties**:
- The ground-truth field corresponding to the consistency constraint is **unknown** — we don't know what the "ideal solution" looks like;
- Training is unstable, requiring carefully designed "discretization curriculum" to progressively tighten constraints;
- Relies on the network's own output as target (self-referential), prone to error accumulation.
{% endnote %}

### 2.3 MeanFlow's Starting Point

MeanFlow's design motivation can be summarized as one question:

{% note info %}
**Core question**: Can we find a ground-truth field with a **clear mathematical definition** such that:
1. It naturally supports one-step generation (knowing the field value at one point suffices);
2. It has an **exact mathematical relationship** with instantaneous velocity $v_t$ (not a heuristic constraint);
3. This relationship can be converted to a **trainable loss function** (no integration or simulation needed).
{% endnote %}

MeanFlow's answer: the **average velocity field**.

Recall the most basic concept in physics: displacement = average velocity × time. If we define "average velocity" as the mean speed along an ODE trajectory over a time interval, then:

$$Z_1 = Z_0 + u(Z_0, 0, 1) \cdot (1 - 0) = Z_0 + u(Z_0, 0, 1)$$

holds **exactly** (not approximately) — this is merely a rewriting of ODE integration. Knowing $u(Z_0, 0, 1)$, one step completes generation.

More crucially, this average velocity field $u$ and instantaneous velocity $v$ are related by an exact identity derived from the **Fundamental Theorem of Calculus** (the MeanFlow Identity), independent of any neural network — providing a principled mathematical foundation for training.

## 3. Step One: Define Average Velocity

For ODE (1), displacement from time $s$ to $t$ ($s < t$) is:

$$Z_t - Z_s = \int_s^t v_\tau(Z_\tau)\,\mathrm{d}\tau$$

We define **average velocity** as displacement divided by time interval:

$$\boxed{u(Z_s, s, t) \;:=\; \frac{1}{t - s} \int_s^t v_\tau(Z_\tau)\,\mathrm{d}\tau} \tag{2}$$

The displacement relation becomes:

$$Z_t = Z_s + (t-s)\,u(Z_s, s, t) \tag{3}$$

{% note success %}
**Key properties**:
- Formula (3) is **exact**, not an approximation — it's just a rewriting of ODE integration.
- As $t \to s$: $u(Z_s, s, t) \to v_s(Z_s)$ — average velocity reduces to instantaneous velocity.
- $u$ is a field depending on $(Z_s, s, t)$, uniquely determined by the underlying $v$.
- One-step generation: set $s=0, t=1$, then $Z_1 = Z_0 + u(Z_0, 0, 1)$.
{% endnote %}

**Notation**: In $u(Z_s, s, t)$: first argument $Z_s$ is the starting point (spatial position); second argument $s$ is departure time; third argument $t$ is arrival time. It describes "the average speed along the ODE trajectory from $Z_s$ over the interval $[s, t]$."

**Problem reformulation**: If we can approximate $u$ with a neural network $u_\theta(z, s, t)$, we achieve one-step generation. But computing $u$ directly requires trajectory integration — exactly what we want to avoid. **Key question**: Can we find a training objective that **doesn't require integration**?

## 4. Step Two: Derive the MeanFlow Identity

For convenience, we write average velocity parameterized by the endpoint<a href="#note-1-en" style="text-decoration:none"><sup>[1]</sup></a>:

$$(t - s)\,u(Z_t, s, t) = \int_s^t v_\tau(Z_\tau)\,\mathrm{d}\tau \tag{4}$$

Differentiate both sides of (4) with respect to $t$ (treating $s$ as independent of $t$):

**Left side** — product rule:

$$\frac{\mathrm{d}}{\mathrm{d}t}\Big[(t-s)\,u(Z_t, s, t)\Big] = u(Z_t, s, t) + (t-s)\,\frac{\mathrm{d}}{\mathrm{d}t}\,u(Z_t, s, t)$$

**Right side** — Fundamental Theorem of Calculus (differentiating upper limit):

$$\frac{\mathrm{d}}{\mathrm{d}t}\int_s^t v_\tau(Z_\tau)\,\mathrm{d}\tau = v_t(Z_t)$$

**Equating** left = right and rearranging, we obtain the **MeanFlow Identity**:

$$\boxed{u(Z_t, s, t) = v_t(Z_t) - (t-s)\,\frac{\mathrm{d}}{\mathrm{d}t}\,u(Z_t, s, t)} \tag{5}$$

{% note warning %}
**Verification**: When $s = t$, $(t-s) = 0$, and the identity reduces to $u = v_t$ — average velocity equals instantaneous velocity. Consistent with the definition.
{% endnote %}

### Expanding the Total Derivative

Note $\frac{\mathrm{d}}{\mathrm{d}t}$ is the **total derivative** along the ODE trajectory. Since $u$ depends on $(Z_t, s, t)$ where $Z_t$ itself depends on $t$ (via ODE) while $s$ doesn't, the chain rule gives:

$$\frac{\mathrm{d}}{\mathrm{d}t}\,u(Z_t, s, t) = \underbrace{\frac{\mathrm{d}Z_t}{\mathrm{d}t}}\_{= v_t(Z_t)} \cdot \partial_z u + \underbrace{\frac{\mathrm{d}s}{\mathrm{d}t}}\_{= 0} \cdot \partial_s u + \underbrace{\frac{\mathrm{d}t}{\mathrm{d}t}}\_{= 1} \cdot \partial_t u = v_t(Z_t) \cdot \partial_z u + \partial_t u \tag{6}$$

Substituting (6) into (5), the fully expanded form is:

$$u(Z_t, s, t) = v_t(Z_t) - (t-s)\big[v_t(Z_t) \cdot \partial_z u + \partial_t u\big] \tag{7}$$

{% note info %}
**Essence of the MeanFlow Identity**: It's a direct consequence of the Fundamental Theorem of Calculus. The left side is "average velocity"; the right side expresses it via "instantaneous velocity" plus "rate of change of average velocity". This identity is **independent** of any neural network — it's an intrinsic mathematical relationship between $u$ and $v$.
{% endnote %}

## 5. Step Three: Construct the Trainable Loss

### Constructing the Regression Target

Viewing identity (7) as "$u$ equals some target," we replace $u$ (its derivatives) on the right with the neural network $u_\theta$ to construct the **effective regression target**:

$$u_{\mathrm{tgt}} := v_t(Z_t) - (t-s)\,\big[v_t(Z_t) \cdot \partial_z u_\theta + \partial_t u_\theta\big] \tag{8}$$

### Loss Function

$$\boxed{\mathcal{L}(\theta) = \mathbb{E}\_{s,t}\;\mathbb{E}\_{X_0, X_1}\;\Big\|u_\theta(X_t, s, t) - \mathrm{sg}\big(u_{\mathrm{tgt}}\big)\Big\|^2} \tag{9}$$

where $\mathrm{sg}(\cdot)$ denotes stop-gradient (treating the target as a constant, no backpropagation through it).

### Key: Replacing Instantaneous Velocity

The $v_t(Z_t)$ in the identity is the **marginal velocity field** — i.e., $\mathbb{E}[X_1 - X_0 \mid X_t = Z_t]$ (from our previous derivation). Using the exact same trick as standard Flow Matching: replace **marginal velocity** with the sampleable **conditional velocity** $X_1 - X_0$.

Under linear interpolation $X_t = (1-t)X_0 + tX_1$, the conditional velocity is $X_1 - X_0$. The final regression target becomes:

$$u_{\mathrm{tgt}} = (X_1 - X_0) - (t-s)\,\big[(X_1 - X_0) \cdot \partial_z u_\theta + \partial_t u_\theta\big] \tag{10}$$

{% note success %}
**Why can we replace marginal with conditional velocity?**

Exactly as in our Flow Matching derivation: given $X_t$, the conditional expectation of $X_1 - X_0$ is the marginal velocity $v_t(X_t) = \mathbb{E}[X_1 - X_0 \mid X_t]$. By $L^2$ orthogonality, using conditional velocity as target is equivalent to using marginal velocity for minimization purposes.

Intuitively: for each training sample $(X_0, X_1)$, we know the velocity along this specific trajectory through $X_t$ is $X_1 - X_0$. Though we only see one direction each time, averaging over many samples is equivalent to regressing against the marginal velocity.
{% endnote %}

### JVP: Meaning and Efficient Computation

The key computation in target (10) is the total derivative:

$$\frac{\mathrm{d}}{\mathrm{d}t}\,u_\theta = (X_1 - X_0) \cdot \partial_z u_\theta + \partial_t u_\theta$$

**What is a Jacobian-Vector Product (JVP)?** Given a differentiable map $F: \mathbb{R}^n \to \mathbb{R}^m$ with Jacobian $J_F(x) \in \mathbb{R}^{m \times n}$ at input $x$, and a "tangent vector" $v \in \mathbb{R}^n$, the **JVP** is the matrix-vector product $J_F(x) \cdot v \in \mathbb{R}^m$. Intuitively, JVP answers: *how does the output change when the input is perturbed along direction $v$?*

In our setting, $u_\theta$ takes input $(z, s, t) \in \mathbb{R}^{d+2}$ with Jacobian:

$$J_{u_\theta} = \big[\,\partial_z u_\theta \;\big|\; \partial_s u_\theta \;\big|\; \partial_t u_\theta\,\big] \in \mathbb{R}^{d \times (d+2)}$$

The total derivative we need is exactly this Jacobian times tangent vector $(X_1 - X_0,\; 0,\; 1) \in \mathbb{R}^{d+2}$:

$$J_{u_\theta} \cdot \begin{pmatrix} X_1 - X_0 \\\ 0 \\\ 1 \end{pmatrix} = (X_1 - X_0) \cdot \partial_z u_\theta + 0 \cdot \partial_s u_\theta + 1 \cdot \partial_t u_\theta = \frac{\mathrm{d}}{\mathrm{d}t}\,u_\theta$$

This is the "directional derivative along the ODE trajectory" — tangent components correspond to $\frac{\mathrm{d}z}{\mathrm{d}t} = v_t = X_1 - X_0$, $\frac{\mathrm{d}s}{\mathrm{d}t} = 0$, $\frac{\mathrm{d}t}{\mathrm{d}t} = 1$.

**Computational efficiency**: In modern autodiff frameworks (PyTorch `torch.func.jvp` or JAX `jax.jvp`), JVP is computed via **forward-mode automatic differentiation** — requiring only one extra forward pass, comparable in cost to standard backpropagation. No need to explicitly construct the full Jacobian matrix ($d$ can be tens of thousands).

Since $u_{\mathrm{tgt}}$ is wrapped in stop-gradient, this JVP computation introduces no higher-order derivatives — the gradient update for $\theta$ involves only standard first-order backpropagation.

## 6. Sampling: One-Step Generation

After training, one-step generation is extremely simple:

1. Sample $X_0 \sim p_0$ (e.g., standard Gaussian);
2. Compute $Z_1 = X_0 + u_\theta(X_0, 0, 1)$.

That's it. No ODE integration needed.

{% note success %}
**Comparison with Flow Matching sampling**: FM requires starting from $Z_0 = X_0$ and integrating ODE for many steps (e.g., 100–250 Euler steps) to reach $Z_1$. MeanFlow needs only **one** forward pass: $Z_1 = X_0 + u_\theta(X_0, 0, 1)$.
{% endnote %}

## 7. Unified View with Flow Matching

{% note info %}
**MeanFlow generalizes Flow Matching**:
- When $s = t$: $(t-s) = 0$, MeanFlow Identity reduces to $u = v_t$, loss reduces to standard CFM.
- When $s \neq t$: the extra $(t-s) \cdot \frac{\mathrm{d}}{\mathrm{d}t}u$ term provides "cross-time" information propagation, enabling the network to learn not just local tangent directions but global displacement directions.

Thus, MeanFlow training is **standard FM plus a JVP correction**:

$$u_{\mathrm{tgt}} = \underbrace{(X_1 - X_0)}\_{\text{FM target}} - \underbrace{(t-s)\big[(X_1 - X_0) \cdot \partial_z u_\theta + \partial_t u_\theta\big]}\_{\text{MeanFlow correction}}$$

In practice, a fraction (e.g., 25%) of training samples use $s \neq t$; the rest use $s = t$, equivalent to standard FM training.
{% endnote %}

## 8. Summary: Three-Step Logic Chain

1. **Define average velocity**: $u(Z_t, s, t) := \frac{1}{t-s}\int_s^t v_\tau(Z_\tau)\,\mathrm{d}\tau$. It condenses ODE integration into a single field — knowing $u(X_0, 0, 1)$, one step goes from noise to data.

2. **Derive the identity**: Differentiate the definition with respect to $t$ (Fundamental Theorem of Calculus) to get $u = v_t - (t-s)\,\frac{\mathrm{d}}{\mathrm{d}t}u$. An intrinsic relation between $u$ and $v$, independent of any neural network.

3. **Construct loss**: Convert identity to regression target, replace marginal velocity with conditional velocity $X_1 - X_0$ (same $L^2$ orthogonality trick as FM), and use stop-gradient + JVP for efficient training.

{% note warning %}
**Difference from Consistency Models**: CM achieves one-step generation by **constraining network behavior** (output consistency across time steps), requiring carefully designed discretization curricula. MeanFlow is based on a **mathematical identity of the underlying fields** — the relationship between $u$ and $v$ is naturally given by the Fundamental Theorem of Calculus, independent of the network. This leads to more stable training without curriculum learning.
{% endnote %}

---

**Notes:**

<a id="note-1-en"></a>**[1]** Parameterizing by starting point or endpoint is equivalent: knowing $Z_s$ and $u$, we get $Z_t = Z_s + (t-s)u$; and vice versa. The MeanFlow paper uses endpoint parameterization. Both describe the same physical quantity.

**Reference**: Geng et al., *Mean Flows for One-step Generative Modeling*, arXiv:2505.13447, 2025.

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
      titleEl.textContent = '从 Flow Matching 到 MeanFlow：一步生成的数学原理';
    } else {
      titleEl.textContent = 'From Flow Matching to MeanFlow: The Mathematics of One-Step Generation';
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
