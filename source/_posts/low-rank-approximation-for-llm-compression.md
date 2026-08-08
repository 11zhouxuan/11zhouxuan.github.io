---
title: "Low-Rank Matrix Approximation: From SVD to Weighted SVD"
date: 2026-08-08
mathjax: true
tags: [math, linear-algebra, SVD, approximation-theory]
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 矩阵低秩逼近：从 SVD 到加权 SVD

### 1. 问题设定

给定矩阵 $W \in \mathbb{R}^{m \times n}$，要找一个秩不超过 $r$ 的矩阵 $\hat W$ 使得 $\hat W \approx W$。

一种参数化方式：取 $A \in \mathbb{R}^{m \times r}$，$B \in \mathbb{R}^{r \times n}$。由矩阵乘法的秩不等式 $\text{rank}(AB) \le \min(\text{rank}(A), \text{rank}(B)) \le r$，$AB$ 的秩自然不超过 $r$。此时存储量从 $mn$ 降到 $(m+n)r$。

**核心问题**：什么是"$\approx$"？不同的度量方式导致不同的最优解。

### 2. Frobenius 范数下的最优逼近（Plain SVD）

#### 2.1 目标

$$\min_{\text{rank}(\hat W) \le r} \lVert W - \hat W\rVert_F^2 = \min_{\text{rank}(\hat W) \le r} \sum_{j=1}^m \sum_{i=1}^n (W_{ji} - \hat W_{ji})^2$$

所有矩阵元素的误差等权相加。

#### 2.2 奇异值分解（SVD）

任何矩阵 $W$ 可以分解为：

$$W = U \Sigma V^T = \sum_{k=1}^{\min(m,n)} \sigma_k  u_k v_k^T$$

其中：
- $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ 为奇异值
- $u_k \in \mathbb{R}^m$ 为左奇异向量（两两正交）
- $v_k \in \mathbb{R}^n$ 为右奇异向量（两两正交）

每一项 $\sigma_k u_k v_k^T$ 是一个秩-1 矩阵，$W$ 被表示为秩-1 矩阵的有序加权和。

#### 2.3 Eckart-Young 定理

**定理**：Frobenius 范数下的最优 rank-$r$ 逼近是截断 SVD：

$$\hat W^* = \sum_{k=1}^r \sigma_k  u_k v_k^T = U_r \Sigma_r V_r^T$$

最优误差：

$$\lVert W - \hat W^*\rVert_F^2 = \sum_{k=r+1}^{\min(m,n)} \sigma_k^2$$

即被丢掉的尾部奇异值的平方和。

**证明思路**：利用 Frobenius 范数的酉不变性 $\lVert W\rVert_F = \lVert U^T W V\rVert_F$，将问题变为在对角矩阵 $\Sigma$ 上选最优的 rank-$r$ 逼近，而对角矩阵的最优截断显然是保留最大的 $r$ 个对角元。

#### 2.4 分解为两个因子

取：

$$A = U_r \sqrt{\Sigma_r} \in \mathbb{R}^{m \times r}, \quad B = \sqrt{\Sigma_r}  V_r^T \in \mathbb{R}^{r \times n}$$

则 $AB = U_r \Sigma_r V_r^T = \hat W^*$。将奇异值的平方根对称地分配到两个因子中。

计算 $\hat W x = A(Bx)$：先做 $\mathbb{R}^n \to \mathbb{R}^r$（降维），再做 $\mathbb{R}^r \to \mathbb{R}^m$（升维）。

#### 2.5 局限

Plain SVD 对所有元素等权——它只关心"矩阵本身的近似程度"，不关心"这个矩阵在使用时哪些方向更重要"。当矩阵被用于 $y = Wx$ 而 $x$ 的分布不均匀时，这不是最优的选择。

### 3. 加权 Frobenius 范数下的最优逼近（Weighted SVD）

#### 3.1 动机

如果我们关心的是**输出误差**而不是权重误差：

$$\min_{\text{rank}(\hat W) \le r} \mathbb{E}_{x \sim \mu}\big[\lVert Wx - \hat Wx\rVert^2\big]$$

其中 $\mu$ 是输入 $x$ 的分布。

#### 3.2 展开

$$\mathbb{E}\lVert(W - \hat W)x\rVert^2 = \text{tr}\big((W - \hat W) \Sigma (W - \hat W)^T\big)$$

其中 $\Sigma = \mathbb{E}[xx^T] \in \mathbb{R}^{n \times n}$ 是输入的二阶矩矩阵。

**推导**：

$$\mathbb{E}\lVert(W-\hat W)x\rVert^2 = \mathbb{E}\big[x^T(W-\hat W)^T(W-\hat W)x\big] = \text{tr}\big((W-\hat W)^T(W-\hat W) \mathbb{E}[xx^T]\big)$$

利用 $\text{tr}(ABC) = \text{tr}(CAB)$，即得上式。

#### 3.3 变量代换与求解

设 $\Sigma = SS^T$（$S$ 可以是 $\Sigma^{1/2}$ 或 Cholesky 因子），则：

$$\text{tr}\big((W-\hat W) \Sigma (W-\hat W)^T\big) = \lVert(W - \hat W)S\rVert_F^2$$

令 $\tilde W = WS$，问题变为：

$$\min_{\text{rank}(\hat W) \le r} \lVert\tilde W - \hat W S\rVert_F^2$$

由于 $\hat W S$ 的秩 $\le r$（秩不因右乘满秩矩阵而增加），且任何 rank-$r$ 矩阵都可写为 $\hat W S$ 的形式（取 $\hat W = M S^{-1}$），问题等价于：

$$\min_{\text{rank}(M) \le r} \lVert\tilde W - M\rVert_F^2$$

这正是 $\tilde W$ 在 Frobenius 下的最优 rank-$r$ 逼近，由 Eckart-Young 直接解出。

**求解步骤**：

1. 计算 $\tilde W = WS$
2. 对 $\tilde W$ 做 SVD：$\tilde W = U\tilde\Sigma V^T$
3. 截断前 $r$ 项：$\tilde W_r = U_r \tilde\Sigma_r V_r^T$
4. 还原：$A = U_r \sqrt{\tilde\Sigma_r}$，$B = \sqrt{\tilde\Sigma_r}  V_r^T S^{-1}$

验证：$(W - AB)S = WS - U_r\tilde\Sigma_r V_r^T = \tilde W - \tilde W_r$，正是 SVD 截断残差，Eckart-Young 最优。

### 4. $S$ 的两种选择

#### 4.1 全协方差（Whiten）

取 $S = \Sigma^{1/2}$ 或 $S = \text{chol}(\Sigma)$。

此时 $\lVert(W-\hat W)S\rVert_F^2 = \mathbb{E}\lVert Wx - \hat Wx\rVert^2$ **精确成立**。

- 需要估计完整的 $n \times n$ 协方差矩阵
- 需要 Cholesky 分解（$O(n^3)$）
- 理论上是给定二阶统计量下的最优解

#### 4.2 对角近似（Diag）

取 $S = \text{diag}(s_1, \ldots, s_n)$，其中 $s_i$ 反映第 $i$ 个输入分量的"重要程度"。

典型选择：$s_i = (\mathbb{E}[|x_i|])^\alpha$，$\alpha > 0$。

此时加权 Frobenius 范数为：

$$\lVert(W - \hat W)\text{diag}(s)\rVert_F^2 = \sum_{j,i} (W_{ji} - \hat W_{ji})^2 s_i^2$$

第 $i$ 列的逼近误差被赋予权重 $s_i^2$：$s_i$ 大的列（对应"活跃"的输入分量）更被重视。

- 只需一个 $n$ 维向量
- 实际操作就是"把 $W$ 的第 $i$ 列乘以 $s_i$，做 SVD，再除回去"
- 是 whiten 的对角近似——忽略输入各分量间的相关性

**$\alpha$ 的作用**：

- $\alpha = 0$：退化为 plain SVD（不加权）
- $\alpha \to \infty$：只关注最活跃的那一个分量
- 中间值在"利用分布信息"和"过度集中"之间权衡

### 5. 三者的统一视角

三种方法在同一个框架下，区别仅在于对输入分布的建模精细程度：

| 方法 | 目标 | 对 $\Sigma$ 的假设 |
|---|---|---|
| Plain SVD | $\min\lVert W - \hat W\rVert_F^2$ | $\Sigma = I$（各向同性） |
| Diag SVD | $\min\lVert(W-\hat W)\text{diag}(s)\rVert_F^2$ | $\Sigma$ 是对角的（各分量独立） |
| Whiten SVD | $\min\lVert(W-\hat W)\Sigma^{1/2}\rVert_F^2$ | 完整的 $\Sigma$ |

三者都由 Eckart-Young 定理给出闭式最优解——区别只在"对什么做 SVD"：

- Plain：对 $W$ 本身
- Diag：对 $W \cdot \text{diag}(s)$
- Whiten：对 $W \cdot \Sigma^{1/2}$

### 6. 理论极限：Kolmogorov 宽度

给定加权矩阵 $\tilde W = WS$，最优 rank-$r$ 逼近的误差为：

$$\inf_{\text{rank}(\hat W) \le r} \lVert(W - \hat W)S\rVert_F^2 = \sum_{k=r+1}^{\min(m,n)} \sigma_k^2(\tilde W)$$

这是 **Kolmogorov $r$-宽度**的矩阵版本——它给出了一个**与算法无关的下界**。

含义：如果 $\tilde W$ 的奇异值衰减快（谱集中），低秩逼近效果好；如果衰减慢（谱均匀分散），任何方法在该加权度量下都无法用少量秩给出好的逼近。这是问题本身的**固有难度**，而非算法的不足。

### 7. 讨论：单矩阵逼近 vs 复合映射逼近

上面的所有结果针对的是**单个矩阵**的最优逼近。在实际系统中，多个矩阵串联使用：$F = W_L \circ \cdots \circ W_1$。此时一个自然的问题是：

每个 $W_\ell$ 各自最优 $\quad\overset{?}{\Longrightarrow}\quad$ 复合映射 $F$ 最优？

答案是**否定的**。原因是复合映射的误差具有跨层交互结构：

$$F - \tilde F = \sum_\ell (\partial F_{>\ell}) \varepsilon_\ell + O(\varepsilon^2)$$

其中 $\varepsilon_\ell = W_\ell - \hat W_\ell$ 是第 $\ell$ 层的逼近误差，$\partial F_{>\ell}$ 是下游映射对该误差的放大。各层误差之间可能存在**相消**（内积为负），使得端到端误差小于各层误差之和。逐矩阵优化不能利用这个自由度。

要最小化复合映射的误差，需要对所有因子联合优化——这超出了闭式解的范围，通常需要迭代优化方法。

</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none">

## Low-Rank Matrix Approximation: From SVD to Weighted SVD

### 1. Problem Setup

Given a matrix $W \in \mathbb{R}^{m \times n}$, find a matrix $\hat W$ with rank at most $r$ such that $\hat W \approx W$.

One parameterization: take $A \in \mathbb{R}^{m \times r}$, $B \in \mathbb{R}^{r \times n}$. By the rank inequality $\text{rank}(AB) \le \min(\text{rank}(A), \text{rank}(B)) \le r$, the product $AB$ has rank at most $r$ automatically. Storage drops from $mn$ to $(m+n)r$.

**Core question**: What does "$\approx$" mean? Different metrics lead to different optimal solutions.

### 2. Optimal Approximation Under the Frobenius Norm (Plain SVD)

#### 2.1 Objective

$$\min_{\text{rank}(\hat W) \le r} \lVert W - \hat W\rVert_F^2 = \min_{\text{rank}(\hat W) \le r} \sum_{j=1}^m \sum_{i=1}^n (W_{ji} - \hat W_{ji})^2$$

All matrix entries are weighted equally.

#### 2.2 Singular Value Decomposition (SVD)

Any matrix $W$ can be decomposed as:

$$W = U \Sigma V^T = \sum_{k=1}^{\min(m,n)} \sigma_k  u_k v_k^T$$

where:
- $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ are the singular values
- $u_k \in \mathbb{R}^m$ are left singular vectors (mutually orthogonal)
- $v_k \in \mathbb{R}^n$ are right singular vectors (mutually orthogonal)

Each term $\sigma_k u_k v_k^T$ is a rank-1 matrix; $W$ is expressed as an ordered weighted sum of rank-1 components.

#### 2.3 The Eckart-Young Theorem

**Theorem**: The optimal rank-$r$ approximation under the Frobenius norm is the truncated SVD:

$$\hat W^* = \sum_{k=1}^r \sigma_k  u_k v_k^T = U_r \Sigma_r V_r^T$$

Optimal error:

$$\lVert W - \hat W^*\rVert_F^2 = \sum_{k=r+1}^{\min(m,n)} \sigma_k^2$$

The sum of squared singular values that were discarded.

**Proof sketch**: Using the unitary invariance of the Frobenius norm ($\lVert W\rVert_F = \|U^TWV\rVert_F$), the problem reduces to finding the best rank-$r$ approximation of the diagonal matrix $\Sigma$, which is obviously to keep the $r$ largest diagonal entries.

#### 2.4 Factored Form

Take:

$$A = U_r \sqrt{\Sigma_r} \in \mathbb{R}^{m \times r}, \quad B = \sqrt{\Sigma_r}  V_r^T \in \mathbb{R}^{r \times n}$$

Then $AB = U_r \Sigma_r V_r^T = \hat W^*$. The square root of the singular values is distributed symmetrically between the two factors.

Computing $\hat W x = A(Bx)$: first $\mathbb{R}^n \to \mathbb{R}^r$ (dimension reduction), then $\mathbb{R}^r \to \mathbb{R}^m$ (expansion).

#### 2.5 Limitation

Plain SVD weights all entries equally — it only cares about "how well the matrix itself is approximated," not "which directions matter more when the matrix is actually used." When the matrix is used in $y = Wx$ and the distribution of $x$ is non-uniform, this is suboptimal.

### 3. Optimal Approximation Under Weighted Frobenius Norm (Weighted SVD)

#### 3.1 Motivation

If we care about **output error** rather than weight error:

$$\min_{\text{rank}(\hat W) \le r} \mathbb{E}_{x \sim \mu}\big[\lVert Wx - \hat Wx\rVert^2\big]$$

where $\mu$ is the distribution of the input $x$.

#### 3.2 Expansion

$$\mathbb{E}\lVert(W - \hat W)x\rVert^2 = \text{tr}\big((W - \hat W) \Sigma (W - \hat W)^T\big)$$

where $\Sigma = \mathbb{E}[xx^T] \in \mathbb{R}^{n \times n}$ is the second moment matrix of the input.

**Derivation**:

$$\mathbb{E}\lVert(W-\hat W)x\rVert^2 = \mathbb{E}\big[x^T(W-\hat W)^T(W-\hat W)x\big] = \text{tr}\big((W-\hat W)^T(W-\hat W) \mathbb{E}[xx^T]\big)$$

Using $\text{tr}(ABC) = \text{tr}(CAB)$, the result follows.

#### 3.3 Change of Variables and Solution

Let $\Sigma = SS^T$ ($S$ can be $\Sigma^{1/2}$ or a Cholesky factor). Then:

$$\text{tr}\big((W-\hat W) \Sigma (W-\hat W)^T\big) = \lVert(W - \hat W)S\rVert_F^2$$

Setting $\tilde W = WS$, the problem becomes:

$$\min_{\text{rank}(\hat W) \le r} \lVert\tilde W - \hat W S\rVert_F^2$$

Since the rank of $\hat WS$ is $\le r$ (rank does not increase by right-multiplication by a full-rank matrix), and any rank-$r$ matrix can be written as $\hat WS$ (take $\hat W = MS^{-1}$), the problem is equivalent to:

$$\min_{\text{rank}(M) \le r} \lVert\tilde W - M\rVert_F^2$$

This is precisely the optimal rank-$r$ approximation of $\tilde W$ under Frobenius, solved by Eckart-Young.

**Solution steps**:

1. Compute $\tilde W = WS$
2. SVD of $\tilde W$: $\tilde W = U\tilde\Sigma V^T$
3. Truncate to rank $r$: $\tilde W_r = U_r \tilde\Sigma_r V_r^T$
4. Recover factors: $A = U_r \sqrt{\tilde\Sigma_r}$, $B = \sqrt{\tilde\Sigma_r}  V_r^T S^{-1}$

Verification: $(W - AB)S = WS - U_r\tilde\Sigma_r V_r^T = \tilde W - \tilde W_r$, which is the SVD truncation residual — Eckart-Young optimal.

### 4. Two Choices of $S$

#### 4.1 Full Covariance (Whiten)

Take $S = \Sigma^{1/2}$ or $S = \text{chol}(\Sigma)$.

Then $\lVert(W-\hat W)S\rVert_F^2 = \mathbb{E}\lVert Wx - \hat Wx\rVert^2$ holds **exactly**.

- Requires estimating the full $n \times n$ covariance matrix
- Requires Cholesky decomposition ($O(n^3)$)
- Theoretically optimal given second-order statistics

#### 4.2 Diagonal Approximation (Diag)

Take $S = \text{diag}(s_1, \ldots, s_n)$, where $s_i$ reflects the "importance" of the $i$-th input component.

Typical choice: $s_i = (\mathbb{E}[|x_i|])^\alpha$, $\alpha > 0$.

The weighted Frobenius norm becomes:

$$\lVert(W - \hat W)\text{diag}(s)\rVert_F^2 = \sum_{j,i} (W_{ji} - \hat W_{ji})^2 s_i^2$$

Column $i$'s approximation error is weighted by $s_i^2$: columns corresponding to "active" input components are prioritized.

- Only requires an $n$-dimensional vector
- In practice: "multiply column $i$ of $W$ by $s_i$, do SVD, divide back"
- A diagonal approximation of the whiten approach — ignores correlations between input components

**Role of $\alpha$**:

- $\alpha = 0$: reduces to plain SVD (no weighting)
- $\alpha \to \infty$: concentrates entirely on the most active component
- Intermediate values trade off between "using distribution information" and "over-concentrating"

### 5. Unified View of the Three Methods

All three methods live in the same framework, differing only in how finely they model the input distribution:

| Method | Objective | Assumption on $\Sigma$ |
|---|---|---|
| Plain SVD | $\min\lVert W - \hat W\rVert_F^2$ | $\Sigma = I$ (isotropic) |
| Diag SVD | $\min\lVert(W-\hat W)\text{diag}(s)\rVert_F^2$ | $\Sigma$ is diagonal (independent components) |
| Whiten SVD | $\min\lVert(W-\hat W)\Sigma^{1/2}\rVert_F^2$ | Full $\Sigma$ |

All three are solved in closed form by the Eckart-Young theorem — the only difference is "what matrix to apply SVD to":

- Plain: $W$ itself
- Diag: $W \cdot \text{diag}(s)$
- Whiten: $W \cdot \Sigma^{1/2}$

### 6. Theoretical Limits: Kolmogorov Width

Given the weighted matrix $\tilde W = WS$, the optimal rank-$r$ error is:

$$\inf_{\text{rank}(\hat W) \le r} \lVert(W - \hat W)S\rVert_F^2 = \sum_{k=r+1}^{\min(m,n)} \sigma_k^2(\tilde W)$$

This is the matrix version of the **Kolmogorov $r$-width** — an **algorithm-independent lower bound**.

Implication: if $\tilde W$'s singular values decay rapidly (concentrated spectrum), low-rank approximation works well. If they decay slowly (uniformly spread spectrum), no method can achieve a good approximation at that rank under that weighted metric. This is the **intrinsic hardness** of the problem, not a shortcoming of the algorithm.

### 7. Discussion: Per-Matrix vs Composite Approximation

All results above concern the optimal approximation of a **single matrix**. In practice, multiple matrices are used in composition: $F = W_L \circ \cdots \circ W_1$. A natural question is:

Each $W_\ell$ individually optimal $\quad\overset{?}{\Longrightarrow}\quad$ composite $F$ optimal?

The answer is **no**. The reason is that the composite error has cross-layer interaction structure:

$$F - \tilde F = \sum_\ell (\partial F_{>\ell}) \varepsilon_\ell + O(\varepsilon^2)$$

where $\varepsilon_\ell = W_\ell - \hat W_\ell$ is the approximation error at layer $\ell$, and $\partial F_{>\ell}$ is the downstream amplification. Error vectors from different layers may partially **cancel** (negative inner products), making the end-to-end error smaller than the sum of per-layer errors. Per-matrix optimization cannot exploit this degree of freedom.

To minimize the composite error, one must jointly optimize all factors — which goes beyond closed-form solutions and typically requires iterative optimization.

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
