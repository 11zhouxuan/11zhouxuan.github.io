---
title: "ACL 2026 Scaling Laws: A Deep Dive into 10 Papers"
date: 2026-07-09 10:00:00
tags:
  - scaling laws
  - LLM
  - pre-training
  - ACL 2026
  - NLP
categories:
  - research
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
**概览**：ACL 2026 共有 10 篇以 Scaling Law 为核心研究主题的论文，覆盖了预训练数据选择、代码模型、多模态SFT、RAG强化学习、推理时计算、量化压缩等多个维度。本文对每篇论文进行深度解读，分析研究动机、方法论和关键结论。
{% endnote %}

## 总览：Scaling Law 研究的新边界

经典 Scaling Law（Kaplan 2020, Chinchilla 2022）建立了模型参数 $N$、训练数据量 $D$ 与性能之间的幂律关系。但在 2026 年，研究者们发现这种简单关系在很多新场景下不再成立——代码训练比自然语言更"数据饥渴"、多模态SFT需要同时平衡三个维度、RAG系统受限于检索质量瓶颈、推理模型存在真实性逆缩放效应。这10篇论文共同勾画了 Scaling Law 研究从"单一幂律"走向"多维度、任务感知、约束优化"的新范式。

---

## 1. Perplexity-Aware Data Scaling Law

**论文**: Perplexity-Aware Data Scaling Law: Perplexity Landscapes Predict Performance for Continual Pre-training

**作者**: Lei Liu, Hao Zhu, Xiaoyan Yang, Yue Shen, Jian Wang, Jinjie Gu, Zhixuan Chu, Kui Ren (浙江大学, 蚂蚁集团)

**Track**: Main Conference | [PDF](https://aclanthology.org/2026.acl-long.999.pdf)

### 研究动机

继续预训练(CPT)是将通用LLM适配到特定领域的主流范式。经典Scaling Law假设"每个token对学习的贡献相同"，但在CPT中这一假设严重失效——领域语料中充斥着冗余内容（重复已知事实的生物医学文献）和噪声（非结构化的临床笔记），简单增大数据量会快速遇到收益递减。核心问题是：**在基模型已具备大量知识的前提下，如何量化每个训练样本的信息价值？**

### 核心方法

作者提出利用模型自身的**困惑度(Perplexity)**作为数据信息量的代理信号：低PPL的序列对模型而言是冗余的（已知知识），高PPL的序列可能是噪声或不可理解的内容，最有效的训练数据处于"适中困惑度"的甜蜜区间。

**Perplexity-Aware Data Scaling Law** 的最终形式为：

$$\hat{L}(\mu, \sigma, D) = E + \frac{D_c}{\mu^{\alpha_\mu(\sigma)} \cdot \sigma^{\alpha_\sigma(\mu)} \cdot D^{\alpha_D}}$$

其中 $\mu$ 和 $\sigma$ 分别是数据子集PPL分布的均值和方差，关键创新在于引入了交互项：

$$\alpha_\mu(\sigma) = \alpha_0 + \alpha_1\sigma, \quad \alpha_\sigma(\mu) = \beta_0 + \beta_1\mu$$

这捕捉了一个重要观察：PPL均值和方差对损失的影响**不是独立的**——低均值数据从高方差中受益更多，而高均值数据在方差过高时反而受损。

**Distance-to-Optimum Selection (DOS)** 算法：拟合scaling law后识别最优的 $(\hat{\mu}, \hat{\sigma}^2)$，然后贪心地选择使数据子集统计量逼近最优点的chunks：

$$J(\mathcal{S}) = w_\mu(\mu(\mathcal{S}) - \hat{\mu})^2 + w_\sigma(\sigma^2(\mathcal{S}) - \hat{\sigma}^2)^2$$

整个流程仅需一次前向传播即可完成。

### 关键结论

- 在 Qwen3-14B + PubMed 医学CPT实验中，DOS-CPT 达到 72.48 平均分，超越随机采样(71.34)和低PPL采样(71.22)
- 在 DeepSeek-V3 上同样有效（75.26 vs 74.24 RS-CPT）
- **不会导致灾难性遗忘**：通用能力保持甚至略有提升（84.43 vs 84.16 baseline）
- PPL景观形成碗形3D曲面（均值×方差→损失），多条梯度下降路径收敛到同一最优点
- 实际意义：**做CPT时不要盲目堆数据，用一次前向传播评估PPL分布就能指导最优数据选择**

---

## 2. Scaling Laws for Code: A More Data-Hungry Regime

**论文**: Scaling Laws for Code: A More Data-Hungry Regime

**作者**: Xianzhen Luo, Wenzhen Zheng, Qingfu Zhu 等 (哈尔滨工业大学, 中科院, 复旦大学)

**Track**: Main Conference | [PDF](https://aclanthology.org/2026.acl-long.1101.pdf)

### 研究动机

NL的Scaling Law（Chinchilla, Farseer）是否适用于代码？代码具有与自然语言根本不同的统计性质——严格语法、复杂长程依赖、独特词汇分布、高重复性。现有Code LLM（OpenCoder, Qwen2.5-Coder, StarCoder等）大多≤32B参数，其D/N比显著偏离NL最优预测。这是因为代码模型scaling更快饱和，还是数据不够？

### 核心方法

**117次实验**，模型规模0.2B-3.8B，训练token 2B-128B，log均匀采样，总计约13,600 H100 GPU天。

拟合两种Scaling Law：

**Chinchilla法则**:
$$L(N, D) = 0.2193 + \frac{534.374}{N^{0.4853}} + \frac{76.0743}{D^{0.2983}}$$

**Farseer法则**（指数本身是N的函数）:
$$L(N,D) = \exp(s \cdot N^q + S) + \exp(B \cdot N^b + Q) \cdot D^{-\exp(A \cdot N^a + E)}$$

另外进行了234次额外实验测试代码与NL的混合训练效果。

### 关键结论

- **代码比NL更"数据饥渴"**: 计算最优的D/N比远高于NL，且随算力预算增加超线性增长。在 $C = 5.36 \times 10^{21}$ FLOPs下，最优D/N=150（NL约为20）
- **Farseer优于Chinchilla**: 在最优点相对误差仅 1.17‰（vs Chinchilla 33.67‰）
- **代码scaling未饱和**: Farseer预测不可约损失为零，代码的损失曲面在大规模下仍保持陡峭下降梯度
- **NL混合的效果反转**: 低算力时加NL有帮助（正则化作用），高算力（N>0.94B或D/N>300）时纯代码更优
- **为何Code LLM普遍较小**: 不是因为scaling饱和，而是因为高质量代码数据不足以最优训练更大模型。瓶颈在数据而非模型规模

---

## 3. Scaling Laws for Code: Every Programming Language Matters

**论文**: Scaling Laws for Code: Every Programming Language Matters

**作者**: Jian Yang, Shuyue Guo, Linzheng Chai 等 (北航, Renmin University)

**Track**: Findings | [PDF](https://aclanthology.org/2026.findings-acl.487.pdf)

### 研究动机

现代代码LLM训练使用多语言代码（Python, Java, TypeScript等），但现有Scaling Law将训练数据视为同质的，忽略了不同编程语言具有本质不同的scaling行为。问题是：**每种语言的scaling参数是否不同？语言间存在哪些协同/拮抗效应？如何最优分配跨语言的token预算？**

### 核心方法

**1000+实验**（等效336,000+ H800 GPU小时），对7种语言分别拟合独立Scaling Law。

**核心贡献**:

1. **语言特定Scaling Law**: 每种语言有独立的 $(\alpha_N, \alpha_D, L_\infty)$
2. **协同增益矩阵**: $\Delta(L_i, L_j) = L(L_i + L_j) - L(L_i + L_i)$，量化28种双语组合的相互影响
3. **比例依赖的多语言Scaling Law**:
$$L(N, D; \mathbf{p}) = A \cdot N^{-\alpha_N(\mathbf{p})} + B \cdot D_x^{-\alpha_D(\mathbf{p})} + L_\infty(\mathbf{p})$$

其中有效数据量包含跨语言迁移系数：$D_x = D_{all} \cdot (1 + \gamma \sum_{L_i \neq L_j} p_{L_i} p_{L_j} \tau_{ij})$

### 关键结论

- **解释型语言从scale中受益更多**: Python的 $\alpha_N$ 和 $\alpha_D$ 最大；Rust因严格类型系统更容易学习
- **Java受益于混合最多**: C#(+20.58%), JavaScript(+12.62%), TypeScript(+12.08%)的协同效应显著
- **Python地位特殊**: 帮助别人但自己受损——只有Java对Python有轻微正向协同(+1.36%)
- **并行配对数据显著提升跨语言迁移**: 模型使用Python作为隐式桥接语言（Java→Python→Go）
- **优化分配一致优于均匀分配**: 在1.5B模型上，优化分配的Pass@1平均25.00 vs 均匀23.17

---

## 4. Scaling Law for Multimodal Large Language Model Supervised Fine-Tuning

**论文**: Scaling Law for Multimodal Large Language Model Supervised Fine-Tuning

**作者**: YiFan Zhang, Tao Yu, Feng Li 等 (CASIA, UCAS, HKUST, NTU, NJU, Meta AI)

**Track**: Main Conference | [PDF](https://aclanthology.org/2026.acl-long.603.pdf)

### 研究动机

MLLM的SFT阶段至关重要，但没有Scaling Law指导最优模型-数据配置。与LLM预训练不同，MLLM SFT受更多因素影响：模型大小 $N$、预训练token $D_{\text{pretrain}}$、SFT token $D_{\text{SFT}}$。直接求解类似Chinchilla的计算最优前沿（$N \propto D^{0.5}$）在多模态场景下极其困难。

### 核心方法

提出两种互补的Scaling Law范式：

**From-Scratch Scaling Law**（训练数据量已知时）:
$$P(N, D_{\text{pretrain}}, D_{\text{SFT}}) = A - \frac{B}{N^\alpha} - \frac{C}{D_{\text{pretrain}}^\beta} - \frac{E}{D_{\text{SFT}}^\gamma}$$

拟合参数: $A=256.76, B=143.75, C=288.56, E=96.17, \alpha=0.039, \beta=0.054, \gamma=0.074$

**Pre-Trained Model Scaling Law**（使用开源模型，预训练数据未知时）:
$$P(N, P_{\text{base}}, D_{\text{SFT}}) = F \cdot P_{\text{base}} - \frac{G}{N^\delta} - \frac{H}{D_{\text{SFT}}^\zeta}$$

其中 $P_{\text{base}} = w_1 P_{\text{NLI}}^{k_1} + w_2 P_{\text{Commonsense}}^{k_2} + w_3 P_{\text{Reasoning}}^{k_3}$

**训练Loss与下游性能的关系**:
$$P(L) = P_{\text{min}} + \frac{P_{\text{max}} - P_{\text{min}}}{1 + k \cdot L^\gamma}$$

构建了60个模型（50M-8B），1560个checkpoint，在10+多模态基准上评估。

### 关键结论

- **最优资源分配**: 1B模型应配20.2B预训练token + 9.2B SFT token。$D_{\text{SFT}} \approx 0.48 \times D_{\text{pretrain}}^{0.98}$（近线性关系）
- **LLM底座能力主导**: 预训练LLM的基准性能 $P_{\text{base}}$ 对下游性能的贡献**远大于**模型大小或SFT数据量
- **常识推理影响最大**: LLM的常识推理能力对MLLM性能影响最大，NLI影响较小
- **任务特异性**: OCR任务强依赖SFT数据量（$H=146.2$），感知任务更依赖模型规模（$\delta=0.13$）
- **Loss-Performance强相关**: $R^2=0.98$，可用于early stopping和资源重分配

---

## 5. The Retrieval Bottleneck: Scaling Laws for Reinforcement Learning in RAG

**论文**: The Retrieval Bottleneck: Scaling Laws for Reinforcement Learning in RAG

**作者**: Shu Zhou, Jinman Leng, Yufei Song, Xin Wang, Tao Fan, Hao Wang (南京大学, 百度)

**Track**: Main Conference | [PDF](https://aclanthology.org/2026.acl-long.1478.pdf)

### 研究动机

Scaling Law已扩展到预训练和推理RL，但RAG系统的RL scaling行为完全未被研究。RAG引入了标准RL不具备的三个复杂性：(1)**信息瓶颈**——检索质量硬性约束了性能上限，无限RL计算也无法突破；(2)**多组件交互**——retriever和generator有不同的学习动态；(3)**信用分配复杂性**——错误可能来自检索、生成或两者交互。

### 核心方法

**Retrieval Bottleneck Hypothesis**:
$$A_{\text{RAG}} = \min(A_{\text{ret}}, A_{\text{gen}}) \cdot \eta$$

其中 $A_{\text{ret}}$ 是检索天花板，$A_{\text{gen}}$ 是生成天花板，$\eta \in (0,1]$ 是信息传递效率。实验验证 $A_{\text{gen}}$(91-97% EM) >> $A_{\text{ret}}$(61-89%)，确认**检索是约束瓶颈**。

**RAG-RL Scaling Law**:
$$R = R_0 + (A_{\text{eff}} - R_0) \cdot f(C_{\text{RL}})$$

其中 $f(C_{\text{RL}}) = \frac{1}{1 + (C_{\text{mid}}/C_{\text{RL}})^B}$ 是S形计算缩放项，$A_{\text{eff}} = A_{\text{RAG}} \cdot g(k, L)$。

总计约150,000 GPU小时实验。

### 关键结论

- **三大原则**: (1)检索质量决定性能上限——提升检索比算法创新收益更大；(2)设计选择调节计算效率（B和C_mid）；(3)稳定配置支持4倍算力外推（3.1%误差）
- **最优k随训练增加**: 8K步时k*=5，32K步时k*=10。检索配置应作为training schedule
- **End-to-End训练在16K步后超越Gen-Only**(+4.2 EM)，但方差更大
- **归因感知奖励关键**: Answer+Attribution+Process组合奖励最优（52.4 EM, 81.2% Attr.Acc.）
- **RAG-SCALE RL系统**: 在HotpotQA/NQ/FEVER上比Self-RAG分别提升+5.0/+5.4/+4.3 EM

---

## 6. a1: Steep Test-time Scaling Law via Environment Augmented Generation

**论文**: a1: Steep Test-time Scaling Law via Environment Augmented Generation

**作者**: Lingrui Mei, Shenghua Liu, Yiwei Wang 等 (中科院, UC Merced)

**Track**: Findings | [PDF](https://aclanthology.org/2026.findings-acl.1240.pdf)

### 研究动机

LLM在推理时仍然存在幻觉、逻辑错误和无法自我修正的问题。现有CoT方法在需要精确逐步验证的任务中失效——模型在一次前向传播中规划整个解决方案，如果早期步骤有误，错误会复合累积。没有统一框架同时结合即时验证、分支探索和自适应学习。

### 核心方法

**Environment Augmented Generation (EAG)**: 将推理形式化为MDP $(S, A, F, T, R)$，核心创新包括：

- **分支价值函数**: $V_B(s) = \lambda_I \cdot D_{KL}(P(f|a,s) \| P_{prior}(f)) + \lambda_P \cdot (t/T) \cdot \mathbb{1}[\text{Success}(f)] + \lambda_C \cdot \mathbb{1}[f \text{ contains errors}]$
- **混合策略**: $\pi_{\text{hybrid}}(a|s) = \alpha \cdot \pi_{LM}(a|s) + (1-\alpha) \cdot \pi_{\text{feedback}}(a|s, f_{<t})$
- 将自然语言推理转换为可执行Python代码+环境反馈

基于Qwen2.5-32B-Instruct在EAG-2K数据集上SFT训练（8×A100, 12小时）。

### 关键结论

- **陡峭scaling特征**: 初始token投入在环境交互上产生长期性能红利。4K-8K token时超越基线s1，32K时优势加速到15pp
- **a1-32B在32B模型中达到SOTA**: AIME24=74.4（匹配o1的>100B参数），MATH500=94.8
- **比s1-32B在AIME24上提升+24.4**，比QwQ-32B-Preview提升+24.4%
- 消融实验：移除分支探索使AIME24下降21.1pp；仅用数值反馈下降17.7pp
- **核心洞察**: 环境交互+系统化分支探索建立了可靠机器推理的新范式

---

## 7. When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning

**论文**: When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning

**作者**: Sitong Fang, Wenjing Cao, Jiahao Li 等 (北京大学, HKUST)

**Track**: Findings | [PDF](https://aclanthology.org/2026.findings-acl.63.pdf)

### 研究动机

推理模型（慢思考、System II范式）被广泛认为能通过更深入思考得到更可靠的答案。但一个关键问题未被回答：**更慢的推理是否必然导致更真实的答案？** 当面对不完整或误导性视觉输入时，慢思考模型是否更容易编造看似合理但虚假的细节？

### 核心方法

**TruthfulVQA基准**: 5000个VQA对，三级层次化提示：
- Level 1: 基本感知（直接视觉识别）
- Level 2: 归纳误导（欺骗性上下文线索）
- Level 3: 虚假前提推理（需要抵抗无效逻辑的虚假叙事）

**Logit Advantage Loss (LAL)**: 分解误导效应为"正确退化"和"错误放大"。

**TruthfulJudge**: 基于Qwen2.5-VL-7B-Instruct微调的专门评估器（88.4%准确率，Cohen's κ=0.79）。

### 关键结论

- **发现真实性逆缩放定律**: 推理模型在真实性上**一致劣于**其chat对应版本。准确率从L1(81.85%)降至L2(55.37%)再到L3(44.96%)
- **DFS vs BFS**: 推理模型遵循深度优先搜索——过早承诺初始解释而不考虑替代方案；Chat模型倾向广度优先搜索
- **校准问题**: 推理模型ECE一致偏高（QVQ-72B: 0.325 vs Qwen2.5-VL-72B: 0.188）
- **CoT降低chat模型真实性**: 对5个chat模型应用CoT，准确率下降2.8-8.3个百分点——证实是序列化推理拓扑而非模型容量导致脆弱性
- **核心洞察**: 推理增强并非万能药——在误导性输入下，更长的思考链可能强化错误前提

---

## 8. Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs

**论文**: Task-Stratified Knowledge Scaling Laws for Post-Training Quantized Large Language Models

**作者**: Chenxi Zhou, Pengfei Cao, Jiang Li 等 (中科院)

**Track**: Findings | [PDF](https://aclanthology.org/2026.findings-acl.1165.pdf)

### 研究动机

训练后量化(PTQ)是LLM部署的关键技术，但对性能影响的理解仍不充分。两个关键缺口：(1)细粒度PTQ因素（组大小、校准集大小）被忽视，当它们实际上对低bit量化下保持能力是决定性的；(2)对多样知识能力的不同影响未被探索——LLM的记忆、应用、推理能力可能对量化展现出不同敏感度。

### 核心方法

**统一乘法幂律形式**:
$$-\ln(\text{Acc}_{adj}) = A_{\text{task}} \cdot N^{\alpha_{\text{task}}} \cdot (\log_2 B)^{\beta_{\text{task}}} \cdot (\log_2 C_b)^{\gamma_{\text{task}}} \cdot G^{\delta_{\text{task}}}$$

其中 $N$=模型大小，$B$=位宽，$C_b$=校准集大小，$G$=组大小。

**三层知识分级**（基于Bloom分类学）：
- L1 知识记忆(KM): TriviaQA, NQ, SQuAD等
- L2 知识应用(KA): HellaSwag, MMLU, ARC-Easy等
- L3 知识推理(KR): GSM8K, StrategyQA, ARC-Challenge等

293种PTQ配置，跨Qwen3(0.6B-14B)和Llama-3(1B-8B)系列。

### 关键结论

- **整体拟合**: Adj.$R^2$=0.9475，验证MAE仅0.0630
- **推理最精度敏感**: $\beta=-1.356$（位宽指数最大），组大小敏感度最高($\delta=0.087$)
- **应用最规模响应**: $\alpha=-0.409$（模型大小指数最大）
- **记忆最校准敏感**: $\gamma=-0.040$（校准集大小指数最大）
- **2-bit相变**: <2B模型普遍崩溃；≥4B模型能维持KM和KA，但**KR完全崩溃**($R^2$≈0.22)
- **跨架构泛化**: 在Llama-3上Adj.$R^2$>0.92，确认普适性

---

## 9. Scaling Laws or Threshold Effects: Optimal Vocabulary Size for Low-Resource Languages

**论文**: Scaling Laws or Threshold Effects: Exploring the Optimal Vocabulary Size for Balancing Performance and Efficiency in Low-Resource Languages

**作者**: Ao Han, Andong Chen, Yuan Sun, Xiaobing Zhao (中央民族大学, 哈工大)

**Track**: Findings | [PDF](https://aclanthology.org/2026.findings-acl.1588.pdf)

### 研究动机

词表扩展的Scaling Law在高资源语言中已有验证，但在低资源语言（蒙古语、藏语、维吾尔语）中完全未知。Byte-level BPE (BBPE)在这些语言中面临严重的过分割问题（token fertility高达英语的10倍）。研究者任意选择词表大小，不知道如何在缓解过分割与避免稀疏token表示不稳定之间取得平衡。

### 核心方法

**联合三语扩展(JTE)**: 蒙/藏/维三语等量分配，10个词表规模（140到195,000 token）。

**多目标Pareto框架**: 四个元目标——质量效用($U_Q$)、公平性指数($I_F$, Rawlsian Maximin原则)、效率成本($C_E$)、资源成本($C_R$)。

$$B(v) = \frac{U_Q(v) + I_F(v)}{2} - \alpha \cdot \frac{C_E(v) + C_R(v)}{2}$$

### 关键结论

- **BPE vs BBPE行为根本不同**: BPE表现为单调递增（词表越大越好，收益递减）；BBPE表现为**U型曲线**——先恶化再恢复
- **性能退化区间**: BBPE在<3,000 token/语言时性能反而低于基线（扰动了预训练的字节级表示）
- **通用拐点**: ~9,000 joint token的拐点在1.5B-8B模型间保持锚定
- **Pareto最优点**: BBPE为79,500 token，BPE为63,000 token
- **效率增益**: 79.5k配置将继续预训练时间减少71%，同时提升下游性能
- **JTE优于独立扩展**: 联合扩展避免了藏语摘要任务中的模型崩溃

---

## 10. MaskTab: Scalable Masked Tabular Pretraining with Scaling Laws

**论文**: MaskTab: Scalable Masked Tabular Pretraining with Scaling Laws and Distillation for Industrial Classification

**作者**: Bo Zheng, Yudong Chen, Zihua Xiong 等 (浙江大学, 蚂蚁集团)

**Track**: Findings | [PDF](https://aclanthology.org/2026.findings-acl.2053.pdf)

### 研究动机

工业表格数据具有高维、大量缺失值和标签稀缺等特点，目前仍以XGBoost等手工特征方法为主。表格数据领域缺乏通用的自监督预训练框架，尤其是在异构类型和系统性缺失的现实条件下，scaling law完全未被探索。

### 核心方法

**MaskTab** 三大设计原则：

1. **可学习缺失值token**: 区分"模型掩码"（用于自监督学习）和"自然缺失"（真实数据问题），共享初始化
2. **双路径混合预训练**: 重建路径（掩码输入）和分类路径（原始输入）共享Transformer参数，避免掩码导致的训练-测试偏移
3. **MoE重建头**: 路由专家处理异构类型的重建

**自适应掩码率**: $r_{\text{mask}}(x) = r_{\max} \cdot (1 - \eta(x)/\eta_{\max})^\alpha$

**知识蒸馏**: $\mathcal{L}_{\text{align}} = 1 - \cos(z_a, e_t)$

### 关键结论

- **TabReD基准**: MaskTab-Base平均排名2.3，超越XGBoost(4.4)和LightGBM(5.5)
- **工业信用风险数据集**(13M+13M, 2500特征, 49%缺失): MaskTab-L在ROC AUC上超XGBoost +5.04%，KS +8.28%
- **明确的Scaling Law**: 沿无标签数据、特征维度、模型参数三个轴均呈现一致的scaling趋势
- **蒸馏**: MaskTab-Distill实现9.3倍推理加速，仅用500个可解释特征
- **OOD鲁棒性**: 显著缩小训练-OOT性能差距

---

## 总结：10篇论文的核心洞察

| 论文 | 核心发现 |
|------|---------|
| Perplexity-Aware | 数据质量(PPL分布)比数据量更重要，适中困惑度是最优区间 |
| Code: Data-Hungry | 代码比NL更需要数据，最优D/N=150远超NL的20 |
| Code: Every PL | 编程语言间存在量化的协同/拮抗效应，需语言感知分配 |
| MLLM SFT | SFT数据≈0.48×预训练数据，LLM底座能力主导最终性能 |
| RAG Retrieval Bottleneck | 检索质量是硬上限，无限RL也无法突破，应先投资检索 |
| a1 Test-time | 环境交互产生陡峭scaling，32B可匹配100B+推理模型 |
| Inverse Truthfulness | 推理增强导致真实性逆缩放——DFS式推理放大错误前提 |
| PTQ Knowledge | 推理→精度敏感，应用→规模响应，记忆→校准敏感 |
| Vocabulary Size | BBPE呈U型阈值效应，BPE呈单调scaling，本质不同 |
| MaskTab | 表格数据存在明确scaling趋势，自监督预训练超越GBDT |

**大趋势**: Scaling Law研究已从单一的"N↑D↑→Performance↑"幂律，演化为多维度、任务感知、约束条件下的精确建模。2026年的核心信息是：**知道在哪里投入资源（数据选择、语言分配、检索质量、位宽分配）比盲目扩大规模重要得多。**

</div>

<!-- English Version -->
<div class="lang-content lang-en">

<!-- more -->

{% note info %}
**Overview**: ACL 2026 features 10 papers with Scaling Laws as their core research topic, spanning pre-training data selection, code models, multimodal SFT, RAG reinforcement learning, test-time compute, quantization, and more. This post provides deep analysis of each paper's motivation, methodology, and key findings.
{% endnote %}

## The New Frontier of Scaling Law Research

Classical scaling laws (Kaplan 2020, Chinchilla 2022) established power-law relationships between model parameters $N$, training data $D$, and performance. But in 2026, researchers discovered that this simple relationship breaks down in many new settings — code training is more "data-hungry" than natural language, multimodal SFT requires balancing three dimensions simultaneously, RAG systems are bounded by retrieval quality, and reasoning models exhibit inverse truthfulness scaling. Together, these 10 papers chart the evolution from "single power law" to "multi-dimensional, task-aware, constrained optimization."

---

## 1. Perplexity-Aware Data Scaling Law

**Paper**: Perplexity-Aware Data Scaling Law: Perplexity Landscapes Predict Performance for Continual Pre-training

**Authors**: Lei Liu, Hao Zhu, Xiaoyan Yang, Yue Shen, Jian Wang, Jinjie Gu, Zhixuan Chu, Kui Ren (Zhejiang University, Ant Group)

**Track**: Main Conference | [PDF](https://aclanthology.org/2026.acl-long.999.pdf)

### Motivation

Continual Pre-Training (CPT) adapts general LLMs to specific domains. Classical scaling laws assume "each token contributes equally to learning," but this assumption fails during CPT — domain corpora contain massive redundancy (biomedical literature restating known facts) and noise (unstructured clinical notes). Simply scaling data volume hits diminishing returns rapidly. The core question: **given a base model with existing knowledge, how do we quantify the informational value of each training sample?**

### Core Method

The authors propose using the model's own **perplexity (PPL)** as a proxy for data informativeness: low-PPL sequences are redundant (already known), high-PPL sequences are likely noise, and the most effective data lies in a "sweet spot" of moderate perplexity.

**Perplexity-Aware Data Scaling Law**:

$$\hat{L}(\mu, \sigma, D) = E + \frac{D_c}{\mu^{\alpha_\mu(\sigma)} \cdot \sigma^{\alpha_\sigma(\mu)} \cdot D^{\alpha_D}}$$

where $\mu$ and $\sigma$ are the mean and variance of the data subset's PPL distribution, with interaction terms:

$$\alpha_\mu(\sigma) = \alpha_0 + \alpha_1\sigma, \quad \alpha_\sigma(\mu) = \beta_0 + \beta_1\mu$$

**Distance-to-Optimum Selection (DOS)**: After fitting the scaling law to identify optimal $(\hat{\mu}, \hat{\sigma}^2)$, greedily select data chunks minimizing:

$$J(\mathcal{S}) = w_\mu(\mu(\mathcal{S}) - \hat{\mu})^2 + w_\sigma(\sigma^2(\mathcal{S}) - \hat{\sigma}^2)^2$$

The entire pipeline requires only a single forward pass over unlabeled data.

### Key Findings

- DOS-CPT achieves 72.48 average on medical benchmarks (vs 71.34 random, 71.22 low-PPL) on Qwen3-14B
- No catastrophic forgetting: general performance maintained at 84.43 (vs 84.16 baseline)
- PPL landscape forms a bowl-shaped 3D surface with a single well-defined minimum
- **Takeaway**: Don't blindly scale data for CPT — one forward pass to assess PPL distribution guides optimal selection

---

## 2. Scaling Laws for Code: A More Data-Hungry Regime

**Authors**: Xianzhen Luo et al. (HIT, CAS, Fudan) | **Track**: Main | [PDF](https://aclanthology.org/2026.acl-long.1101.pdf)

### Motivation

Do NL scaling laws (Chinchilla, Farseer) apply to code? Code has fundamentally different statistical properties — strict syntax, long-range dependencies, high repetitiveness. Most Code LLMs are ≤32B parameters with D/N ratios deviating from NL predictions. Is code scaling saturating faster, or is it a data scarcity issue?

### Core Method

117 experiments (0.2B-3.8B, 2B-128B tokens), ~13,600 H100 GPU-days. Fit both Chinchilla and Farseer laws. Additional 234 experiments for code-NL mixing.

### Key Findings

- **Code is fundamentally more data-hungry**: Optimal D/N=150 at $5.36 \times 10^{21}$ FLOPs (vs ~20 for NL), growing super-linearly with compute
- **Farseer >> Chinchilla for code**: 1.17‰ vs 33.67‰ relative error at optimum
- **Code scaling hasn't saturated**: Farseer predicts zero irreducible loss; loss surface maintains steep gradient at large scale
- **NL mixing effect reverses**: Helpful at low compute (regularization), harmful at high compute (distributional shift beyond N>0.94B)
- **Why Code LLMs are smaller**: Not because scaling saturates, but because insufficient high-quality data to optimally train larger models

---

## 3. Scaling Laws for Code: Every Programming Language Matters

**Authors**: Jian Yang et al. (Beihang, RUC) | **Track**: Findings | [PDF](https://aclanthology.org/2026.findings-acl.487.pdf)

### Key Contribution

1000+ experiments deriving **language-specific scaling parameters** for 7 programming languages, a **synergy gain matrix** quantifying all pairwise interactions, and a **proportion-dependent multilingual scaling law**:

$$L(N, D; \mathbf{p}) = A \cdot N^{-\alpha_N(\mathbf{p})} + B \cdot D_x^{-\alpha_D(\mathbf{p})} + L_\infty(\mathbf{p})$$

### Key Findings

- Interpreted languages (Python) benefit more from scale than compiled languages (Rust)
- Java benefits enormously from mixing: C# (+20.58%), JavaScript (+12.62%)
- Python helps others but is hurt by mixing — only Java provides +1.36% synergy to Python
- Optimized allocation consistently outperforms uniform (Pass@1: 25.00 vs 23.17 at 1.5B)

---

## 4. Scaling Law for Multimodal LLM Supervised Fine-Tuning

**Authors**: YiFan Zhang et al. (CASIA, UCAS, HKUST, NTU, NJU, Meta AI) | **Track**: Main | [PDF](https://aclanthology.org/2026.acl-long.603.pdf)

### Key Contribution

First systematic scaling law framework for MLLM SFT with two paradigms:

**From-Scratch**: $P = A - B/N^\alpha - C/D_{\text{pretrain}}^\beta - E/D_{\text{SFT}}^\gamma$

**Pre-Trained**: $P = F \cdot P_{\text{base}} - G/N^\delta - H/D_{\text{SFT}}^\zeta$

60 models (50M-8B), 1,560 checkpoints, 10+ benchmarks.

### Key Findings

- Optimal $D_{\text{SFT}} \approx 0.48 \times D_{\text{pretrain}}^{0.98}$ (near-linear relationship)
- Pre-trained LLM baseline performance ($P_{\text{base}}$) dominates downstream performance far more than model size or SFT data
- Commonsense reasoning has greatest impact; loss predicts performance with $R^2=0.98$
- Task-specific dynamics: OCR relies on SFT data ($H=146.2$), perception tasks benefit more from model scaling

---

## 5. The Retrieval Bottleneck: Scaling Laws for RL in RAG

**Authors**: Shu Zhou et al. (Nanjing U, Baidu) | **Track**: Main | [PDF](https://aclanthology.org/2026.acl-long.1478.pdf)

### Key Contribution

**Retrieval Bottleneck Hypothesis**: $A_{\text{RAG}} = \min(A_{\text{ret}}, A_{\text{gen}}) \cdot \eta$

First systematic RAG-RL scaling law with ~150,000 GPU-hours of experiments.

### Key Findings

- Retrieval quality hard-bounds performance — improving retrieval yields larger gains than algorithmic innovations
- RAG-RL follows S-shaped curves but with lower efficiency (B: 1.28 vs 1.52) and ~1.8x more compute needed vs reasoning-only RL
- Optimal document count $k^*$ increases with compute budget (schedule, not fixed hyperparameter)
- RAG-SCALE RL achieves +5.0/+5.4/+4.3 EM over Self-RAG on HotpotQA/NQ/FEVER
- **Invest in retrieval first** before scaling RL compute

---

## 6. a1: Steep Test-time Scaling via Environment Augmented Generation

**Authors**: Lingrui Mei et al. (CAS, UC Merced) | **Track**: Findings | [PDF](https://aclanthology.org/2026.findings-acl.1240.pdf)

### Key Contribution

**Environment Augmented Generation (EAG)**: Reasoning as MDP with executable code + environment feedback + branch exploration. Trains a1-32B on just 2K traces (8×A100, 12 hours).

### Key Findings

- **Steep scaling pattern**: Initial token investment in environment interaction yields long-term performance dividends accelerating with task complexity
- a1-32B matches o1 (>100B) on AIME24 (74.4), outperforms s1-32B by +24.4
- Removing branch exploration drops AIME24 by 21.1pp
- Core insight: environment interactivity establishes a new paradigm for reliable machine reasoning

---

## 7. When Slower Isn't Truer: Inverse Scaling Law of Truthfulness

**Authors**: Sitong Fang et al. (PKU, HKUST) | **Track**: Findings | [PDF](https://aclanthology.org/2026.findings-acl.63.pdf)

### Key Contribution

**TruthfulVQA**: 5,000 VQA pairs with 3-tier hierarchical misleading prompts. Discovers **inverse scaling law of truthfulness** in multimodal reasoning.

### Key Findings

- Reasoning models consistently **less truthful** than chat counterparts under misleading visual inputs
- Accuracy drops: L1 (81.85%) → L2 (55.37%) → L3 (44.96%)
- Reasoning models follow DFS topology (premature commitment); chat models use BFS (balanced exploration)
- Applying CoT to chat models degrades accuracy by 2.8-8.3pp — confirming it's the serialized reasoning topology causing vulnerability
- **Longer thinking chains can reinforce false premises rather than correct them**

---

## 8. Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs

**Authors**: Chenxi Zhou et al. (CAS) | **Track**: Findings | [PDF](https://aclanthology.org/2026.findings-acl.1165.pdf)

### Key Contribution

Unified multiplicative power-law integrating model size, bit-width, group size, and calibration set size across three knowledge levels (Memorization, Application, Reasoning).

### Key Findings

- **Reasoning is precision-critical** ($\beta=-1.356$), Application is scale-responsive ($\alpha=-0.409$), Memorization is calibration-sensitive
- 2-bit phase transition: Reasoning collapses completely while Memorization/Application survive
- Cross-architecture generalization: Adj.$R^2$>0.92 on both Qwen3 and Llama-3
- Under low-bit quantization, optimizing fine-grained factors (group size, calibration) is essential, not optional

---

## 9. Scaling Laws or Threshold Effects: Optimal Vocabulary Size

**Authors**: Ao Han et al. (Minzu University, HIT) | **Track**: Findings | [PDF](https://aclanthology.org/2026.findings-acl.1588.pdf)

### Key Contribution

First systematic study of vocabulary expansion scaling for low-resource languages (Mongolian, Tibetan, Uyghur) using multi-objective Pareto optimization.

### Key Findings

- **BPE = monotonic scaling** (larger vocab → better, diminishing returns)
- **BBPE = U-shaped threshold effect** (performance first degrades, then recovers at ~9,000 tokens)
- The 9,000-token inflection point is anchored across model scales (1.5B-8B)
- Pareto optimal: 79,500 tokens (BBPE), 63,000 (BPE)
- Joint trilingual expansion outperforms independent monolingual expansion

---

## 10. MaskTab: Tabular Pretraining with Scaling Laws

**Authors**: Bo Zheng et al. (Zhejiang U, Ant Group) | **Track**: Findings | [PDF](https://aclanthology.org/2026.findings-acl.2053.pdf)

### Key Contribution

Self-supervised pretraining framework for industrial tabular data with learnable missing-value tokens, twin-path hybrid training, and MoE reconstruction heads. Establishes scaling laws along three axes.

### Key Findings

- Outperforms XGBoost on TabReD (avg rank 2.3 vs 4.4) and industrial credit risk (+5.04% AUC, +8.28% KS)
- Clear scaling trends: unlabeled data, feature dimensionality, model capacity all scale predictably
- Distillation achieves 9.3x inference speedup with 500 interpretable features
- **Tabular data admits foundation-model treatment** when its statistical idiosyncrasies are respected

---

## Summary: Key Insights Across 10 Papers

| Paper | Core Finding |
|-------|-------------|
| Perplexity-Aware | Data quality (PPL distribution) matters more than quantity; moderate perplexity is optimal |
| Code: Data-Hungry | Code needs optimal D/N=150, far exceeding NL's ~20 |
| Code: Every PL | Programming languages have quantifiable synergy/antagonism; language-aware allocation needed |
| MLLM SFT | SFT data ≈ 0.48× pretrain data; LLM base capability dominates final performance |
| RAG Bottleneck | Retrieval quality is a hard ceiling; no amount of RL can overcome poor retrieval |
| a1 Test-time | Environment interaction enables steep scaling; 32B matches >100B reasoning models |
| Inverse Truthfulness | Reasoning augmentation causes inverse truthfulness scaling — DFS amplifies false premises |
| PTQ Knowledge | Reasoning→precision-critical, Application→scale-responsive, Memorization→calibration-sensitive |
| Vocabulary Size | BBPE shows U-shaped threshold; BPE shows monotonic scaling — fundamentally different |
| MaskTab | Tabular data exhibits clear scaling trends; self-supervised pretraining beats GBDT |

**The Big Picture**: Scaling law research has evolved from the simple "N↑ D↑ → Performance↑" power law into multi-dimensional, task-aware, constraint-driven precise modeling. The core message of 2026: **knowing *where* to invest resources (data selection, language allocation, retrieval quality, bit-width assignment) matters far more than blindly scaling up.**

</div>
