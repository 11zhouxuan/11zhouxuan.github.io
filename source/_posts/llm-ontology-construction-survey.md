---
title: "LLM/Agent-Driven Ontology Construction: A Survey of 18 Papers (2024-2026)"
date: 2026-07-13 10:00:00
tags:
  - ontology
  - knowledge graph
  - LLM
  - multi-agent
  - survey
categories:
  - research
mathjax: false
---

<div class="lang-switch">
  <button id="btn-en" class="lang-btn active" onclick="switchLang('en')">English</button>
  <button id="btn-zh" class="lang-btn" onclick="switchLang('zh')">中文</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh" style="display:none;">

<!-- more -->

{% note info %}
**概览**：本文综述了 2024–2026 年间 18 篇利用 LLM/Agent 自动或辅助构建本体（Ontology）、分类体系（Taxonomy）、知识图谱 Schema 的代表性论文。从 Multi-Agent 全自动构建到人机协作工具，覆盖方法谱系的全部光谱。
{% endnote %}

## 研究背景

本体（Ontology）是组织领域知识的形式化表示，传统构建过程依赖领域专家和知识工程师的大量人工劳动。2024 年以来，随着大语言模型（LLM）能力的飞跃和 Agent 系统的成熟，自动化本体构建成为一个爆发式增长的研究方向。

本文将 18 篇论文按方法论分为六大类：

| 类别 | 自动化程度 | 论文数量 |
|------|-----------|----------|
| Multi-Agent 自动构建 | 全自动 | 3 |
| LLM Pipeline / 框架 | 高度自动 | 6 |
| Taxonomy 自动构建 | 高度自动 | 3 |
| Ontology Engineering 辅助工具 | 半自动 | 4 |
| 特定领域应用 | 按需自动 | 4 |
| Benchmark / 评测 | 评测基准 | 1 |

---

## 一、Multi-Agent 自动构建本体

### 1.1 Towards Automated Ontology Generation from Unstructured Text: A Multi-Agent LLM Approach

| 项目 | 内容 |
|------|------|
| **arXiv** | 2604.23090 |
| **时间** | 2026.04 |
| **作者** | Abid Talukder, Maruf Ahmed Mridul, Oshani Seneviratne |

**动机**：从非结构化文本自动生成形式化本体是知识工程的核心挑战。尽管 LLM 展现潜力，但哪些架构设计选择驱动生成质量仍不清楚。

**方法**：

- 首先建立 Single-Agent 基线，识别关键失败模式：ODP 合规性差、结构冗余、迭代修复无效
- 提出四角色 Multi-Agent 架构：
  - **Domain Expert**：提供领域知识
  - **Manager**：协调流程与规划
  - **Coder**：实现 OWL 结构化组件
  - **Quality Assurer**：验证输出质量
- 强调 planning-first, artifact-driven 策略

**实验**：

- 数据集：保险合同领域
- 评估：架构质量（异构 LLM judges panel）+ 功能可用性（SPARQL + RAG 评估）
- 对比：Single-agent vs. Multi-agent

**结果**：Multi-agent 方法显著提升结构质量，适度增强可查询性。增益主要来自前置规划（front-loaded planning）。

{% note success %}
**核心启示**：Planning-first 的多角色协作是可扩展自动本体工程的有效路径。
{% endnote %}

---

### 1.2 AI Agent-Driven Framework for Automated Product KG Construction in E-Commerce

| 项目 | 内容 |
|------|------|
| **arXiv** | 2511.11017 |
| **时间** | 2025.11 |
| **作者** | Dimitar Peshevski, Riste Stojanov, Dimitar Trajanov |
| **会议** | GOBLIN Workshop 2025 |

**动机**：电商平台大量非结构化产品数据的 KG 构建仍然是复杂人工过程。

**方法**：全自动 AI Agent 驱动框架，分三阶段：

1. **Ontology Creation & Expansion Agent**：自动创建和扩展产品本体
2. **Ontology Refinement Agent**：精炼本体结构
3. **KG Population Agent**：从产品描述中填充知识图谱

不依赖预定义 schema 或手工抽取规则。

**实验**：真实空调产品描述数据集。

**结果**：属性覆盖率超过 **97%**，最小冗余。

{% note success %}
**核心启示**：Agent-based 方法在电商领域可实现从非结构化描述到结构化知识的全自动转换。
{% endnote %}

---

### 1.3 HyDRA: A Hybrid-Driven Reasoning Architecture for Verifiable Knowledge Graphs

| 项目 | 内容 |
|------|------|
| **arXiv** | 2507.15917 |
| **时间** | 2025.07 |
| **作者** | Adrian Kaiser, Claudiu Leoveanu-Condrei, Ryan Gold, Marius-Constantin Dinu, Markus Hofmarcher |

**动机**：KG 自动化构建面临输出可靠性、一致性和可验证性挑战（结构不一致、概念混淆等）。

**方法**：

1. 协作 neurosymbolic agents panel 共同协商 Competency Questions (CQs)
2. 基于 CQ 构建本体图
3. 本体图指导从文档中自动抽取三元组
4. Design-by-Contracts (DbC) 原则：可验证合约控制 LLM 生成

**评估**：提出超越标准 benchmark 的评估框架，利用 SymbolicAI 进行符号验证。

{% note success %}
**核心启示**：符号验证 + 神经生成的混合架构可提升自动 KG 构建的可靠性。
{% endnote %}

---

## 二、LLM Pipeline / 框架

### 2.1 OntoLearner: A Modular Python Library for Ontology Learning with LLMs

| 项目 | 内容 |
|------|------|
| **arXiv** | 2607.01977 |
| **时间** | 2026.07 |
| **作者** | Hamed Babaei Giglou, Jennifer D'Souza, Andrei Aioanei, Nandana Mihindukulasooriya, Sören Auer |
| **状态** | Under review at Nature Communications |
| **开源** | github.com/sciknoworg/OntoLearner (MIT) |

**动机**：Ontology learning 研究碎片化严重——方法、领域、评估实践各自为政。

**方法**：OntoLearner 统一三大模块：Ontology Access + LLM-Driven Learning Pipelines + Standardized Benchmarking。覆盖 Term Typing、Taxonomy Discovery、Non-Taxonomic Relation Extraction 三大核心任务。

**实验**：180 个机器可读本体，22 个领域，评测 22 个检索模型 + 12 个 LLM。

**核心发现**：

> "Failure modes scale with ontological complexity rather than model size."

关键瓶颈是**模型知识编码方式与本体组织结构之间的 structural mismatch**，而非模型能力不足。

{% note danger %}
**重要发现**：本体复杂度（而非模型规模）才是真正瓶颈——这颠覆了"更大模型更好"的直觉。
{% endnote %}

---

### 2.2 Automatic Ontology Construction: LLMs as Memory, Verification, and Planning

| 项目 | 内容 |
|------|------|
| **arXiv** | 2604.20795 |
| **时间** | 2026.04 |
| **作者** | Pavel Salovskii, Iuliia Gorshkova |

**动机**：LLM 缺乏长期记忆、结构理解弱、推理有限，仅依赖参数知识和向量检索不够。

**方法**：LLM + 外部本体记忆层混合架构：

- 自动管线：实体识别 → 关系抽取 → 归一化 → RDF/OWL 三元组生成
- 数据来源：文档、API、对话日志
- 验证：SHACL + OWL 约束
- 推理：向量检索 + 图推理 + 外部工具
- 形成 generation → verification → correction 循环

**实验**：Tower of Hanoi 多步推理 benchmark。

**结果**：本体增强在多步推理场景中显著优于基线 LLM。

---

### 2.3 Specific Domain Ontology Construction Using LLMs

| 项目 | 内容 |
|------|------|
| **arXiv** | 2606.20691 |
| **时间** | 2026.06 |
| **作者** | Vivian Magri Alcaldi Soares, Renata Wassermann |

**动机**：特定领域缺乏参考本体，手工构建费时费力。

**方法**：使用 GPT-3.5 和 GPT-4 扮演领域专家角色，为给定初始概念构建概念层次结构。

**实验**：自动构建 20 个本体，领域为巴西海洋领土（Blue Amazon）。人类领域专家评审。

**结果**：模型能构建整体连贯的领域概念化，但**没有任何输出无需精炼即完全满意**。GPT-4 优于 GPT-3.5。

{% note warning %}
**启示**：LLM 适合生成本体草稿，但仍需 human-in-the-loop 精炼。
{% endnote %}

---

### 2.4 From Prompt to Graph: LLM-Based IE for Domain-Specific Ontology Development

| 项目 | 内容 |
|------|------|
| **arXiv** | 2602.00699 |
| **时间** | 2026.02 |
| **作者** | Xuan Liu, Ziyu Li 等 13 人 |
| **会议** | Industry of the Future and Smart Manufacturing (2025) |

**动机**：铸造制造等专业领域传统本体构建依赖人工标注，成本高昂。

**方法**：对比三种 LLM 策略：

1. Pre-trained LLM-driven（直接使用预训练 LLM）
2. In-Context Learning (ICL)
3. Fine-tuning

在有限数据条件下从领域文本抽取术语和关系。

**实验**：铸造制造领域，领域专家验证。

**结果**：选择最优策略成功构建经验证的铸造本体。在有限数据条件下不同策略差异显著。

---

### 2.5 Large Ontology Models (LOM): Construct, Align, and Reason

| 项目 | 内容 |
|------|------|
| **arXiv** | 2602.00029 |
| **时间** | 2026.01 |
| **作者** | Yao Zhang, Hongyin Zhu |

**动机**：企业级知识管理面临多源异构数据集成和语义推理挑战。传统 KG 在隐式关系发现和复杂问答方面不足。

**方法**：提出 Large Ontology Model (LOM) 统一框架：

1. **Construct**：从结构化 DB + 非结构化文本构建双层企业本体
2. **Align**：三阶段训练——本体指令微调 → 文本-本体对齐 → 多任务课程学习
3. **Reason**：基于本体结构的语义推理和生成

**实验**：自建多种本体推理任务的 benchmark，4B 参数 LOM。

**结果**：准确率 **89.47%**，在复杂图推理上**超越 DeepSeek-V3.2**。

---

### 2.6 Unifying Ontology Construction and Semantic Alignment (LOM-CAR)

| 项目 | 内容 |
|------|------|
| **arXiv** | 2604.09608 |
| **时间** | 2026.03 |
| **作者** | Hongyin Zhu |

**动机**：现有神经符号方法依赖割裂管线，错误传播严重。

**方法**：Construct-Align-Reason (CAR) 统一管线：

1. 从原始数据自主构建领域本体
2. Graph-aware encoder + 强化学习对齐
3. 在拓扑、节点属性和关系类型上执行确定性推理

**实验**：多样真实企业数据集 benchmark，LOM-4B。

**结果**：本体补全 **88.8%**，复杂图推理 **94%**，显著超越 SOTA LLMs。

---

## 三、Taxonomy 自动构建

### 3.1 BoostTaxo: Zero-Shot Taxonomy Induction via Boosting-Style Agentic Reasoning

| 项目 | 内容 |
|------|------|
| **arXiv** | 2605.12520 |
| **时间** | 2026.04 |
| **作者** | Yancheng Ling, Zhenlin Qin, Leizhen Wang, Zhenliang Ma |

**动机**：现有 taxonomy induction 方法在泛化、结构可靠性和效率上受限，zero-shot 场景表现差。

**方法**：BoostTaxo 以 coarse-to-fine 方式进行 parent identification：

1. Retrieval-Augmented Definition Refinement
2. Hybrid Parent Candidate Selection：轻量 LLM 过滤 + 大规模 LLM 排序打分
3. Structure-Aware Score Calibration：结构特征校准边权重

**实验**：WordNet, DBLP, SemEval-Sci 三个 benchmark。

**Ablation Study**：
- Hybrid parent candidate selection 贡献显著
- Structure-aware score calibration 贡献显著
- 分析了 candidate selection size 对质量的影响

**结果**：在三个 benchmark 上达到 SOTA 或可比性能。

---

### 3.2 EvoTaxo: Building and Evolving Taxonomy from Social Media Streams

| 项目 | 内容 |
|------|------|
| **arXiv** | 2603.19711 |
| **时间** | 2026.03 |
| **作者** | Yiyang Li, Tianyi Ma, Yanfang Ye |

**动机**：社交媒体帖子短、噪声大、语义纠缠、时间动态变化，构建 taxonomy 极具挑战。

**方法**：

1. 将每个帖子转化为对当前 taxonomy 的 structured draft actions
2. 跨时间窗口累积结构证据
3. Dual-View Clustering：语义相似性 + 时间局部性
4. Refinement-and-Arbitration 选择可靠编辑
5. Concept Memory Banks 保持语义边界

**实验**：两个 Reddit 语料库 + /r/ICE_Raids case study。

**结果**：更好的平衡性、更清晰的帖子-叶节点映射、更优的语料覆盖率、更强的结构质量。

---

### 3.3 GIST: Taxonomy Maintenance over Evolving Scholarly Data

| 项目 | 内容 |
|------|------|
| **arXiv** | 2607.09149 |
| **时间** | 2026.07 |
| **作者** | Daomin Ji, Hui Luo, Zhifeng Bao, Junhao Gan, Zi Huang |
| **录用** | SIGMOD 2027 |

**动机**：科学出版物快速增长使学术分类体系迅速过时，需要从静态构建转向持续维护。

**方法**：GIST 框架：

1. 从论文 Related Work 章节提取部分层次结构
2. Geometric Box-Embedding 空间整合（box 包含编码 is-a 偏置）
3. 双向映射：word embedding ↔ box embedding
4. Novelty-Aware Coreset Selection 支持增量更新
5. Hypothesized Concept Generator + Cost-Effective Evidence Retrieval

**实验**：真实 arXiv 数据集。

**结果**：
- Node F1 提升 **+11.0%**
- Edge F1 提升 **+13.1%**
- 仅需 **9.6%** 运行时间
- 仅需 **12.7%** 货币成本

{% note success %}
**核心启示**：Taxonomy 维护比一次性构建更实际。专家证据 + geometric embedding 远优于纯 LLM 方法，且成本极低。
{% endnote %}

---

## 四、Ontology Engineering 辅助工具

### 4.1 Open Ontologies: Tool-Augmented Ontology Engineering with Stable Matching

| 项目 | 内容 |
|------|------|
| **arXiv** | 2605.09184 |
| **时间** | 2026.05 |
| **作者** | Fabio Rovai |
| **开源** | MIT License, Rust 实现 |

**动机**：本体工程需要 LLM 构建、形式推理和对齐的有效集成。

**方法**：集成 LLM-Driven Construction + Formal OWL Reasoning + Stable Matching Alignment (via MCP)。

**实验与 Ablation**：

- OAEI Anatomy Track: F1 = 0.832 (P = 0.963, R = 0.733)
- OAEI Conference Track: F1 = 0.438
- 5 种权重配置 ablation：有 stable matching 时 F1 变化 < 0.004；去除后 F1 降至 0.728

**惊人发现**：
- LLM 读取原始 OWL 文件：F1 = 0.323
- LLM 无文件（纯参数知识）：F1 = 0.431
- LLM + 结构化 MCP 工具访问：F1 = **0.717**

{% note danger %}
**重要发现**：让 LLM 直接读取 OWL 文件反而比不给文件更差！结构化工具接口提供了质上不同的访问模式。
{% endnote %}

---

### 4.2 From Subsumption to Satisfiability: LLM-Assisted Active Learning for OWL Ontologies

| 项目 | 内容 |
|------|------|
| **arXiv** | 2604.16672 |
| **时间** | 2026.04 |
| **作者** | Haoruo Zhao, Wenshuo Tang, Duncan Guthrie, Michele Sevegnani, David Flynn, Paul Harvey |

**动机**：如何利用 LLM 作为 "teacher" 回答主动学习中的 membership queries？

**方法**：

1. 将候选公理重新表述为反概念（counter-concept）
2. 用受控自然语言表达后呈现给 LLM
3. LLM 提供真实世界示例逼近反概念实例
4. 关键设计：**只可能出现 Type II 错误**（漏报），不引入不一致性

**实验**：测试 13 个商用 LLM，在多个成熟本体上评估。

**结果**：Recall 在各本体上保持稳定，保证不引入不一致。

---

### 4.3 A RAG Approach for Generating Competency Questions in Ontology Engineering

| 项目 | 内容 |
|------|------|
| **arXiv** | 2409.08820 |
| **时间** | 2024.09 |
| **作者** | Xueli Pan, Jacco van Ossenbruggen, Victor de Boer, Zhisheng Huang |
| **会议** | MTSR 2024 |

**动机**：Competency Questions (CQs) 制定依赖大量人力。

**方法**：RAG + GPT-4 自动生成 CQs，输入为科学论文集（非已有本体）。研究论文数量和 temperature 的影响。

**实验**：两个领域本体工程任务，与专家 ground-truth 对比。

**结果**：RAG 添加领域知识后显著优于 zero-shot prompting。

---

### 4.4 My Ontologist: Evaluating BFO-Based AI for Definition Support

| 项目 | 内容 |
|------|------|
| **arXiv** | 2407.17657 |
| **时间** | 2024.07 |
| **作者** | Carter Benson, Alec Sculley, Austin Liebers, John Beverley |

**动机**：评估 GPT-4 辅助基于 BFO 标准的本体定义生成。

**方法**：My Ontologist 系统，使用结构化规则约束 GPT-4 生成符合 BFO 的本体定义。

**结果**：Version 3.0 表现良好，但 **GPT-4o 升级破坏了已建立的工作流**。

{% note warning %}
**警告**：LLM 版本升级可能破坏本体工程工作流——模型行为不稳定性是集成 LLM 的关键风险。
{% endnote %}

---

## 五、特定领域应用

### 5.1 OntoKG: Ontology-Oriented KG Construction with Intrinsic-Relational Routing

| 项目 | 内容 |
|------|------|
| **arXiv** | 2604.02618 |
| **时间** | 2026.04 |
| **作者** | Yitao Li, Zhanlin Liu, Anuranjan Pandey, Muni Srikanth |

**方法**：Intrinsic-Relational Routing——将属性分为内在属性/关系属性，路由到对应 schema 模块。

**实验**：2026.01 Wikidata dump，34.6M entities。

**结果**：Category coverage 93.3%，Module assignment 98.0%，生成 34.0M nodes / 61.2M edges / 38 relationship types。

---

### 5.2 AutoPKG: Dynamic E-commerce Product-Attribute KG Construction

| 项目 | 内容 |
|------|------|
| **arXiv** | 2604.16950 |
| **时间** | 2026.04 |
| **作者** | Pollawat Hongwimol 等 |
| **录用** | ACL 2026 Findings |

**方法**：Multi-agent LLM 框架：产品类型诱导 → 属性键生成 → 多模态值抽取 → 中央决策 Agent 维护一致性。

**实验**：Lazada (Alibaba) 真实目录 + 3 个公开 benchmark。

**结果**：
- 产品类型 WKE: 0.953，属性键 WKE: 0.724
- **线上 A/B 测试**: Badge GMV +3.81%, Search +5.32%, Recommendation +7.89%

{% note success %}
**亮点**：少数报告了线上 A/B 实验和直接商业价值的论文。
{% endnote %}

---

### 5.3 From USD Scenes to Knowledge Graphs: Zero-Shot Ontology Grounding

| 项目 | 内容 |
|------|------|
| **arXiv** | 2606.09134 |
| **时间** | 2026.06 |
| **作者** | Jiangtao Shuai, Zongxiong Chen, Manfred Hauswirth, Sonja Schimmler |

**方法**：LLM zero-shot 将 3D 场景对象映射到 SOMA-HOME Ontology 类。

**实验**：厨房场景 125 个对象。

**Ablation Study**：
- 描述性名称：90-96% 准确率
- 匿名化语义线索：准确率降至 0-6%
- 仅几何信息：仅 4-17%

**结论**：LLM 主要利用场景图语义线索（sibling names + parent paths），而非几何特征。

---

### 5.4 LLM-Guided Robot Ontology Population from URDF

| 项目 | 内容 |
|------|------|
| **arXiv** | 2606.17073 |
| **时间** | 2026.06 |
| **作者** | Bastien Dussard, Guillaume Sarthou |
| **会议** | ICSR 2026 |

**方法**：URDF → 本体自动转换，LLM 推理语义关系，majority voting + schema 验证确保可靠性。

**结果**：有效桥接低级机器人描述与结构化知识表示的鸿沟。

---

## 六、Benchmark / 评测

### 6.1 OntoAxiom: Ontology Learning with LLMs — Axiom Identification Benchmark

| 项目 | 内容 |
|------|------|
| **arXiv** | 2512.05594 |
| **时间** | 2025.12 |
| **作者** | Roos M. Bakker, Daan L. Di Scala, Maaike H.T. de Boer, Stephan A. Raaijmakers |

**方法**：OntoAxiom benchmark——9 个中等规模本体（17,118 triples, 2,771 axioms），对比 Direct vs. Axiom-by-Axiom (AbA) prompting，测试 12 个 LLM。

**结果**：
- AbA prompting 优于 Direct
- FOAF 本体 subclass F1: 0.642，Music 本体仅 0.218
- 性能因本体差异极大

**结论**：LLM 可提供有价值的候选公理辅助本体工程师，但**不足以实现完全自动化**。

---

## 总结：关键趋势与启示

### 方法论谱系

```
完全自动化                                    人类主导
|<--- Multi-Agent --->|<--- Pipeline --->|<--- Tool/Assistant --->|
     (1.1-1.3)            (2.1-2.6)            (4.1-4.4)
```

### 八大趋势

1. **Multi-Agent 架构兴起**：将本体构建分解为多角色协作（2026 主流）
2. **Planning-First**：前置规划比迭代修复更有效
3. **结构化验证必不可少**：SHACL/OWL/SPARQL 验证是质量保障关键
4. **结构 Mismatch 是核心瓶颈**：不是模型不够强，而是知识编码与本体组织不匹配
5. **LLM 版本不稳定性**：模型更新可能破坏已有工作流
6. **工具接口 > 原始文件**：结构化工具访问远优于让 LLM 读取 OWL 语法
7. **动态维护优于一次性构建**：本体需持续适应
8. **商业验证**：已有论文报告 A/B 测试和 GMV 提升

</div>

<!-- English Version -->
<div class="lang-content lang-en">

{% note info %}
**Overview**: This post surveys 18 representative papers (2024–2026) on using LLMs and Agents to automatically or semi-automatically construct Ontologies, Taxonomies, and Knowledge Graph Schemas. From fully automated Multi-Agent systems to human-in-the-loop tools, we cover the full spectrum of approaches.
{% endnote %}

## Background

Ontologies are formal representations for organizing domain knowledge. Traditional construction relies heavily on domain experts and knowledge engineers. Since 2024, with the leap in LLM capabilities and the maturation of Agent systems, automated ontology construction has become an explosive research direction.

We organize 18 papers into six categories:

| Category | Automation Level | Papers |
|----------|-----------------|--------|
| Multi-Agent Construction | Fully Automated | 3 |
| LLM Pipeline / Framework | Highly Automated | 6 |
| Taxonomy Construction | Highly Automated | 3 |
| Ontology Engineering Tools | Semi-Automated | 4 |
| Domain-Specific Applications | On-demand | 4 |
| Benchmark / Evaluation | Baseline | 1 |

---

## I. Multi-Agent Ontology Construction

### 1.1 Towards Automated Ontology Generation from Unstructured Text: A Multi-Agent LLM Approach

| Item | Detail |
|------|--------|
| **arXiv** | 2604.23090 |
| **Date** | 2026.04 |
| **Authors** | Abid Talukder, Maruf Ahmed Mridul, Oshani Seneviratne |

**Motivation**: Automatically generating formal ontologies from unstructured text remains a central knowledge engineering challenge. It's unclear which architectural choices drive generation quality.

**Method**:

- Establish a Single-Agent baseline; identify failure modes: poor ODP compliance, structural redundancy, ineffective iterative repair
- Propose a four-role Multi-Agent architecture:
  - **Domain Expert**: Contributes subject matter knowledge
  - **Manager**: Orchestrates planning
  - **Coder**: Implements OWL components
  - **Quality Assurer**: Validates outputs
- Emphasizes planning-first, artifact-driven generation

**Experiments**:

- Dataset: Insurance contract domain
- Evaluation: Architectural quality (heterogeneous LLM judges) + Functional usability (SPARQL + RAG assessment)
- Comparison: Single-agent vs. Multi-agent

**Results**: Multi-agent significantly improves structural quality, modestly enhances queryability. Gains driven primarily by front-loaded planning.

{% note success %}
**Key Insight**: Planning-first multi-role collaboration is a promising path for scalable automated ontology engineering.
{% endnote %}

---

### 1.2 AI Agent-Driven Framework for Automated Product KG Construction in E-Commerce

| Item | Detail |
|------|--------|
| **arXiv** | 2511.11017 |
| **Date** | 2025.11 |
| **Authors** | Dimitar Peshevski, Riste Stojanov, Dimitar Trajanov |
| **Venue** | GOBLIN Workshop 2025 |

**Motivation**: Product KG construction from unstructured e-commerce data remains manual and complex.

**Method**: Fully automated AI Agent-driven framework in three stages:

1. **Ontology Creation & Expansion Agent**
2. **Ontology Refinement Agent**
3. **KG Population Agent**

No predefined schemas or handcrafted extraction rules required.

**Experiments**: Real-world air conditioner product descriptions.

**Results**: Property coverage over **97%** with minimal redundancy.

{% note success %}
**Key Insight**: Agent-based approaches can achieve fully automated unstructured-to-structured knowledge conversion in e-commerce.
{% endnote %}

---

### 1.3 HyDRA: A Hybrid-Driven Reasoning Architecture for Verifiable Knowledge Graphs

| Item | Detail |
|------|--------|
| **arXiv** | 2507.15917 |
| **Date** | 2025.07 |
| **Authors** | Adrian Kaiser, Claudiu Leoveanu-Condrei, Ryan Gold, Marius-Constantin Dinu, Markus Hofmarcher |

**Motivation**: Automated KG construction faces reliability, consistency, and verifiability challenges.

**Method**:

1. Collaborative neurosymbolic agents agree on Competency Questions (CQs)
2. Build ontology graph from CQs
3. Ontology guides automated triplet extraction
4. Design-by-Contracts (DbC): verifiable contracts steer LLM generation

**Evaluation**: Novel framework using SymbolicAI for functional correctness verification.

{% note success %}
**Key Insight**: Combining symbolic verification with neural generation improves reliability of automated KG construction.
{% endnote %}

---

## II. LLM Pipeline / Framework

### 2.1 OntoLearner: A Modular Python Library for Ontology Learning with LLMs

| Item | Detail |
|------|--------|
| **arXiv** | 2607.01977 |
| **Date** | 2026.07 |
| **Authors** | Hamed Babaei Giglou, Jennifer D'Souza, Andrei Aioanei, Nandana Mihindukulasooriya, Sören Auer |
| **Status** | Under review at Nature Communications |
| **Open Source** | github.com/sciknoworg/OntoLearner (MIT) |

**Motivation**: Ontology learning research is fragmented across methods, domains, and evaluation practices.

**Method**: OntoLearner unifies: Ontology Access + LLM-Driven Learning Pipelines + Standardized Benchmarking. Covers Term Typing, Taxonomy Discovery, and Non-Taxonomic Relation Extraction.

**Experiments**: 180 machine-readable ontologies across 22 domains; evaluated 22 retrieval models + 12 LLMs.

**Key Finding**:

> "Failure modes scale with ontological complexity rather than model size."

The bottleneck is a **structural mismatch** between how models encode knowledge and how ontologies organize it.

{% note danger %}
**Critical Finding**: Ontological complexity, not model size, is the real bottleneck — this challenges the "bigger is better" intuition.
{% endnote %}

---

### 2.2 Automatic Ontology Construction: LLMs as Memory, Verification, and Planning

| Item | Detail |
|------|--------|
| **arXiv** | 2604.20795 |
| **Date** | 2026.04 |
| **Authors** | Pavel Salovskii, Iuliia Gorshkova |

**Motivation**: LLMs lack long-term memory, have weak structural understanding, and limited reasoning.

**Method**: Hybrid LLM + external ontological memory architecture:

- Pipeline: Entity recognition → Relation extraction → Normalization → RDF/OWL triple generation
- Sources: Documents, APIs, dialogue logs
- Validation: SHACL + OWL constraints
- Inference: Vector retrieval + graph reasoning + external tools
- Forms a generation → verification → correction loop

**Experiments**: Tower of Hanoi multi-step reasoning benchmark.

**Results**: Ontology augmentation significantly improves multi-step reasoning over baseline LLMs.

---

### 2.3 Specific Domain Ontology Construction Using LLMs

| Item | Detail |
|------|--------|
| **arXiv** | 2606.20691 |
| **Date** | 2026.06 |
| **Authors** | Vivian Magri Alcaldi Soares, Renata Wassermann |

**Motivation**: Many specific domains lack reference ontologies; manual crafting is laborious.

**Method**: Use GPT-3.5 and GPT-4 as domain experts to build conceptual hierarchies for given initial concepts.

**Experiments**: Automatically constructed 20 ontologies for Brazil's maritime territory (Blue Amazon). Human expert evaluation.

**Results**: Models produce overall coherent conceptualizations, but **none was completely satisfactory without refinement**. GPT-4 outperforms GPT-3.5.

{% note warning %}
**Takeaway**: LLMs are suitable for ontology draft generation, but human-in-the-loop refinement remains essential.
{% endnote %}

---

### 2.4 From Prompt to Graph: LLM-Based IE for Domain-Specific Ontology Development

| Item | Detail |
|------|--------|
| **arXiv** | 2602.00699 |
| **Date** | 2026.02 |
| **Authors** | Xuan Liu, Ziyu Li, et al. (13 authors) |
| **Venue** | Industry of the Future and Smart Manufacturing (2025) |

**Motivation**: Traditional ontology construction in casting manufacturing is expensive.

**Method**: Compares three LLM strategies:

1. Pre-trained LLM-driven
2. In-Context Learning (ICL)
3. Fine-tuning

Extracts terms and relations from domain text with limited data.

**Experiments**: Casting manufacturing domain, expert-validated.

**Results**: Best-performing strategy successfully builds a validated casting ontology. Strategy choice significantly impacts quality under limited data.

---

### 2.5 Large Ontology Models (LOM): Construct, Align, and Reason

| Item | Detail |
|------|--------|
| **arXiv** | 2602.00029 |
| **Date** | 2026.01 |
| **Authors** | Yao Zhang, Hongyin Zhu |

**Motivation**: Enterprise knowledge management struggles with multi-source heterogeneous data integration and semantic reasoning.

**Method**: Large Ontology Model (LOM) unified framework:

1. **Construct**: Dual-layer enterprise ontology from structured DBs + unstructured text
2. **Align**: Three-stage training — ontology instruction fine-tuning → text-ontology grounding → multi-task curriculum learning
3. **Reason**: Ontology-based semantic reasoning and generation

**Experiments**: Custom benchmark covering diverse ontology reasoning tasks; 4B-parameter LOM.

**Results**: Accuracy **89.47%**, outperforms **DeepSeek-V3.2** on complex graph reasoning.

---

### 2.6 Unifying Ontology Construction and Semantic Alignment (LOM-CAR)

| Item | Detail |
|------|--------|
| **arXiv** | 2604.09608 |
| **Date** | 2026.03 |
| **Authors** | Hongyin Zhu |

**Motivation**: Existing neuro-symbolic approaches rely on disjoint pipelines with error propagation.

**Method**: Construct-Align-Reason (CAR) unified pipeline:

1. Autonomously construct domain-specific ontology from raw data
2. Graph-aware encoder + reinforcement learning for alignment
3. Deterministic reasoning over topology, node attributes, and relation types

**Experiments**: Diverse real-world enterprise datasets; LOM-4B.

**Results**: Ontology completion **88.8%**, complex graph reasoning **94%**, significantly outperforms SOTA LLMs.

---

## III. Taxonomy Construction

### 3.1 BoostTaxo: Zero-Shot Taxonomy Induction via Boosting-Style Agentic Reasoning

| Item | Detail |
|------|--------|
| **arXiv** | 2605.12520 |
| **Date** | 2026.04 |
| **Authors** | Yancheng Ling, Zhenlin Qin, Leizhen Wang, Zhenliang Ma |

**Motivation**: Existing taxonomy induction methods lack generalization, structural reliability, and efficiency in zero-shot scenarios.

**Method**: BoostTaxo performs coarse-to-fine parent identification:

1. Retrieval-Augmented Definition Refinement
2. Hybrid Parent Candidate Selection: lightweight LLM filters + large-scale LLM ranks
3. Structure-Aware Score Calibration: structural features calibrate edge weights

**Experiments**: WordNet, DBLP, SemEval-Sci benchmarks.

**Ablation Study**:
- Hybrid parent candidate selection: significant contribution
- Structure-aware score calibration: significant contribution
- Analysis of candidate selection size impact on taxonomy quality

**Results**: Achieves SOTA or comparable performance across all three benchmarks.

---

### 3.2 EvoTaxo: Building and Evolving Taxonomy from Social Media Streams

| Item | Detail |
|------|--------|
| **arXiv** | 2603.19711 |
| **Date** | 2026.03 |
| **Authors** | Yiyang Li, Tianyi Ma, Yanfang Ye |

**Motivation**: Social media posts are short, noisy, semantically entangled, and temporally dynamic.

**Method**:

1. Convert posts into structured draft actions over current taxonomy
2. Accumulate structural evidence across temporal windows
3. Dual-View Clustering: semantic similarity + temporal locality
4. Refinement-and-Arbitration for reliable edit selection
5. Concept Memory Banks to preserve semantic boundaries

**Experiments**: Two Reddit corpora + /r/ICE_Raids case study.

**Results**: Improved balance, clearer post-to-leaf assignments, superior corpus coverage, stronger structural quality.

---

### 3.3 GIST: Taxonomy Maintenance over Evolving Scholarly Data

| Item | Detail |
|------|--------|
| **arXiv** | 2607.09149 |
| **Date** | 2026.07 |
| **Authors** | Daomin Ji, Hui Luo, Zhifeng Bao, Junhao Gan, Zi Huang |
| **Accepted** | SIGMOD 2027 |

**Motivation**: Rapid growth of publications makes scholarly taxonomies quickly obsolete. Need continuous adaptation, not static construction.

**Method**: GIST framework:

1. Extract partial hierarchies from "Related Work" sections
2. Integrate in geometric box-embedding space (containment = is-a)
3. Bidirectional word ↔ box embedding mapping
4. Novelty-Aware Coreset Selection for incremental updates
5. Hypothesized Concept Generator + cost-effective evidence retrieval

**Experiments**: Real-world arXiv datasets.

**Results**:
- Node F1: **+11.0%** over strongest baseline
- Edge F1: **+13.1%**
- Only **9.6%** runtime
- Only **12.7%** monetary cost

{% note success %}
**Key Insight**: Taxonomy maintenance > one-time construction. Expert evidence + geometric embeddings vastly outperform pure LLM approaches at a fraction of the cost.
{% endnote %}

---

## IV. Ontology Engineering Tools

### 4.1 Open Ontologies: Tool-Augmented Ontology Engineering with Stable Matching

| Item | Detail |
|------|--------|
| **arXiv** | 2605.09184 |
| **Date** | 2026.05 |
| **Authors** | Fabio Rovai |
| **Open Source** | MIT License, Rust implementation |

**Motivation**: Ontology engineering needs effective integration of LLM construction, formal reasoning, and alignment.

**Method**: Integrates LLM-Driven Construction + Formal OWL Reasoning + Stable Matching Alignment (via MCP).

**Experiments & Ablation**:

- OAEI Anatomy Track: F1 = 0.832 (P = 0.963, R = 0.733)
- OAEI Conference Track: F1 = 0.438
- 5 weight configurations: With stable matching, F1 varies < 0.004; without it, drops to 0.728

**Surprising Finding**:
- LLM reading raw OWL file: F1 = 0.323
- LLM with no file (parametric only): F1 = 0.431
- LLM + structured MCP tool access: F1 = **0.717**

{% note danger %}
**Critical Finding**: Having an LLM read raw OWL files performs WORSE than no file at all! Structured tool interfaces provide a qualitatively different access mode.
{% endnote %}

---

### 4.2 From Subsumption to Satisfiability: LLM-Assisted Active Learning for OWL Ontologies

| Item | Detail |
|------|--------|
| **arXiv** | 2604.16672 |
| **Date** | 2026.04 |
| **Authors** | Haoruo Zhao, Wenshuo Tang, Duncan Guthrie, Michele Sevegnani, David Flynn, Paul Harvey |

**Motivation**: How to leverage LLMs as "teachers" for active learning membership queries?

**Method**:

1. Reformulate candidate axioms into counter-concepts
2. Verbalize in controlled natural language for LLM
3. LLM provides real-world examples approximating counter-concept instances
4. Design property: **only Type II errors possible** (delays, never inconsistencies)

**Experiments**: 13 commercial LLMs, evaluated across well-established ontologies.

**Results**: Recall remains stable across ontologies; guarantees no inconsistency introduction.

---

### 4.3 A RAG Approach for Generating Competency Questions in Ontology Engineering

| Item | Detail |
|------|--------|
| **arXiv** | 2409.08820 |
| **Date** | 2024.09 |
| **Authors** | Xueli Pan, Jacco van Ossenbruggen, Victor de Boer, Zhisheng Huang |
| **Venue** | MTSR 2024 |

**Motivation**: CQ formulation for ontology development is time-consuming and labor-intensive.

**Method**: RAG + GPT-4 for automatic CQ generation from scientific papers (not existing ontologies). Studies impact of paper count and temperature.

**Experiments**: Two domain ontology engineering tasks; comparison against expert ground-truth.

**Results**: Adding domain knowledge via RAG significantly outperforms zero-shot prompting.

---

### 4.4 My Ontologist: Evaluating BFO-Based AI for Definition Support

| Item | Detail |
|------|--------|
| **arXiv** | 2407.17657 |
| **Date** | 2024.07 |
| **Authors** | Carter Benson, Alec Sculley, Austin Liebers, John Beverley |

**Motivation**: Evaluate whether GPT-4 can assist BFO-trained ontologists.

**Method**: My Ontologist system using structured rules to constrain GPT-4 for BFO-compliant definitions.

**Results**: Version 3.0 showed promise, but **GPT-4o release disrupted established performance**.

{% note warning %}
**Warning**: LLM version upgrades can break ontology engineering workflows — model instability is a key integration risk.
{% endnote %}

---

## V. Domain-Specific Applications

### 5.1 OntoKG: Ontology-Oriented KG Construction with Intrinsic-Relational Routing

| Item | Detail |
|------|--------|
| **arXiv** | 2604.02618 |
| **Date** | 2026.04 |
| **Authors** | Yitao Li, Zhanlin Liu, Anuranjan Pandey, Muni Srikanth |

**Method**: Intrinsic-Relational Routing — classifies each property as intrinsic or relational, routes to corresponding schema module.

**Experiments**: January 2026 Wikidata dump, 34.6M entities.

**Results**: Category coverage 93.3%, Module assignment 98.0%; generates 34.0M nodes / 61.2M edges / 38 relationship types.

---

### 5.2 AutoPKG: Dynamic E-commerce Product-Attribute KG Construction

| Item | Detail |
|------|--------|
| **arXiv** | 2604.16950 |
| **Date** | 2026.04 |
| **Authors** | Pollawat Hongwimol et al. |
| **Accepted** | ACL 2026 Findings |

**Method**: Multi-agent LLM framework: product type induction → attribute key generation → multimodal value extraction → centralized decision agent.

**Experiments**: Lazada (Alibaba) real-world catalog + 3 public benchmarks.

**Results**:
- Product type WKE: 0.953; Attribute key WKE: 0.724
- **Online A/B test**: Badge GMV +3.81%, Search +5.32%, Recommendation +7.89%

{% note success %}
**Highlight**: One of few papers reporting online A/B experiments with direct business value.
{% endnote %}

---

### 5.3 From USD Scenes to Knowledge Graphs: Zero-Shot Ontology Grounding

| Item | Detail |
|------|--------|
| **arXiv** | 2606.09134 |
| **Date** | 2026.06 |
| **Authors** | Jiangtao Shuai, Zongxiong Chen, Manfred Hauswirth, Sonja Schimmler |

**Method**: LLM zero-shot grounding of 3D scene objects to SOMA-HOME Ontology classes.

**Experiments**: Kitchen scene with 125 objects.

**Ablation Study**:
- Descriptive names: 90-96% accuracy
- Anonymized semantic cues: drops to 0-6%
- Geometry alone: only 4-17%

**Conclusion**: LLMs primarily exploit semantic cues in the scene graph (sibling names, parent paths), not geometry.

---

### 5.4 LLM-Guided Robot Ontology Population from URDF

| Item | Detail |
|------|--------|
| **arXiv** | 2606.17073 |
| **Date** | 2026.06 |
| **Authors** | Bastien Dussard, Guillaume Sarthou |
| **Venue** | ICSR 2026 |

**Method**: URDF → ontology automatic conversion; LLM infers semantic relationships; majority voting + schema validation ensures reliability.

**Results**: Effectively bridges low-level robot descriptions and structured knowledge representations.

---

## VI. Benchmark / Evaluation

### 6.1 OntoAxiom: Ontology Learning with LLMs — Axiom Identification Benchmark

| Item | Detail |
|------|--------|
| **arXiv** | 2512.05594 |
| **Date** | 2025.12 |
| **Authors** | Roos M. Bakker, Daan L. Di Scala, Maaike H.T. de Boer, Stephan A. Raaijmakers |

**Method**: OntoAxiom benchmark — 9 medium ontologies (17,118 triples, 2,771 axioms); compares Direct vs. Axiom-by-Axiom (AbA) prompting across 12 LLMs.

**Results**:
- AbA outperforms Direct
- FOAF subclass F1: 0.642; Music: only 0.218
- Performance varies dramatically across ontologies

**Conclusion**: LLMs provide valuable candidate axioms to support ontology engineers, but are **insufficient for full automation**.

---

## Summary: Key Trends

### Methodology Spectrum

```
Fully Automated                               Human-Led
|<--- Multi-Agent --->|<--- Pipeline --->|<--- Tool/Assistant --->|
       (1.1-1.3)           (2.1-2.6)            (4.1-4.4)
```

### Eight Key Trends

1. **Multi-Agent Architectures Rising**: Ontology construction decomposed into multi-role collaboration (2026 mainstream)
2. **Planning-First**: Upfront planning beats iterative repair
3. **Structured Validation Essential**: SHACL/OWL/SPARQL validation is key to quality assurance
4. **Structural Mismatch is the Core Bottleneck**: Not model weakness, but encoding-vs-organization mismatch
5. **LLM Version Instability**: Model updates can break established workflows
6. **Tool Interface > Raw Files**: Structured tool access vastly outperforms LLMs reading OWL syntax
7. **Dynamic Maintenance > One-Time Construction**: Ontologies need continuous adaptation
8. **Commercial Validation**: Papers now report A/B tests and GMV improvements

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

  var titleEl = document.querySelector('.post-title');
  if (titleEl) {
    if (lang === 'zh') {
      titleEl.textContent = 'LLM/Agent 驱动的本体自动构建：18 篇论文综述 (2024-2026)';
    } else {
      titleEl.textContent = 'LLM/Agent-Driven Ontology Construction: A Survey of 18 Papers (2024-2026)';
    }
  }

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

document.addEventListener('DOMContentLoaded', function() {
  switchLang('en');
});
</script>
