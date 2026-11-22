---
title: "125B MoE on One Consumer GPU: How Much Does 2.5-Bit Quantization Actually Cost?"
date: 2026-11-22 10:00:00
tags:
  - LLM
  - quantization
  - benchmark
  - AMD
  - RDNA3
  - MoE
categories:
  - machine learning
mathjax: true
---

<div class="lang-switch">
  <button id="btn-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
  <button id="btn-en" class="lang-btn" onclick="switchLang('en')">English</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh">

## 单卡消费级显卡跑 125B MoE：2.5 bit 量化到底损失了什么？

### 1. 背景

Qwen3.8-Flash-Next 是一个 125B 总参数、6B 激活的稀疏 MoE 模型（外加 51B n-gram 嵌入表与 4B MTP 头）。官方权重是 BF16，而消费级显卡显然装不下。社区里最流行的做法是用 2–3 bit 的 GGUF 量化，把权重压到 40 GB 上下，再靠"专家分页"（把放不进显存的部分留在内存、按需搬运）跑起来。

问题在于：**这些量化和官方 BF16 的差距到底有多大？** 社区里常见的是"感觉还行"这类主观描述。这篇文章给出实测数据。

### 2. 测试环境

| 组件 | 配置 |
|---|---|
| GPU | AMD Radeon RX 7900 XTX，24 GB（gfx1100 / RDNA3） |
| CPU / RAM | Intel i9-14900KF，64 GB DDR4-3200 |
| 存储 | NVMe SSD（n-gram 表 28.8 GB 按需直读） |
| 推理引擎 | Strata（RDNA3 分支，自行在 Windows 上编译） |
| 量化 | IQ2_XS，约 2.5 bit，权重 39.2 GB |
| 上下文 | 262,144 tokens，KV 用 k8v4 混合格式 |
| 投机解码 | MTP，窗口 4 |

实测速度：**解码 89–95 token/s**，预填 13K 上下文 2,115 token/s、53K 上下文 2,325 token/s，3 路并发合计约 116 token/s。

需要强调的是：**模型、量化、引擎都是社区方案，不是官方部署**。下面所有"官方分"都取自模型卡片。

### 3. 评测结果

所有评测都通过本地 OpenAI 兼容接口调用，采样参数按官方推荐（temperature 1.0、top_p 0.95、top_k 20、reasoning_effort xhigh），每题只用最终答案判分。

| 榜单 | 官方 BF16 | 本地 IQ2_XS | 差距 | 保持率 |
|---|---:|---:|---:|---:|
| RealWorldQA | 88.5 | **90.0** | +1.5 | 102% |
| GPQA Diamond | 91.7 | **86.0** | −5.7 | 94% |
| IFBench | 81.3 | **73.0** | −8.3 | 90% |
| LiveCodeBench v6 | 91.9 | **80.0** | −11.9 | 87% |
| MathVision | 90.6 / 95.7 | **78.0** | −12.6 ~ −17.7 | 81–86% |
| SWE-bench Pro（试点） | 62.5 | **0 / 4** | — | ~0% |

样本量：RealWorldQA 50 题、GPQA 50 题、IFBench 300 题（全量）、LiveCodeBench 20 题、MathVision 50 题。因此除 IFBench 外都存在数个百分点到十余个百分点的统计误差，结论应看趋势而非小数点。

### 4. 三个值得注意的观察

**（一）损失不是均匀的，而是按"精确性"分布**

把上面几个榜单按能力性质重新排列，规律非常清楚：

| 能力类型 | 保持率 | 典型例子 |
|---|---|---|
| 看懂图片、识别物体 | ~100% | RealWorldQA |
| 知识型推理 | 94% | GPQA Diamond |
| 指令遵循（简单约束） | 90% | IFBench |
| 单函数代码生成 | 87% | LiveCodeBench |
| 看图 + 精确计算 | 81–86% | MathVision |
| 仓库级多步编辑 | ~0% | SWE-bench |

IFBench 的分项更能说明问题：`count:numbers`、`words:no_consecutive`、`format:newline` 这类简单约束通过率 **100%**，而 `format:quotes` 是 **0/8**、`ratio:overlap` 只有 17%、`words:odd_even_syllables` 22%。MathVision 里角度题 5/5 满分，组合计数只有 1/4。RealWorldQA 的错题也全部是"数错个数""左右判反"，而不是"看不懂图"。

一句话：**量化伤的是精确性，不是理解力。**

**（二）仓库级 agent 任务直接失效**

在 SWE-bench Lite 上试了 4 个真实 issue。模型能读代码、能定位到正确的函数，但**产不出可直接应用的补丁**：三次输出 0 字节 diff，一次给出了补丁但内容错误。明确要求它用文件编辑工具时，它倾向于"解释问题"而不是"动手修改"。

这和 LiveCodeBench 80% 形成鲜明对比：**写一个函数没问题，做一次仓库级的、多步骤的修改就崩了**。低比特量化在多步累积下，指令偏移会被放大。

**（三）视觉几乎无损**

RealWorldQA 拿到 90.0，和官方 88.5 在误差范围内持平。视觉编码器只有约 0.45B 参数（907 MB BF16），并且是把图片编码成 token 后交给同一个语言模型处理——所以只要"看图"这一步没坏，后面的推理损失就和文本侧一致。

### 5. 方法学说明（以及我们的局限）

- **采样参数**：官方卡片明确推荐思考模式用 temperature 1.0 / top_p 0.95 / top_k 20。我们全程采用这套参数。要注意 IFBench 原论文的协议是 temperature 0，两者不同，因此和论文数字对比时需谨慎。
- **判分**：全部使用各榜单的官方校验器或参考答案匹配。IFBench 用官方 `run_eval`（83 个约束校验器），MathVision 与 RealWorldQA 用标准答案归一化匹配，LiveCodeBench 在容器内跑官方测试用例（严格 pass@1）。
- **样本量**：如前述，除 IFBench 外都是抽样。50 题的 95% 置信区间大约是 ±14 个百分点（p≈0.8 时），所以"−5.7"这种差距更适合读作"几个点"。
- **不确定项**：MathVision 官方报告 "with CI / without CI" 两个数，但数据集文档没有定义 CI 的含义；我们的 prompt 带有"逐步求解"的引导，因此更接近 with CI 那一档。
- **不是苹果对苹果**：官方数字来自 BF16 权重、官方推理栈、完整评测集；我们的数字来自 2.5 bit 量化、社区引擎、抽样。差距里既有量化因素，也有引擎与实现差异。

### 6. 结论

在一个 24 GB 消费级显卡上，2.5 bit 量化的 Qwen3.8-Flash-Next 可以跑到 90 token/s 的解码速度，并且在**知识问答、图片理解、简单指令遵循**上保持了官方 90–100% 的水平。

代价集中在需要"逐字符、逐数字、多步累积"的任务上：代码生成掉一成多，精确计数和多步计算掉一到两成，仓库级 agent 任务基本不可用。

如果你的场景是"读文档、答问题、写小段代码"，这套方案完全够用；如果你要它做长时间自主的代码修改，就别指望低比特量化。

---

*评测脚本与配置：Strata RDNA3 分支 + 自编译的 Windows 引擎；评测用 lm-evaluation-harness、IFBench 官方校验器、以及为 chat 接口自写的 MCQ / 代码执行脚本。所有评测数据可在本地复现。*

</div>

<!-- English Version -->
<div class="lang-content lang-en" style="display:none;">

## 125B MoE on one consumer GPU: what does 2.5-bit quantization really cost?

### 1. Background

Qwen3.8-Flash-Next is a sparse MoE with 125B total parameters and 6B active, plus a 51B n-gram embedding table and a 4B MTP head. The official weights are BF16, which no consumer card can hold. The common workaround is a 2–3 bit GGUF quantization that brings the weights to about 40 GB and pages the experts that do not fit in VRAM into system RAM.

The open question is simple: **how far does such a quantization actually fall from the official BF16 model?** Community reports are mostly qualitative. This post gives measurements.

### 2. Setup

| Component | Configuration |
|---|---|
| GPU | AMD Radeon RX 7900 XTX, 24 GB (gfx1100 / RDNA3) |
| CPU / RAM | Intel i9-14900KF, 64 GB DDR4-3200 |
| Storage | NVMe SSD (the 28.8 GB n-gram table is read directly from disk) |
| Engine | Strata (RDNA3 fork, compiled from source on Windows) |
| Quantization | IQ2_XS, about 2.5 bits per weight, 39.2 GB |
| Context | 262,144 tokens, k8v4 mixed KV format |
| Speculative decoding | MTP, window 4 |

Measured speed: **89–95 tokens/s decode**, prefill 2,115 tokens/s at 13K context and 2,325 tokens/s at 53K, about 116 tokens/s aggregate with three concurrent requests.

All of this is a community stack: the engine is a fork, the quantization is community-made. Every "official" number below comes from the model card.

### 3. Results

Each benchmark was driven through the local OpenAI-compatible endpoint with the sampling the model card recommends for thinking mode (temperature 1.0, top_p 0.95, top_k 20, reasoning effort xhigh).

| Benchmark | Official BF16 | Local IQ2_XS | Delta | Retained |
|---|---:|---:|---:|---:|
| RealWorldQA | 88.5 | **90.0** | +1.5 | 102% |
| GPQA Diamond | 91.7 | **86.0** | −5.7 | 94% |
| IFBench | 81.3 | **73.0** | −8.3 | 90% |
| LiveCodeBench v6 | 91.9 | **80.0** | −11.9 | 87% |
| MathVision | 90.6 / 95.7 | **78.0** | −12.6 to −17.7 | 81–86% |
| SWE-bench Pro (pilot) | 62.5 | **0 / 4** | — | ~0% |

Sample sizes: 50 items for RealWorldQA, GPQA and MathVision, 300 (the full set) for IFBench, 20 for LiveCodeBench. Except for IFBench, treat these as trends rather than precise points.

### 4. Three observations

**Loss is not uniform; it tracks precision.**

Sorted by the kind of ability required, the pattern is striking:

| Ability | Retained | Example |
|---|---|---|
| Reading an image | ~100% | RealWorldQA |
| Knowledge-heavy reasoning | 94% | GPQA Diamond |
| Instruction following (simple constraints) | 90% | IFBench |
| Single-function code generation | 87% | LiveCodeBench |
| Figure + exact computation | 81–86% | MathVision |
| Repository-level editing | ~0% | SWE-bench |

The IFBench breakdown makes the same point at constraint level: simple constraints such as `count:numbers`, `words:no_consecutive` and `format:newline` pass at **100%**, while `format:quotes` scores **0/8**, `ratio:overlap` 17% and `words:odd_even_syllables` 22%. In MathVision, angle questions score 5/5 while combinatorics scores 1/4. The RealWorldQA failures are all miscounts and flipped left/right judgements, never "the model did not see the object".

**Quantization damages precision, not understanding.**

**Repository-level agent work collapses.**

On four real SWE-bench Lite issues the model located the right function and read the code, but could not produce an applicable patch: three outputs were empty, one was wrong. Asked to edit files with a tool, it preferred to explain the issue instead of changing the code.

That contrasts sharply with 80% on LiveCodeBench: single functions are fine, but a multi-step repository edit is not. Under low-bit quantization, instruction drift compounds across steps.

**Vision is essentially untouched.**

RealWorldQA came in at 90.0 against the official 88.5, inside the noise. The vision encoder is small (about 0.45B parameters, a 907 MB BF16 projection) and simply turns an image into tokens for the same language model, so once seeing is intact the text-side losses dominate.

### 5. Method and limitations

- **Sampling.** The card recommends temperature 1.0 / top_p 0.95 / top_k 20 for thinking mode and that is what we used. The IFBench paper itself evaluates at temperature 0, so comparisons with published IFBench figures need care.
- **Grading.** We used the official verifiers wherever they exist: IFBench's `run_eval` (83 constraint checkers), the official test cases inside a container for LiveCodeBench (strict pass@1), and answer matching against the dataset's reference answers for MathVision and RealWorldQA.
- **Sample size.** Everything except IFBench is a sample; with 50 items the 95% interval around p≈0.8 is roughly ±11 points, so a "−5.7" is better read as "a few points".
- **Unknowns.** MathVision reports "with CI" and "without CI" numbers and the dataset card does not define CI. Our prompts ask for step-by-step reasoning, so we sit closer to the with-CI column.
- **Not apples to apples.** Official numbers come from BF16 weights, the official stack and full test sets. Ours come from a 2.5-bit file, a community engine and samples.

### 6. Takeaway

On a single 24 GB consumer card, a 2.5-bit Qwen3.8-Flash-Next decodes at about 90 tokens/s and keeps 90–100% of the official quality on knowledge questions, image understanding and simple instruction following.

The cost lands on tasks that need character-level, number-level or multi-step precision: code generation loses about a tenth, exact counting and multi-step arithmetic lose one to two tenths, and repository-level agent work is effectively out of reach.

If your workload is reading documents, answering questions and writing small pieces of code, this is a very usable setup. If you need autonomous multi-step code editing, low-bit quantization is not the tool.

---

*Evaluation scripts and configuration: the Strata RDNA3 fork with a self-built Windows engine; lm-evaluation-harness, the official IFBench verifier, and custom MCQ and container-executed code runners for the chat API. Everything is reproducible locally.*

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
      titleEl.textContent = '125B MoE 单卡消费级显卡：2.5 bit 量化到底损失了什么？';
    } else {
      titleEl.textContent = '125B MoE on One Consumer GPU: How Much Does 2.5-Bit Quantization Actually Cost?';
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
  switchLang('zh');
});
</script>
