# AFLOW

> [!note] ICLR 2025

AFlow: Automating Agentic Workflow Generation

一句话总结：将 Agent 交互行为编码为节点与行为，利用蒙特卡洛树搜索方式自动探索合适的 Workflow 结构。

## Abstract

Large language models (LLMs) have demonstrated remarkable potential in solving complex tasks across diverse domains, typically by employing agentic workflows that follow detailed instructions and operational sequences. However, constructing these workflows requires significant human effort, limiting scalability and generalizability. Recent research has sought to automate the generation and optimization of these workflows, but existing methods still rely on initial manual setup and fall short of achieving fully automated and effective workflow generation. To address this challenge, we reformulate workflow optimization as a search problem over code-represented workflows, where LLM-invoking nodes are connected by edges. We introduce AFlow, an automated framework that efficiently explores this space using Monte Carlo Tree Search, iteratively refining workflows through code modification, tree-structured experience, and execution feedback. Empirical evaluations across six benchmark datasets demonstrate AFlow's efficacy, yielding a 5.7% average improvement over state-of-the-art baselines. Furthermore, AFlow enables smaller models to outperform GPT-4o on specific tasks at 4.55% of its inference cost in dollars. The code is available at [this https URL](https://github.com/FoundationAgents/AFlow).
大语言模型（LLMs）在解决多领域复杂任务方面展现出巨大潜力，通常通过遵循详细指令和操作序列的智能体工作流来实现。然而，构建这类工作流需要大量人工投入，限制了其可扩展性和通用性。尽管近期研究尝试自动生 成与优化工作流，但现有方法仍依赖初始人工设计，尚未实现高效全自动的工作流生成。
为此，我们将工作流优化重新定义为在**以代码表示的工作流空间中进行搜索**的问题——其中调用 LLM 的节点通过边连接。我们提出了 **AFlow**，一个自动化框架，利用**蒙特卡洛树搜索**（MCTS）高效探索该空间，并通过代码修改、树状经验积累和执行反馈迭代优化工作流。
在六个基准数据集上的实验表明，AFlow 显著优于当前最先进的方法，平均性能提升 5.7%。此外，AFlow 能让小型模型在特定任务上超越 GPT-4o，而推理成本仅为后者的 **4.55%**。代码已开源（见文末链接）。

## Method

### 定义

本文想要解决的是自动化构建最优 Agent 工作流的方式。为此，作者首先将一个 Agent 工作流分解为了 Node, Operator, Edge 三种结构。
- Node：Node 是一个单一的 Model，及相关的输入和输出的定义。这里作者为了简化搜索空间，将模型、Model 类型、输出类型作为超参数固定，只搜索 Prompt。
- Operator：Operator 是独立于 Model 的操作，是调用 Node 的方式，例如简单调用 Node 生成结果、调用多个 Node 进行辩论等。
- Edge：是 Node 之间的执行顺序，在本文中，使用代码 (Code) 的形式决定不同 Node 和 Operator 的执行顺序。

![[98_Assets/AFlow.png]]

![[98_Assets/AFlow-1.png]]

### 流程

![[98_Assets/AFlow-2.png]]

搜索的流程如下。在进行搜索前，会先将数据划分为 Validation Set 和 Test Set，搜索时使用 Validation Set。
1. 初始化一个简单的 Workflow，这个 Workflow 什么都不做。
2. 开始迭代。
	1. 从搜索树中利用蒙特卡洛方法选择一个节点。
	2. 利用 LLM-based 的方法将这个节点拓展。
	3. 评估拓展后的节点，并更新其分数。
	4. 如果找到更好的节点，更新全局最优 Workflow。

#### 初始化

由于调用 API 有代价，因此这里搜索时只使用整个数据集中的一小部分（20%）。为了得到更加有效的部分，作者对数据集进行了筛选：
1. 使用一个什么都不做的 Workflow，计算当前数据的分数。
2. 重复 5 次，得到平均和方差。
3. 选择方差小的样本，因为这样的样本更加容易看到不同 Workflow 的区别。

### 蒙特卡洛树搜索

这里的选择方式是按照加权概率进行随机选择，每个节点的加权概率如下：
$$
P_{\text{mixed}}(i)=\lambda \cdot \dfrac{1}{n} + (1 - \lambda) \cdot \dfrac{\exp{(\alpha \cdot (s_{i}-s_{\text{max}}))}}{\sum_{j=1}^n\exp(\alpha \cdot(s_{j} - s_{\text{max}}))}
$$
其中 $n$ 是 workflow 的数量，$s_{i}$ 是 workflow 的分数，$s_{\text{max}}$ 是 workflow 的最大分数，$\alpha$ 控制分数的影响，$\alpha$ 权衡探索与利用。

#### 探索

探索使用 LLM-based 的方法。使用 Prompt 
