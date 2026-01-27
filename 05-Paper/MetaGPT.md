# MetaGPT

> [!note] ICLR 2024

MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework

一句话：提出了一个 Multi-Agent 架构，模拟现实中 SOPs 的流程。**用文档的形式传递信息，而非自然语言**。

> [!note] SOP
> SOP（Standard Operating Procedure，标准操作程序）是一份清晰、书面化的操作指南，详细说明某项常规任务或流程应如何一致、安全、高效地执行。其目的是确保不同人员在不同时间执行同一任务时，结果稳定可靠，并符合质量、合规或安全要求。

## Abstract

Recently, remarkable progress has been made on automated problem solving through societies of agents based on large language models (LLMs). Previous LLM-based multi-agent systems can already solve simple dialogue tasks. More complex tasks, however, face challenges through logic inconsistencies due to cascading hallucinations caused by naively chaining LLMs. Here we introduce MetaGPT, an innovative meta-programming framework incorporating efficient human workflows into LLM-based multi-agent collaborations. MetaGPT encodes Standardized Operating Procedures (SOPs) into prompt sequences for more streamlined workflows, thus allowing agents with human-like domain expertise to verify intermediate results and reduce errors. MetaGPT utilizes an assembly line paradigm to assign diverse roles to various agents, efficiently breaking down complex tasks into subtasks involving many agents working together. On collaborative software engineering benchmarks, MetaGPT generates more coherent solutions than previous chat-based multi-agent systems.
近期，基于大语言模型（LLMs）的多智能体系统在自动化问题求解方面取得了显著进展。以往的 LLM 多智能体系统已能处理简单的对话任务，但在更复杂的任务中，由于简单串联 LLM 所引发的级联幻觉，常导致逻辑不一致。为此，我们提出了 MetaGPT——一种创新的元编程框架，将高效的人类工作流程融入基于 LLM 的多智能体协作中。MetaGPT 将标准操作流程（SOPs）编码为提示序列，以实现更流畅的工作流，使具备类人领域专长的智能体能够验证中间结果、减少错误。该框架采用流水线范式，为不同智能体分配多样化角色，高效地将复杂任务拆解为多个子任务，由多个智能体协同完成。在协同软件工程基准测试中，MetaGPT 生成的解决方案比以往基于聊天的多智能体系统更具连贯性。

## Method

本文将 Agents 组织为了一个软件公司的流水线，并使用 SOPs 的方法进行信息传递。

![[98_Assets/MetaGPT.png]]

### Case Example

这是 Boss (Human) 的一次输入后系统运行的流程图。

![[98_Assets/MetaGPT-1.png]]
整个系统按照软件公司中常见的人员分类，并按照流水线的形式顺序进行。
- 每个 Agents 有自己的角色 prompt，并只进行职责范围内容的任务。
- 每个 Agents 的输出形式由角色 prompt 规定，例如，产品经理输出文档，架构师输出架构图，工程师输出代码。

不同 Agent 之间通过共享信息池 (Shared Message Pool) 交互，每个 Agent 会根据自己的需要查询对应的信息，减少了模型上下文负担。

![[98_Assets/MetaGPT-2.png]]

## Experiments

数据集使用 HumanEval, MBPP 和一个自己生成的 SoftwareDev 数据集。
- HumanEval 包含 164 个手写程序任务，包含了函数定义、描述、参考代码和测试。
- MBPP 包含 427 个 Python 任务，包含了 Python 的核心概念和标准库特性，包含了描述、参考代码和自动化测试。
- SoftwareDev 数据集包含了 70 个可演示的软件开发任务，每个任务包含自己的任务 Prompt。这里数据集包含了各种领域，例如小游戏、图片处理算法、数据可视化等。

验证指标：
1. HumanEval 和 MBPP 采用 Pass@k：进行 k 次，通过一次测试的概率。公式：$\text{Pass@k}=\mathbb{E}_{\text{Problems}}\left[ 1-\dfrac{\tbinom{n-c}{k}}{\tbinom{n}{k}} \right]$
2. 在 SoftwareDev 项目中，优先考虑实际应用价值，并通过**人工评估**（A、E）或**统计分析**（B、C、D）来衡量性能，具体指标如下：
	- **(A) 可执行性（Executability）**: 该指标对生成代码的功能完整性进行评分，范围为 1 到 4：  
		- 1 分：代码无法运行或完全无功能  
		- 2 分：代码可运行，但存在明显缺陷  
		- 3 分：代码基本正确，仅有细微问题  
		- 4 分：代码完美无误，可直接使用
	- **(B) 成本（Cost）**: 综合评估以下三个方面：  
		1. 运行时间（Execution time）  
		2. Token 使用量（Token usage）  
		3. 实际费用（Expenses，如 API 调用成本等）
	- **(C) 代码统计信息（Code Statistics）**
		1. 生成的代码文件数量  
		2. 每个文件的代码行数  
		3. 总代码行数
	- **(D) 开发效率（Productivity）**，定义为：**Token 使用量 ÷ 总代码行数**，用于衡量生成每行代码所消耗的 Token 数量，值越低表示效率越高。
	- **(E) 人工修正成本（Human Revision Cost）**，指需要人工介入修正代码的次数，常见问题包括：  
		- 包导入错误（import errors）  
		- 类名错误（incorrect class names）  
		- 引用路径不完整（incomplete reference paths）  
		- 注：每次修正通常涉及不超过 3 行代码。
