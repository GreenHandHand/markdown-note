# Agent 方向

总结了一下目前看的一些 Agent 的文章/项目。

基本的思路都是：

1. 数据处理：把数据处理成 LLM 可以接受的形式。这部分一般是项目处理起来最麻烦的地方。
	1. 爬取网上的数据放到数据库中
	2. 利用 VLM 将图片数据翻译为文本
	3. 也有 base 直接使用多模态模型的
2. 一个 Master Agent 来管理整个项目，负责分配其他 Agent 的任务、输入等。一般这个 Agent 的 prompt 里面会包含所有 Agent 的能力、工具等信息。
	1. 这里不一定只有一个 Agent，也有多层架构的，比如先拆解任务，然后再分配任务。
3. 每个 Agent 根据自己的 prompt 和输出，调用不同的工具。
	1. 在这一步中，需要特别处理的是不同 Agent 之间的信息共享的方式。
	2. 目前看到的方式包括：
		1. 议会形式，不同 Agent 针对一个任务的不同看法**投票**。
		2. 辩论形式，不同 Agent 输出自己的看法，尝试说服其他的持有不同看法的 Agent，然后交由一个最后的 Agent 进行判断。
		3. Shared Memory 形式，每个 Agent 都会把输出记录到 Shared Memory 中作为共享的记忆。
4. 后处理：最后由一个 Master Agent （或者一个专门的总结的 Agent） 输出最终结果或者与用户交互。

> [!note] 关于 Reflection
> 这部分在 Agent 设计中很常见，一般的处理过程都是有外层循环，例如当前 Agent 处理不了的任务，会在下一次循环中重新尝试处理（换 Agent/ 换任务）。

> [!tip] 辩论和投票的对比
> 见 NeruIPS 2025 **[Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](https://openreview.net/forum?id=iUjGNJzrF1)**
> 
> 这篇的观点是简单的 Vote 就比多轮的 Debate 有效。

在前面的框架下，有不同的实现方法，包括：

- **不微调**：仅一个框架，调用现有的 API 完成任务。目前看到的 github 上面开源的项目都是这种处理方式。
- **SFT + 强化学习**：一个基本的共识是要专门微调 Master Agent，使其可以处理对应的领域的下游任务。这里一般都是先有监督微调，然后再用强化学习方法进一步调整。

## NeurIPS

这里给的是两篇 NeurIPS 2025 中有关 Multi-Agent 框架的论文。

### OWL: Optimized Workforce Learning

这篇提出的了泛用的多智能体框架。

![[98_Assets/InfiniCore 比赛.png]]

![[98_Assets/Agent-2.png]]

## Multi-Agent Collaboration via Evolving Orchestration

引入一个中心化的、可学习的协调器（orchestrator），动态决定在每一步由哪个智能体（agent）进行推理，从而实现灵活、自适应的协作流程。

- 简单来说之前的 Master Agent 只在一开始分配任务时起作用，而这篇文章提出的 Master Agent 在每一步都会决定下一个执行的 Agent 是谁。
- 类似于传球。球就是每一步 Agent 的输出信息。
- 实验发现，经过强化学习训练后，智能体之间的协作图会逐渐演化出更紧凑的结构，并包含循环连接（如 A→B→A），这种结构有助于高效迭代和减少冗余计算。

![[98_Assets/Agent.png]]

![[98_Assets/Agent-3.png]]