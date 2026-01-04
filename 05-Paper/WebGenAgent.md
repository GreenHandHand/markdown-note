# WebGen-Agent: Towards Visual Feedback Driven Web Generation

arxiv, 2025.11

 主要内容：这篇论文瞄准了网页代码生成中反馈太弱的问题，通过截图和 GUI 代理的视觉交互反馈来闭环迭代代码，并把这些反馈转化为步级强化学习信号，帮助小模型生成更好看的网页。

> [!note] GRPO (Group Relative Policy Optimization)  
> 一种相对策略优化方法，通过比较一组候选策略的相对表现来更新模型，弱化对绝对 reward 标定的依赖，常用于 LLM 强化学习场景。

---

## Key Contributions

- Propose WebGen-Agent, a web generation framework that incorporates visual feedback from screenshots and a GUI testing agent.
- Introduce Step-GRPO, which leverages screenshot-based and GUI-based feedback as reward signals for training.

- 作者提出了 WebGen-Agent，将网页生成从纯文本 → 代码的单向生成，重构为一个 *以真实网页渲染结果为中心的闭环系统*。核心转变在于：网页是否“正确”，不再仅由代码语义或 LLM 自评决定，而是由 *视觉层面的可观察状态* 与 *可交互行为* 来共同裁决。
- 在系统层闭环之外，作者进一步将视觉与交互反馈“制度化”为强化学习信号，提出 Step-GRPO，使得模型不仅在推理阶段可被纠正，在参数层面也能逐步内化“什么样的网页是好的”。

---

## Method

- 首先是提出的网页生成系统 WebGen-Agent，该系统利用屏幕截图和 GUI agent 生成的内容作为反馈，来迭代优化生成网页的功能和外观。
- 在网页生成系统的基础上，再引入了 Step-GRPO 方法。该方法利用屏幕截图和 GUI agent 反馈作为分数，利用 GRPO 对模型进行训练。

### WebGen-Agent

流程非常直观：

1. Code Agent 生成代码
2. 执行代码，获取控制台输出（报错则重新生成）
3. 获取网页截图。利用 VLM（视觉语言模型）检查网页实现效果，输出分数、修改建议
	1. VLM 检查网页是否报错（如 404、堆栈跟踪等），如果报错，则重新生成代码
	2. VLM 根据截图给出描述，并给出修改建议
4. 将（输入，生成代码，控制台输出，描述，分数，修改建议）全部返回给 Coding Agent，由 Coding Agent 判断是否实现了功能
5. 如果实现了功能，则开启一个 GUI Test 对话，用 GUI Agent 判断网页实现是否正确
	1. 按照输入指令，生成 Web-navigation 的 GUI Agent 的指令
	2. GUI Agent 根据指示进行互动，并生成分数

每一步生成一段正确的代码，以此迭代。

> [!warning]
> 这个闭环流程看起来简单，但实际执行时可能会导致生成时间大幅增加，论文没有讨论实际应用中的效率问题。

### Step GROP

作者的直觉是：网页的*好*与*坏*是连续的，不是二元的。传统强化学习用 " 正确/错误 " 作为奖励，但网页的好坏是程度问题，比如 " 搜索功能部分工作 " 比 " 完全不工作 " 好，但两者都是 " 错误 "。Step-GRPO 通过相对奖励，让模型学会 " 什么样的网页更好 "，而不是简单地学习 " 正确/错误 "。

该部分使用前面的屏幕截图分数和 GUI 分数作为强化学习的目标。

Step-GRPO 的核心是利用截图和 GUI 反馈作为相对奖励信号。论文提到作者使用 DeepSeek-V3 事先生成了 700 个策略，并进行 SFT（监督微调）作为 Warm start。这相当于给模型一个起点，避免训练初期的不稳定。

> [!warning]
> 作者说*使用 DeepSeek-V3 生成 700 个策略*，但没有解释这些策略是如何生成的，以及它们是否代表了真实的网页生成空间。

> [!note] 消融实验显示，如果只给截图分或只给 GUI 分，效果都会显著下降

---

## future

- 作者提到可以将 WebGen-Agent 扩展到更多类型的 Web 应用，如电商网站、社交媒体平台等。
- 作者也提到可以进一步优化 Step-GRPO，使其更适应不同的 Web 生成任务。

> [!note]
> - 将 WebGen-Agent 与实时用户行为分析结合，让系统根据真实用户点击数据自动优化网页设计，而不仅仅是依赖预设的测试用例。
> - 探索更高效的视觉反馈机制，减少对 VLM 的依赖，比如使用轻量级的图像分析模型，降低成本和延迟。
> - 研究如何将 WebGen-Agent 应用到动态 Web 应用生成中，而不仅仅是静态网页，比如实时生成交互式仪表盘或数据可视化。

总结一下，思路简单直觉，但是可以借鉴。
