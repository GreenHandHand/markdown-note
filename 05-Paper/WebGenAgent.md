# WebGen-Agent: Towards Visual Feedback Driven Web Generation

arxiv, 2025.11

本文着重于：一种通过 视觉反馈闭环（Screenshot + GUI Agent）来指导 Web 代码生成与强化学习训练的 Agentic Web Generation 框架。

> [!note] GRPO (Group Relative Policy Optimization)  
> 一种相对策略优化方法，通过比较一组候选策略的相对表现来更新模型，弱化对绝对 reward 标定的依赖，常用于 LLM 强化学习场景。

## Key Contributions

- Propose WebGen-Agent, a web generation framework that incorporates visual feedback from screenshots and a GUI testing agent.
- Introduce Step-GRPO, which leverages screenshot-based and GUI-based feedback as reward signals for training.

- 作者提出了 WebGen-Agent，将网页生成从纯文本 → 代码的单向生成，重构为一个 *以真实网页渲染结果为中心的闭环系统*。核心转变在于：网页是否“正确”，不再仅由代码语义或 LLM 自评决定，而是由 *视觉层面的可观察状态* 与 *可交互行为* 来共同裁决。
- 在系统层闭环之外，作者进一步将视觉与交互反馈“制度化”为强化学习信号，提出 Step-GRPO，使得模型不仅在推理阶段可被纠正，在参数层面也能逐步内化“什么样的网页是好的”。

## Method

- 首先是提出的网页生成系统 WebGen-Agent，该系统利用屏幕截图和 GUI agent 生成的内容作为反馈，来迭代优化生成网页的功能和外观。
- 在网页生成系统的基础上，再引入了 Step-GRPO 方法。该方法利用屏幕截图和 GUI agent 反馈作为分数，利用 GRPO 对模型进行训练。

### WebGen-Agent

流程非常直观：

1. Code Agent 生成代码。
2. 执行代码，获取控制台输出。（报错则重新生成）
3. 获取网页截图。利用 VLM 检查网页实现效果，输出分数、修改建议。
	1. VLM 检查网页是否报错，如果报错，则重新生成代码。
	2. VLM 根据截图给出描述，并给出修改建议。
4. 将（输入，生成代码，控制台输出，描述，分数，修改建议）全部返回给 Coding Agent，由 Coding Agent 判断是否实现了功能。
5. 如果实现了功能，则开启一个 GUI Test 对话，用一个 GUI Agent 判断网页实现是否正确。
	1. 按照输入指令，生成 Web-navigation 的 GUI Agent 的指令，
	2. GUI Agent 根据指示进行互动，并生成分数。

每一步生成一段正确的代码，以此迭代。

### Step GROP

该部分使用前面的屏幕截图分数和 GUI 分数作为强化学习的目标。

包含一个特别处理：利用 DeepSeek-V3 事先生成了 700 个策略，并进行 SFT，作为 Warm start。

总结一下，思路简单，但是可以借鉴。
