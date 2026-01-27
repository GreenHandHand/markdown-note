# Learning Abstractions for Hierarchical Planning in Program-Synthesis Agents

ICLR 2026 (Under Review, 4, 4, 4, 6)

**核心一句话**：这篇论文针对现有基于理论的强化学习（TBRL）系统依赖人工设计抽象的核心局限，提出 TheoryCoder-2——通过大语言模型（LLM）的上下文学习自动生成可复用的符号抽象（PDDL 格式），并融入分层规划流程，实现跨环境的抽象复用，最终在样本效率、泛化能力和复杂任务求解上超越现有 LLM 规划 Agent 和程序合成 Agent。

---

## Key Contribution

- Propose TheoryCoder-2, a TBRL agent that autonomously learns reusable abstractions in PDDL format via LLM in-context learning, eliminating the need for hand-engineered abstractions.
- Enable gradual growth of an abstraction library through curriculum learning, supporting transfer and composition of abstractions across diverse environments.
- Demonstrate superior sample efficiency, generalization, and ability to solve complex tasks compared to baselines (LLM + π, LLM + P, WorldCoder) on BabyAI, VGDL (Sokoban, Maze), and MiniHack environments.

- 突破 TBRL 的扩展性瓶颈：之前的 TheoryCoder、EMPA 等 TBRL 系统依赖人工抽象，无法适配新领域，TheoryCoder-2 的自动抽象学习让 TBRL 首次摆脱对人工工程的依赖，具备规模化应用潜力。
- 模仿人类的抽象学习模式：通过“从易到难”的课程学习积累抽象库，再通过抽象的复用与组合解决复杂任务，完美复刻了人类“先掌握基础概念，再组合解决复杂问题”的认知过程。
- 兼顾符号推理的高效性与 LLM 的泛化性：用 PDDL 符号抽象支撑高层规划（经典规划器可秒级求解），用 Python 世界模型建模低层动态，既避免了纯 LLM 规划的幻觉和高计算成本，又超越了传统 RL 的样本低效问题。

---

## Method

### 核心直觉

人类通过学习抽象概念（如“包含”“移动”）并组合它们进行规划，而现有 AI 要么依赖人工抽象（TBRL），要么缺乏结构化抽象（纯 LLM/RL）。作者的核心直觉是：LLM 的上下文学习能从少量示例中学会“如何表示抽象”，再通过分层规划将抽象与低层动作衔接，最后通过课程学习逐步积累抽象库——让 Agent 像人一样“学会学习抽象”。

### 1. Abstraction Learning via LLM In-Context Learning

> [!question] 为什么选择 LLM 上下文学习来生成抽象？
> 作者的动机是解决人工抽象的局限性，但为什么不用传统符号学习或微调模型？推测是因为上下文学习能快速适配新环境（少量示例即可），且 LLM 天然擅长生成结构化文本（PDDL 本质是结构化语言），无需大量标注数据，符合“少样本学习抽象”的人类特性。

这一模块的核心是让 LLM 自动生成 PDDL 格式的抽象表示，无需人工干预：
- **输入**：环境初始状态（文本化的面向对象表示）、1 个玩具问题的少量示例（与目标环境无关，仅示范抽象的表示方式）。
- **输出**：PDDL 的 domain 文件和 problem 文件：
  - Domain 文件：定义抽象操作符（operator），每个操作符包含参数、前置条件（precondition）和效果（effect），例如学到的 `moveontop` 操作符（Box 1）：`precondition` 是“未在目标对象上”，`effect` 是“在目标对象上”。
  - Problem 文件：定义当前任务的初始状态、目标条件（可多个子目标），仅使用环境状态字典中的对象名命名。
- **关键设计**：示例设计极简且通用，避免 LLM 过拟合到具体环境，引导其学习“抽象的结构”而非“具体的动作”。例如示例仅用“eat”“clear”等通用操作，让 LLM 能迁移到“move”“pickup”等新操作。

> [!tip] 精妙的约束设计
> 作者在 Prompt 中加入严格约束（如谓词名小写、无下划线、不包含空间关系词），这是避免 LLM 生成无效 PDDL 的关键——我注意到纯 LLM 生成结构化语言时容易出现格式错误，这些约束大幅提升了抽象生成的有效性，这是工程实现上的重要细节。

### 2. Hierarchical (Bi-Level) Planning

> [!note] 分层规划的核心逻辑
> 高层规划负责“做什么”（抽象目标分解），低层规划负责“怎么做”（具体动作序列），两者通过 Python 谓词分类器衔接，确保一致性。

![[Figure 1.png]]
如图 1 所示，TheoryCoder-2 的分层规划流程与基线（LLM+P、WorldCoder）的核心差异是“显式分层 + 抽象复用”，具体流程（对应 Algorithm 1）：
1. **高层规划**：将 PDDL 的 domain 和 problem 文件输入经典规划器（Fast Downward），生成抽象计划（序列 of 抽象操作符，如 `moveontop`→`pickup`→`unlock`）。
2. **低层规划**：用 BFS 算法结合 Python 世界模型（transition model），将每个抽象操作符映射为可执行的原始动作（如“上”“下”“拾取”）。
3. **衔接机制**：生成 Python 谓词分类器，用于检查低层动作执行后是否满足抽象操作符的 `effect`（例如执行“移动”动作后，是否真的达成 `moveontop` 的效果），确保高低层一致性。

> [!warning] 这里存在一个隐含假设
> 该模块依赖环境状态是“文本化的面向对象表示”（如 `agent: [3,4], apple: [5,4]`），但现实中很多环境（如视觉游戏）没有这种结构化输入——作者未讨论如何从非结构化输入（图像）中提取这种表示，这是应用场景的重要局限。

### 3. Reusing and Growing Abstraction Library via Curriculum Learning

> [!question] 为什么需要课程学习？
> 作者的直觉是人类抽象学习是渐进的（先学简单概念，再学复杂概念），但课程学习是否是必需的？消融实验（TC-C）显示，即使去掉课程学习，Agent 仍能解决任务，但样本效率下降——说明课程学习是“加速器”而非“必需品”，核心还是抽象的可复用性。

这一模块的核心是让抽象库随环境交互逐步扩展：
- **课程设计**：环境按难度分组，从简单到复杂（如先 Labyrinth→Maze→Sokoban，再到 BabyAI Boss level）。
- **抽象复用**：在简单环境中学到的抽象（如 `moveontop`）可直接用于后续复杂环境，无需重新生成；新环境仅需学习新增抽象（如 `pickup`→`unlock`）。
- **动态优化**：低层世界模型（Python transition function）可通过 replay buffer 中的过渡数据持续优化，而抽象库仅在新增技能时扩展，平衡稳定性和适应性。

### 4. Algorithm 1 核心流程梳理

1. 生成 PDDL 抽象和初始世界模型；
2. 高层规划生成抽象计划；
3. 低层规划将抽象操作映射为原始动作并执行；
4. 存储过渡数据，检查目标是否达成；
5. 若未达成，优化世界模型或生成新探索计划，重复执行。

---

## Experiments

> [!note] 指标选择的合理性
> 作者使用三个核心指标：
> - Token cost：衡量样本效率（LLM 生成内容的 token 数，越少说明学习越高效）；
> - Compute time：衡量实际运行效率（壁钟时间，避免“样本高效但耗时过长”的伪优势）；
> - Solution rate：衡量任务求解能力（首次尝试成功率，体现泛化性）。
> 这些指标覆盖了“效率 - 效果”两个维度，且针对 LLM Agent 的特性（token 消耗是主要成本），比传统 RL 的“步数”指标更贴切。

### 实验设计与关键结果

1. **环境选择**：
   - 简单环境（VGDL）：Labyrinth、Maze、Sokoban（核心测试抽象学习与复用）；
   - 复杂环境（BabyAI）：Pickup、Unlock、Boss level（核心测试抽象组合与复杂任务求解）；
   - 跨域环境（MiniHack）：5x5 房间、15x15 房间、陷阱、怪物、Wand of Death（核心测试抽象迁移性）。
2. **基线对比**：
   - LLM + π：直接生成原始动作，无抽象；
   - LLM + P：生成 PDDL 但不分层，无抽象复用；
   - WorldCoder：生成 Python 世界模型，无高层抽象；
   - Oracle：人工抽象（上限参考）；
   - 消融变体：TC-P（无执行性抽象和世界模型）、TC-C（无课程学习）。

3. **核心发现**：
   - 样本效率：TheoryCoder-2 的总 token cost（121,529）远低于基线（LLM+P：443,307；WorldCoder：437,719），且在复杂任务中优势更明显（BabyAI Boss level token cost 仅 2000 左右，而基线均超 40,000）。
   - 任务求解：仅 TheoryCoder-2 能解决 BabyAI Boss level 的多个变体（Combined Skills 1/3），其他基线均失败——证明抽象组合能力是解决复杂任务的关键。
   - 抽象复用：MiniHack 中，学到的 `moveontop` 抽象可直接复用，导致 15x15 房间、陷阱、怪物环境的 token cost 为 0，体现跨环境迁移性。
   - 运行时间：如图 3 所示，TheoryCoder-2 比高推理强度的 LLM+π快得多（LLM+π需 3 分钟，TheoryCoder-2 秒级响应），因为经典规划器比 LLM 推理更高效。

![[Figure 3.png]]

> [!warning] 实验设计的潜在漏洞
> - 模型差异：TheoryCoder-2 用 GPT-4o 生成抽象，而 LLM+P 用 o4-mini 生成 PDDL——GPT-4o 的能力可能优于 o4-mini，这是否影响结果？作者通过 TC-P（用 o4-mini）的消融实验部分验证，但未完全排除模型能力的干扰。
> - 环境限制：所有环境均为文本化的面向对象表示，未测试视觉输入——这让方法的适用性打折扣，无法确定在非结构化输入中是否有效。
> - 抽象质量评估：仅通过性能间接证明抽象质量，缺乏定量的抽象“简洁性”“通用性”评估（如抽象的参数数量、跨环境复用次数）。

### Ablation Study 关键结论

- TC-P（去掉可执行抽象和 Python 世界模型）：成功率大幅下降，token cost 飙升——证明“可执行的符号抽象”和“低层世界模型”是高效规划的核心，纯自然语言抽象缺乏一致性和可执行性。
- TC-C（去掉课程学习）：仍能解决任务，但样本效率下降——说明课程学习是优化项，核心价值在于加速抽象积累，而非抽象学习本身的必需条件。

---

## Future

### 作者提及的方向

1. 扩展到视觉环境：需要视觉 - 语言模型提取面向对象的状态表示，解决目标检测、跟踪和属性推断问题。
2. 处理连续域：现有方法针对离散动作和状态，需建模物理规律（如速度、接触）以适配连续环境。
3. 改进抽象鲁棒性：谓词分类器在边缘情况（如多门环境）易失效，需通过试错修订抽象。
4. 动态修订抽象：目前抽象仅在初始生成，未根据新观测修订，需加入抽象迭代优化机制。

### 我认为值得挖掘的潜在方向

1. 自动课程生成：现有课程需人工指定难度顺序，未来可让 Agent 自主评估环境难度，动态调整学习顺序，提升自主性。
2. 多模态抽象学习：从图像、文本等多模态输入中学习抽象，摆脱对文本化状态的依赖，扩大应用场景。
3. 抽象的元学习：学习“如何生成抽象”的元知识，而非仅学习具体抽象，提升跨领域迁移的泛化性（如从导航任务迁移到操作任务）。
4. 轻量化抽象生成：目前依赖 GPT-4o 等大模型，未来可微调小型语言模型生成抽象，降低计算成本，适配边缘设备。
5. 抽象的可解释性评估：建立定量指标评估抽象的人类可解释性，让 Agent 学习的抽象更符合人类认知，便于人机协作。
