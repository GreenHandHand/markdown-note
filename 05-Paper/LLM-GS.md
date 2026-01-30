# Synthesizing Programmatic Reinforcement Learning Policies with Large Language Model Guided Search

ICLR-2025

**核心一句话**：这篇论文要解决程序化强化学习（PRL）样本效率极低（需数千万次环境交互）的问题，核心思路是用大语言模型（LLM）的编程能力和常识缩小搜索范围，再通过专门设计的搜索算法优化程序，最终让不懂编程和领域知识的用户也能靠自然语言描述任务得到高效策略。

---

## Key Contribution

- Propose LLM-GS, a novel framework that leverages LLMs' programming expertise and common sense to bootstrap sample efficiency of search-based PRL methods.
- Design Pythonic-DSL strategy to address LLMs' inability to generate precise, grammatically correct domain-specific language (DSL) programs.
- Develop Scheduled Hill Climbing algorithm to efficiently explore the programmatic search space and consistently improve LLM-generated programs.
- Demonstrate LLM-GS's effectiveness across Karel and Minigrid domains, and its extensibility to novel tasks without requiring user's domain/DSL knowledge.

- LLM-GS 打破了传统 PRL 算法“无假设盲目搜索”的局限，首次将 LLM 的“人类兴趣范围”知识引入搜索初始化，从源头减少无效探索，这是样本效率提升的核心逻辑。
- Pythonic-DSL 没有强行让 LLM 学习陌生的 DSL，而是利用其擅长的 Python 作为过渡，既发挥了 LLM 的编程优势，又通过规则转换保证了 DSL 程序的正确性，比直接生成 DSL 更务实。
- Scheduled Hill Climbing 针对 LLM 初始化的“高质量起点”设计了动态搜索策略，解决了固定邻域大小的 Hill Climbing 在不同任务上适应性差的问题。
- 首次验证了 PRL 对“非专业用户”的友好性，用户只需用自然语言描述任务，无需了解 DSL 或领域细节，就能获得可用策略，拓宽了 PRL 的应用场景。

---

## Method

### 整体框架：LLM-Guided Search (LLM-GS)

> [!question] 核心动机
> 传统 PRL 算法（如 Hill Climbing）需要数千万次环境交互，本质是因为搜索空间过大且无任何引导——相当于在茫茫大海里捞针。作者的直觉是：人类关心的任务对应的有效策略往往是有限的，LLM 学习了海量人类知识，应该能提供“捞针的范围提示”，从而大幅减少无效搜索。

LLM-GS 分为两步：第一步用 LLM 生成初始 DSL 程序，第二步用 Scheduled Hill Climbing 优化这些程序。整体流程如图 3 所示：

![[98_Assets/LLM-GS.png]]
- 图 3(a) 是 LLM 生成 DSL 程序的过程：输入任务描述和 Pythonic-DSL 指令，LLM 先输出 Python 程序，再通过规则转换为 DSL 程序。
- 图 3(b) 是搜索优化过程：LLM 生成的程序作为初始种群，Scheduled Hill Climbing 动态调整邻域大小，逐步找到更优程序。

> [!warning] 潜在疑问
> 作者为什么不直接让 LLM 迭代优化程序，而是要结合搜索算法？后续实验显示，LLM 自身修订程序的性能会快速饱和（图 10），且调用成本高，而搜索算法能*免费*且持续地优化，二者结合是性价比最高的选择。

### 模块一：Domain and Task-Aware Prompting

> [!question] 设计动机
> LLM 本身不懂 PRL 任务的环境动态（比如 Karel 机器人能做什么动作、能感知什么），也不知道具体任务目标。直接让 LLM 写 DSL 程序，相当于让一个不懂游戏规则的人去写游戏攻略——必然失败。所以这个模块的核心是“教 LLM 规则”，但不泄露解题思路。

作者设计的提示包含两部分：
1. 领域知识：告诉 LLM 环境的基本机制（比如 Karel 机器人的动作、感知功能、墙壁和标记物的作用）。
2. 任务知识：用自然语言描述任务的地图、初始位置、目标和奖励规则（如表 1 中 DOORKEY 任务的描述）。

> [!tip] 精妙设计
> 提示没有给出任何“如何解题”的暗示，只是提供了“游戏规则”，这样 LLM 只能依靠自身的编程和推理能力生成策略，避免了直接泄露答案，保证了方法的通用性。

### 模块二：Pythonic-DSL Strategy

> [!question] 设计动机
> LLM 擅长通用编程语言（如 Python），但不熟悉 PRL 专用的 DSL——DSL 可能有特殊语法（如 Karel 的控制流格式）和硬件约束（如不能用临时变量），直接生成 DSL 程序会出现大量语法错误或不可执行的情况。作者的直觉是：既然 LLM 会写 Python，那就先让它写 Python 程序，再转换为 DSL，降低生成难度。

具体步骤：
1. 给 LLM 提供 Pythonic-DSL 指令：明确 Python 中可用的动作（如 move()）、感知（如 frontIsClear()）和约束（如不能定义变量、只能有 run() 函数）。
2. LLM 生成 Python 程序：利用其熟练的 Python 编程能力，生成符合任务要求的程序。
3. 规则转换为 DSL：根据预设的转换规则（如表 4），将 Python 程序转换为目标 DSL 程序。

Karel DSL 的语法如图 1 所示，包含动作、感知和控制流：
![[98_Assets/LLM-GS-1.png]]

> [!tip] 关键优势
> 如表 3 所示，Pythonic-DSL 策略生成的可执行程序比例（Acceptance Rate）和最优回报（Best Return）均高于直接生成 Python 或 DSL 程序——证明了“过渡语言”的有效性。

### 模块三：Scheduled Hill Climbing

> [!question] 设计动机
> 传统 Hill Climbing 用固定的邻域大小（比如每次生成 250 个邻域程序），但不同任务的难度不同：简单任务可能需要小邻域快速收敛，复杂任务可能需要大邻域探索更广阔的空间。作者的直觉是：LLM 生成的初始程序质量已经不错，初期可以用小邻域微调，若长时间没找到最优解，再逐步扩大邻域，平衡探索和利用。

#### 核心公式与直觉

作者设计的调度函数用于动态调整邻域大小 \( k(n) \)：
$$log _{2}k(n)=(1-r(n))log _{2}K_{start}+r(n)log _{2}K_{end}$$
$$r(n)=sin\left[\left(\frac{2 log n}{log N}-1\right) × \frac{\pi}{2}\right]+1$$

- 直觉：邻域大小 \( k(n) \) 随已评估程序数 \( n \) 指数增长，从初始值 \( K_{start} \)（默认 32）增长到最大值 \( K_{end} \)（默认 2048）。
- 参数解释：
  - \( n \)：已评估的程序总数。
  - \( N \)：最大评估程序数（预算上限）。
  - \( r(n) \)：正弦函数，从 0 平滑增长到 1，控制邻域增长速度——初期增长慢（小邻域），中期增长快（扩大探索），后期趋于平稳（大邻域精细搜索）。

#### 优化过程

1. 初始化：将 LLM 生成的程序作为初始搜索中心。
2. 生成邻域：根据当前 \( k(n) \) 生成 \( k \) 个邻域程序（通过修改 AST 节点实现）。
3. 评估与更新：在环境中评估邻域程序的回报，若找到更优程序，则更新为新的搜索中心。
4. 重复：直到找到最优程序或达到预算上限。

> [!warning] 设计细节
> 作者为什么用对数插值而非线性插值？因为程序的搜索空间是 AST 结构，复杂度随程序长度指数增长，邻域大小指数增长能更好地匹配搜索空间的复杂度——线性增长可能在复杂任务中探索不足，对数增长则能平衡效率和探索范围。

### 示例：DOORKEY 任务的程序优化

DOORKEY 是 Karel-Hard 中的难点任务（图 2），需要分两步：先在左房间捡钥匙（标记物），再到右房间放标记物，奖励稀疏且容易陷入局部最优。

![[98_Assets/LLM-GS-2.png]]

LLM 生成的初始程序已经具备两阶段结构，但缺乏导航能力；经过 Scheduled Hill Climbing 优化后，程序增强了导航逻辑，最终能完成任务（图 5）：

![[98_Assets/LLM-GS-3.png]]

- 左图（初始程序）：有捡标记物和放标记物的两阶段结构，但导航逻辑简单，无法找到钥匙和目标位置。
- 右图（优化后程序）：添加了更灵活的转向和移动逻辑，能在左房间探索找到钥匙，再通过开门进入右房间完成目标。

---

## Experiments

> [!note] 关键指标解释
> 实验的核心指标是**样本效率**——即达到特定回报所需的程序评估次数（每次评估包含 32 个任务变体的平均回报）。这个指标直接对应 PRL 的核心痛点：评估次数越少，环境交互成本越低，越接近实际应用。

### 实验设置

- 任务集：Karel（6 个基础任务）、Karel-Hard（4 个复杂任务）、Minigrid（3 个新领域任务）、2 个 novel Karel 任务（PATHFOLLOW、WALLAVOIDER）。
- 基线方法：LEAPS、HPRL、CEBS、Hill Climbing（当前 SOTA）。
- LLM：GPT-4（gpt-4-turbo-2024-04-09），Minigrid 实验用 GPT-4o。

### 核心结果

1. 样本效率大幅超越基线（图 4）：

![[98_Assets/LLM-GS-4.png]]
- 在 Karel 任务中，LLM-GS 能快速收敛到最优回报（1.0），而基线方法需要更多评估次数。
- 在 Karel-Hard 的 DOORKEY 任务中，基线方法即使评估 100 万次也无法收敛（回报约 0.5），而 LLM-GS 仅需 50 万次左右就能收敛到 1.0。

2. 消融实验验证关键模块的有效性：
   - Pythonic-DSL 策略：在 9/10 任务中可执行程序比例最高，7/10 任务中最优回报最高（表 3）。
   - Scheduled Hill Climbing：比固定邻域大小的 Hill Climbing 更适应不同任务（图 6）。
   - LLM 初始程序：比随机初始化的搜索效率高得多（图 7）——证明 LLM 的引导作用是核心。

3. 泛化性验证：
   -  novel 任务：仅修改任务描述，LLM-GS 仍能保持高样本效率（图 8）。
   - 新领域（Minigrid）：适配 Minigrid 的 DSL 后，LLM-GS 依然比 Hill Climbing 高效（图 9）。

### 局限性分析

> [!warning] 潜在漏洞
> 1. 依赖强 LLM：实验用的是 GPT-4/GPT-4o，若使用开源或能力较弱的 LLM，初始程序质量可能下降，样本效率优势是否还存在？作者未验证。
> 2. 任务复杂度上限：实验任务均为网格世界的离散控制任务，若扩展到连续控制或更复杂的环境（如机器人操作），DSL 设计难度会大幅增加，LLM-GS 的有效性是否能保持？
> 3. 基线选择：CEBS 和 Hill Climbing 是当前 SOTA，但作者是否对比了其他结合 LLM 的 PRL 方法？比如直接用 LLM 生成 DSL 程序后微调的方法。
> 4. 数据泄露风险：作者虽在附录 G 中论证了 GPT-4 未见过 Karel-Hard 的最优程序，但无法完全排除 LLM 学习过类似任务的策略，可能存在隐性数据泄露。

---

## Future

### 作者提及的方向

1. 开发更复杂、更贴近现实的 PRL 环境，并设计对应的 DSL。
2. 降低对强 LLM 的依赖，探索用开源 LLM 或小模型实现类似效果。
3. 减少对领域专家的依赖——当前仍需要专家设计 DSL 和领域提示，未来希望能自动生成 DSL。

### 潜在拓展方向（个人观点）

1. 结合 LLM 的反馈机制：当前 LLM 仅用于生成初始程序，可尝试将搜索过程中的反馈（如哪些程序表现好、为什么失败）反馈给 LLM，让 LLM 生成更优的初始程序，形成闭环。
2. 多 LLM 协作：不同 LLM 可能擅长不同类型的任务，可尝试融合多个 LLM 的初始程序，进一步扩大初始搜索的覆盖范围。
3. 连续控制任务适配：设计适用于连续控制的结构化策略表示（如分段线性控制器），将 LLM-GS 扩展到连续控制领域。
4. 动态 DSL 生成：让 LLM 根据任务描述自动生成适配的 DSL，彻底摆脱对领域专家的依赖——这可能是 PRL 走向通用的关键。
