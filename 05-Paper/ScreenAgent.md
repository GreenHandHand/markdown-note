# ScreenAgent: A Vision Language Model-driven Computer Control Agent

IJCAI-2024

本文着重于：**构建一个允许 Vision-Language Model 直接通过“看屏幕 + 输出鼠标/键盘操作”来控制真实计算机的通用交互环境，并配套提出一个 planning–acting–reflecting 的自动化控制 pipeline 与数据集。**

---

# Key Contribution

- We present a Reinforcement Learning (RL) environment that enables the VLM agent to directly interact with a real computer screen via VNC protocol.  
- We develop an automated pipeline that encompasses the planning phase, acting phase, and reflecting phase.  
- We propose the ScreenAgent dataset, which includes action sequences for completing generic tasks on Linux and Windows desktops.  
- We test GPT-4V and two state-of-the-art open-source VLMs, and train a ScreenAgent model with comparable performance to GPT-4V but more precise UI positioning.

- **提出了一个“真实计算机屏幕级”的 RL 交互环境**：状态是截图，动作是函数化的鼠标/键盘事件，通过 VNC 协议直接操作真实 OS，而非 HTML / 模拟 GUI。  
- **设计了一个完整的 plan–act–reflect 控制闭环**：不仅生成动作，还要求模型判断动作是否成功，从而支持连续多步交互。  
- **构建了 ScreenAgent Dataset**：覆盖 Windows/Linux 桌面上的通用日常任务，并提供细粒度的动作级评测指标（CC-Score）。  
- **实证分析了 GPT-4V 与开源 VLM 的差异**：GPT-4V 在规划与语义理解上强，但在精确坐标定位上明显不足；通过定向微调，小模型可在定位能力上反超。

---

# Method

![[98_Assets/ScreenAgent.png]]

## 方法整体目标

目标是在**无结构化 GUI 元数据（如 DOM、控件树）**的条件下，同时实现：
1. 对复杂自然语言任务的**高层规划能力**
2. 对真实计算机屏幕的**低层连续操作能力**
3. 在执行失败或环境变化时的**自我纠错与重规划能力**

---

![[98_Assets/ScreenAgent-1.png]]

## 输入与整体流程

输入预处理与整体流程如下：

1. 环境采集当前屏幕截图，作为状态 $s$
2. 将用户任务与截图输入 VLM
3. VLM 在内部 pipeline 中循环执行：
   - Planning：拆解任务
   - Acting：输出可执行动作
   - Reflecting：判断是否成功 / 是否需要调整
4. 环境执行动作，得到新截图 $s'$

---

## 核心机制拆解

### Planning Phase（任务分解）

模型输入：**Task Prompt + 当前 Screen**

输出：一组高层子任务（PlanAction）

- 子任务是**语义步骤**，而非具体点击坐标  
- 例如：“Open browser”“Search for XXX”“Browse results”

> [!note] Planning 的定位  
> 该阶段更接近 LLM 的“常识 + 工程流程理解”，而不是视觉决策。

> [!tip] 设计假设  
> 作者默认 VLM 能在“仅凭单帧截图”的情况下正确判断当前系统状态并生成合理计划，但论文未给出对规划错误的系统性分析或 ablation。

---

### Acting Phase（低层动作生成）

模型输入：**当前子任务 + 当前 Screen**

输出：**JSON 风格 function call 动作序列**

动作空间包括：

- Mouse：click / double_click / move / scroll / drag
- Keyboard：press / text
- Wait

> [!note] 动作即 API  
> 动作被显式建模为函数调用，这是 LLM tool-use 能力的直接映射。

> [!tip] 潜在问题  
> 让语言模型直接回归像素级坐标在归纳上并不自然，这也是 GPT-4V 在 Mouse Position 上表现不佳的根本原因。

---

### Reflecting Phase（执行评估）

模型输入：**执行后 Screen + 当前子任务**

输出三选一判断：

- `sub_task_success`
- `need_retry`
- `need_reformulate`

> [!note] Reflect 的灵感来源  
> 作者引用 Kolb Experiential Learning Cycle，将“反思”显式引入 agent loop。

> [!warning] 经验性假设  
> 实验表明即便是 GPT-4V，在 reflecting 阶段的判断 F1 也仅约 0.60，说明“单帧 + 语言推断成功与否”本身就是一个高噪声问题。

---

## Computer Control Environment

- **State**：单帧屏幕截图  
- **Action**：JSON 解析后的鼠标 / 键盘事件  
- **Reward**：未固定，接口开放（主要用于数据收集而非在线 RL）

> [!tip] RL 名称的模糊性  
> 虽然作者称其为 RL environment，但实际训练过程更接近 supervised fine-tuning + RLHF 数据收集，而非在线策略优化。

---

# Dataset & Metric

## ScreenAgent Dataset

- 273 个完整任务 session  
- 覆盖 39 种子任务类型  
- 涉及办公、浏览、系统操作、娱乐等真实桌面场景  
- 标注流程：GPT-4V 生成 → 人工修正 → RLHF pair

> [!note] 数据设计取向  
> 数据集刻意避开 DOM / 控件信息，逼迫模型“像人一样看屏幕”。

---

## CC-Score（Computer Control Score）

- 对动作序列进行 **顺序对齐**
- 对不同动作属性（类型 / 按键 / 坐标）分别打分
- 最终得到归一化序列相似度

> [!note] 指标优点  
> CC-Score 明确区分了“会不会调用工具”和“调用得准不准”。关于这个指标的详细内容暂时记录在 [[00_Inbox/关于 CC-Score|关于 CC-Score]] 里了。

---

# Experiment

总体评价：**方法论上成立，系统工程完整，但能力上仍高度受限于 VLM 的视觉定位与反思可靠性。**

![[98_Assets/ScreenAgent-2.png]]

- GPT-4V：规划与语义理解最强，但拒绝或错误输出精确坐标  
- 开源 VLM：规划弱，但可通过数据显著改善定位  
- ScreenAgent（微调后）：  
  - Mouse Position 明显优于 GPT-4V  
  - Planning 能力仍存在明显差距

> [!tip] 关键结论  
> 精确 GUI 操作 ≠ 大模型规模，而是强烈依赖**任务对齐的数据设计**。

> [!tip] Mouse Position 优势是否源于环境与分辨率偏置？
> ScreenAgent 在 Mouse Position 指标上的显著优势，**很可能部分来源于训练与测试环境的高度同分布性**，而非模型本身具备真正的跨屏幕几何泛化能力。需要注意的是，VNC 协议仅负责像素级 framebuffer 传输与输入事件转发，并**不会自动处理不同屏幕分辨率、DPI 或窗口布局之间的坐标对齐问题**。尽管论文在 CC-Score 评测阶段对坐标进行了归一化以保证分数可比性，但模型在推理时仍直接回归绝对（或归一化后的）像素坐标，这使其更容易拟合特定桌面环境中 UI 元素的固定位置分布。因此，ScreenAgent 在 Mouse Position 上优于 GPT-4V，更合理的解释是其**对作者所用桌面配置的高精度适配**，而非已经学会尺度不变的 GUI 定位能力；论文中缺乏跨分辨率或跨 DPI 的泛化实验，因而该结论的外推性仍然存疑。
>
> 尽管如此，作者的方法得到的结果对比没有微调过的 CogAgent 还是有很大的进步的。

---

# Future Work

作者提及：

- 支持多帧 / 视频输入
- 提升反思阶段的可靠性
- 扩展非英文界面支持

进一步值得探索的方向：

- 用显式中间表示（heatmap / region proposal）替代坐标回归  
- 将 reflecting 从“语言判断”升级为“状态差分判定”  
- 引入跨任务的长期记忆与失败模式归纳
