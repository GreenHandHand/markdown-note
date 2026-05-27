# EduMirror: Modeling Educational Social Dynamics with Value-driven Multi-agent Simulation

ICML-2026

**核心一句话**：作者想把教育场景中的同伴互动、欺凌、课堂治理、家校关系等社会动态，转成一个可控的 LLM 多智能体仿真实验室。它的核心差异在于，智能体的行动不只由 prompt 扮演角色驱动，还被心理需求和社会价值取向约束，并且用双轨测量同时追踪外显行为和隐性心理状态。

---

## Key Contribution

- A Controllable Simulation Framework for Educational Social Dynamics
- Value-Driven Agents for Educational Role Play
- A User Toolkit for Educational Study
- Case Studies & Counterfactual Analysis

作者的贡献可以拆成三层。第一层是**实验场景层**，EduMirror 把教育社会现象组织成可配置场景，并支持用户在关键节点保存状态、施加干预、生成平行时间线。这个设计针对的是教育研究里的一个硬问题：真实学生身上很难做强干预实验，尤其是欺凌、排斥、惩罚这类高风险情境。

第二层是**智能体动机层**。作者不满足于让 LLM 直接扮演学生或教师，而是给智能体加入心理需求系统和社会价值取向。这里的核心判断是：教育社会行为经常来自内部状态，比如安全感、自尊、归属感、心理健康，而这些状态在行为日志中并不会直接出现。

第三层是**测量层**。作者引入 LLM Rater 和 LLM Surveyor，前者对完成后的互动轨迹做行为评分，后者对智能体内部状态做后验问卷测量。这个设计试图把多智能体仿真从故事生成推进到可分析的计算实验。

> [!tip] 我注意到的关键点
> 这篇论文真正有价值的地方不在于又做了一个教育 Agent 平台，而在于它试图把**生成式仿真、心理变量、反事实干预、后验测量**接到同一条实验链路上。对于 Agent for Education 方向，这种链路比单个任务效果更重要。

---

## Method

### Overall Framework

**Intuition**：作者的直觉是，教育社会动态可以被看作一个复杂多智能体系统，单次问答无法表达它的演化过程，因此需要一个能持续运行、能干预、能测量的仿真平台。

![[Figure 1 EduMirror Core Concept.png]]

Figure 1 把 EduMirror 画成一面镜子。左侧是真实教育场景，包括个体行为、家校互动、同伴群体、课堂文化；右侧是仿真系统，通过 what-if 分支测试不同干预策略。这里作者想表达的是，平台不是直接给教育建议，而是先模拟不同策略可能引发的社会后果，再辅助研究者或教育者判断。

![[Figure 2 EduMirror Architecture.png]]

Figure 2 是整篇论文最重要的结构图。系统由四块组成：Agent Model Repository、Scenario Design、Concordia-based Simulation Engine、User Toolkits。Scenario Design 负责把理论、情境、角色、指标组织起来；Simulation Engine 由 Game Master 管理场景、规则、叙事和时间；User Toolkits 负责测量、可视化和干预分支。

> [!question] 动机是否充分
> 作者对教育实验的伦理限制讲得比较充分，对传统 ABM 的僵硬规则也讲得合理。但我注意到一个隐含前提：只要 LLM 生成的互动足够像真实案例，它就能用于反事实教育研究。这个前提需要非常谨慎，因为**像真实**和**具有因果可信度**之间还有明显距离。

### Theory-Grounded Scenario Design

**Intuition**：作者希望场景不是随便写 prompt，而是从教育学或心理学理论出发，把理论概念变成智能体配置和测量指标。

作者给出五步流程：

1. 选择 grounding theory，例如 Social Comparison Theory。
2. 拆出 core constructs，例如 upward comparison、self-esteem。
3. 映射到 agent persona，包括 traits、goals、formative memories。
4. 用 validated scales 操作化这些 constructs。
5. 用 LLM Rater 和 LLM Surveyor 建立双轨测量协议。

![[Figure 3 Scenario Library Distribution.png]]

Figure 3 显示场景库有 20 个预设教育场景，覆盖 Peer & Group Dynamics、Individual Social Cognition、Classroom Culture、Home-School Dynamics 四类主题，比例分别是 35%、25%、15%、25%。这些场景运行在课堂、宿舍、操场、食堂、家庭、教师办公室、体育馆、图书馆等预配置环境中。

> [!tip] 这里值得借鉴
> 作者没有把场景设计写成纯 prompt engineering，而是把它包装成**理论到构造变量再到测量协议**的流程。这个写法对教育 Agent 研究很重要，因为它让仿真任务更像研究设计，而不是 demo 编排。

> [!warning] 潜在问题
> 场景库虽然覆盖面较广，但固定角色配置和固定测量协议可能会让系统对作者预设的理论结构过拟合。换句话说，它更适合测试已知理论下的 what-if，而不一定适合发现全新的教育社会机制。

### Value-driven Cognitive Architecture for Agents

**Intuition**：作者认为教育场景中的行为需要内部动机支撑。学生被欺凌后的退缩、反抗、求助，或者班级竞选中的合作、竞争，都不能只靠角色描述生成，需要一个能随互动更新的价值系统。

作者把智能体行为形式化为条件生成：

$$
a_t \sim Agent_\phi(\cdot \mid H_{<t}, I, P, e)
$$

其中，$a_t$ 是第 $t$ 步生成的活动，$H_{<t}$ 是历史行动和观察，$I$ 是额外定制信息，比如目标、需求状态、社会偏好，$P$ 是 agent profile，$e$ 是环境上下文。这里的重点是，作者把 Agent 看作一个带条件的序列生成器，然后把心理需求和 SVO 放进条件中。

#### Individual Value System

**模块目的**：Individual Value System 用来追踪个体心理需求的满足程度，让智能体行为受到安全感、自尊、归属感、心理健康、意义与成长等变量影响。

作者定义未满足需求差距：

$$
\Delta_t(d)=clip(v^*(d)-v_t(d),0,S_{max})
$$

这里 $d$ 是某个心理需求维度，$v^*(d)$ 是期望值，$v_t(d)$ 是当前值，$\Delta_t(d)$ 表示当前状态距离理想状态有多远。这个 gap 越大，说明该需求越缺失，后续行动评估时就越应该考虑如何修复它。作者把五大类心理需求扩展为 13 个子维度，每个维度用 0 到 10 的 Likert 分数表示。

> [!note] 这里的建模含义
> 这个系统本质上是在给 LLM 一个可更新的心理状态表。它不是让 LLM 自由想象 Alice 很难过，而是让 Alice 的安全感、归属感、自尊等变量在互动后发生变化，然后再影响下一步行动。

#### Social Value System

**模块目的**：Social Value System 用来控制智能体在自我满足和他人收益之间如何权衡。

作者引入 Social Value Orientation，将智能体初始化为 Altruistic、Prosocial、Individualistic、Competitive 等类型。系统计算两个信号：

$$
S_{self}(t)=clip(\sum_d \Delta_t(d),0,S_{max})
$$

$$
S_{other}(t)=clip(\sum_d \Delta_t(d),0,S_{max})
$$

然后用角度表示社会价值取向：

$$
\theta_t=\arctan \left(\frac{S_{other}(t)+\epsilon}{S_{self}(t)+\epsilon}\right)
$$

当 $\theta_t$ 较小时，智能体更偏向自我相关需求；当 $\theta_t$ 较大时，智能体更偏向他人相关结果。作者把这个角度作为 prompt-level control signal 传给 planner，让行动评估时显式考虑自我需求、他人影响和 SVO profile 的一致性。

> [!warning] 这里有一个反直觉点
> SVO 本来是社会心理学中的偏好测量概念，作者把它变成了生成行为时的控制信号。这个转换很巧妙，但也带来问题：LLM 是否真的按照角度连续变化来改变行为，还是只是被文字标签引导出符合刻板印象的行为？这需要更强的消融和机制分析。

#### Value-driven Planner

**模块目的**：Planner 负责把心理需求和社会价值取向转成具体行动。它先生成候选行动，再比较这些行动对需求满足和他人影响的效果，最后选择一个行动。

$$
A_t=Gen_\phi(H_{<t},P,e,I,\Delta_t,\theta_t)
$$

$$
q_a=E_\phi(a\mid \Delta_t,\theta_t)
$$

$$
a_t=\arg\max_{a\in A_t}q_a
$$

这里 $A_t$ 是候选动作集合，$q_a$ 是 LLM 给出的比较评分，$E_\phi$ 不是手写 utility function，而是结构化 LLM judgment。作者在 appendix 中进一步说明，候选动作不会被转换成显式数值效用，而是通过当前 SVO 作为上下文控制信号，引导 LLM 比较自我满足和他人满足。

![[Figure 9 Prosocial Agent SVO Trajectory.png]]

Figure 9 展示了一个 prosocial agent 的 SVO 轨迹。Alice 在数学讨论、竞赛、求助老师、解释题目等行为之间波动，但整体保持 prosocial orientation。作者想证明的是，SVO 不是固定标签，而是会受具体行动影响产生短期波动，同时又被 profile-specific reference interval 稳定住。

> [!question] 原文没有完全讲清楚
> 作者说 $E_\phi$ 是 structured LLM-based judgment，但主文没有详细展开评分格式、候选动作数量、比较过程是否稳定。这里会影响可复现性。Appendix 提供了 prompt 模板，但如果模型换成另一个 LLM，planner 的行为稳定性仍然是疑问。

### Dual-Track Measurement Protocol

**Intuition**：作者意识到只看互动文本不够，因为教育社会动态里最重要的变量往往是隐性的，所以需要同时测量外显行为和内在心理状态。

![[Figure 2 Dual Track Measurement.png]]

LLM Rater 读取完成后的 interaction traces，对可观察行为评分。LLM Surveyor 读取仿真中记录的内部状态，后验发放心理量表，测量自尊、SVO 等心理构念。这个分离很关键，因为 Surveyor 不参与仿真过程，理论上不会改变智能体互动轨迹。

> [!tip] Aha moment
> 这个 dual-track 的真正价值是把 simulation log 转成可量化变量。没有这一步，EduMirror 只是在生成故事；有了这一步，它才有机会进入实验比较和干预评估。

> [!warning] 测量闭环风险
> Rater 和 Surveyor 都是 LLM，Agent 也是 LLM。作者虽然把测量放到 post-hoc，但仍然存在同源偏差：生成器和评价器共享相似语言先验，可能会高估仿真的心理合理性。尤其是 Surveyor 测量的变量，有一部分已经被编码进 agent internal state，验证结果可能包含自证成分。

---

## Experiments

### System-Level Validation

作者先做系统级验证，覆盖 17 个教育场景，包括 6 个代表性场景、Case Study 1 的 8 个场景、Case Study 2 的 3 个场景。baseline 包括 ReAct、BabyAGI、LLMob、JAG-Concordia、D2A。评价重点是场景级 realism 和 human-likeness，而不是某个单一任务的成功率。

![[Figure 4 Pairwise Win Rate Heatmap.png]]

Figure 4 是模型之间的 pairwise win-rate heatmap。作者用它说明 EduMirror 在多数场景中胜过 baseline，尤其在需要心理状态和社会价值一致性的场景中更稳定。这里的实验定位很清楚：它不是证明模型完成任务更强，而是证明生成的社会过程更像人类互动。

作者还在 kindergarten scenario 中测试 5、15、30 个 agents 的可扩展性。EduMirror 的平均分分别是 4.80、4.18、4.03，均高于 LLMob、BabyAGI、D2A、ReAct。指标包括 Naturalness、Coherence、Plausibility、Developmental Typicality，其中 Developmental Typicality 检查行为和情绪反应是否符合幼儿发展阶段。

> [!note] 重要指标解释
> Developmental Typicality 是教育场景中特别有意义的指标。普通社会仿真只要求行为自然，教育仿真还要求角色年龄、认知发展阶段、道德判断方式合理。

> [!warning] 实验漏洞
> 系统级验证高度依赖 LLM-based post-hoc evaluation。虽然作者加入了部分 human evaluation，但整体判断仍然可能偏向语言流畅性、叙事完整性和心理描写充分性。真正的外部效度需要和真实纵向数据、课堂观察数据或专家标注进行更强对齐。

### Case Study 1: School Bullying Simulation

**Intuition**：作者用欺凌场景验证 Individual Value System，因为欺凌是一个典型的心理状态动态变化问题。受害者的安全感、归属感、自尊、心理健康会随互动持续恶化，也会因干预策略不同而恢复。

![[Figure 5 Psychological Need Dynamics under Bullying.png]]

Figure 5 对比了不同初始状态下 Alice 的心理需求曲线。初始健康状态下，Alice 更有韧性；初始脆弱状态下，心理变量波动更剧烈，更容易进入受害轨迹。这里作者想证明，Individual Value System 不只是装饰性变量，它会改变行为生成和心理演化路径。

作者还做了真实案例和模拟案例辨别实验。实验使用 10 个真实欺凌案例和 10 个模拟案例，并用 GPT-4o 统一重写叙事风格，152 名参与者判断哪些是真实案例。结果显示，多组参与者区分准确率较低，6 组低于 30%，所有组都有超过 10% 选择 difficult to distinguish，第 6 组达到 52.63%。作者据此说明模拟案例具有较强叙事真实性。

![[Figure 13 Human Evaluation Real vs Simulated Bullying.png]]

> [!warning] 这里需要谨慎
> 这个实验能说明模拟文本难以和真实新闻式叙事区分，但不能直接说明系统掌握了欺凌的生成机制。因为 GPT-4o 统一改写可能消除了真实案例和模拟案例的语言差异，也可能把两者都拉向同一种叙事模板。

![[Figure 6 Teacher Intervention Strategies.png]]

Figure 6 比较四种教师干预策略：Neglectful Intervention、Authoritative-Punitive、Supportive-Individual、Supportive-Cooperative。结果显示，忽视会让心理需求持续下降，权威惩罚能改善安全感和归属感但对自尊和意义感有限，个体支持带来中等改善，合作式支持在五个心理需求维度上效果最好。

![[Figure 15 Internal Value vs RSES Score.png]]

Appendix 中的 Figure 15 用 RSES 问卷变化验证内部心理变量变化。作者把 Self-Worth 和 Sense of Respect 的变化作为内部自尊指标，并观察它与外部 RSES score 的变化呈正相关。这个结果用来支持 Individual Value System 的 construct validity。

> [!tip] 对教育 Agent 研究的启发
> 欺凌仿真给出了一个比较好的任务原型：同一个初始场景，施加不同教师策略，然后比较受害者心理状态曲线。这比单纯让 Agent 回答如何处理欺凌更接近可评估的教育研究任务。

### Case Study 2: Social Interaction Simulation

**Intuition**：作者用同伴互动和班级竞选验证 Social Value System，因为这些场景需要区分稳定人格倾向，比如利他、亲社会、个人主义、竞争。

作者选择了三个复杂度递增的场景：小组学习资源共享、班级协作任务、班长竞选。agents 被赋予 Altruistic、Prosocial、Individualistic、Competitive profile，然后用 LLM Rater 识别合作与竞争行为。

![[Table 6 SVO Ablation Behavioral Distribution.png]]

SVO ablation 显示，移除 SVO 后，不同性格 profile 之间的合作竞争模式区分变弱。这是整篇论文中比较关键的消融，因为它直接支撑 Social Value System 的必要性。

![[Figure 7 Malicious Competition Intervention.png]]

Figure 7 分析班长竞选中的恶性竞争。作者比较 No Intervention、Team Competition、Teacher Reminder、Pre-Education。结果显示，团队竞争和公平导向教育能降低方差和极端恶性竞争，控制组波动最大。这支持一个教育实践层面的结论：无约束竞选可能放大敌意竞争，而结构化合作任务和公平框架能稳定互动。

![[Figure 10 SVO Questionnaire Measurement.png]]

作者还让 LLM Surveyor 用 slider-based SVO questionnaire 对 agents 做外部测量，并将问卷得到的 SVO angle 与系统内部 SVO value 对比。结果显示二者较接近，用于验证 Social Value System。

> [!warning] 这里的消融仍然不够
> 移除 SVO 会削弱人格差异，这个结果合理。但还需要进一步比较更朴素的 profile-only prompt，例如只在 persona 中写明 competitive 或 prosocial，而不引入角度更新机制。否则很难判断收益来自 SVO 数学结构，还是来自更详细的人格描述。

### Experiment Summary

这篇论文的实验强项是覆盖了**系统级、多场景、两类个案、干预分支、人类评价、问卷一致性、消融实验**。对于一个平台型论文，这个实验组合比较完整。

但我注意到，它的评估主轴仍然是**仿真可信度**，不是**真实干预效果预测能力**。如果要把 EduMirror 推向更严肃的教育科学工具，下一步需要验证：仿真中有效的干预，在真实教育数据或专家评审中是否也更可信。

---

## Critical Thinking

> [!question] 这篇论文最核心的假设是什么
> 作者假设心理学理论约束下的 LLM 多智能体仿真，可以作为教育社会动态的 in silico 实验环境。这个假设有研究价值，但需要分清两件事：第一，系统能否生成符合理论和人类直觉的互动；第二，系统能否预测真实世界中干预后的因果变化。论文主要证明了第一点，对第二点保持谨慎。

> [!warning] 可能的评价偏差
> EduMirror 的生成、测量、评价大量依赖 LLM。即使 Rater 和 Surveyor 是后验的，也可能共享语言世界模型和社会常识模板。它们可能更偏好心理描写细腻、叙事连贯、道德框架清晰的输出，从而给 value-driven agent 更高分。

> [!warning] 反事实分支的因果边界
> 用户可以保存状态并施加干预，生成 parallel timelines。这个功能很适合 what-if 分析，但平行时间线里的差异仍然来自模型内部生成机制，而不是真实随机对照实验。它更适合作为 hypothesis generator，而不是直接作为 policy evidence。

> [!tip] 可以学习的写法
> 作者将平台贡献写成 workflow，而不是只列模块。Theory-grounded scenario design、value-driven agent、dual-track measurement、intervention branching 连成了一条研究流程，这种组织方式很适合教育 Agent 方向的论文写作。

---

## Future

作者自己提到的未来方向包括：把 individual values 和 social values 从并列配置进一步耦合；做更长时间尺度的 longitudinal simulation；加入更显式的记忆整合、情绪调节等认知过程；用仿真内部状态反过来验证问卷设计；测试不同 LLM、文化背景、年龄群体下的泛化性，并扩展到更大规模学校级网络。

我认为后续最值得挖的方向有三个。

第一是**从场景仿真走向任务定义**。EduMirror 可以进一步抽象成教育社会动态 benchmark：给定理论、初始状态、干预策略，预测心理状态曲线和行为分布。这样才能成为 Agent for Education 方向的可复现任务。

第二是**引入真实数据校准**。现在的心理变量变化主要由系统内部规则和 LLM judgment 支撑。后续可以用真实问卷、课堂观察、访谈编码或公开教育统计数据做 calibration，让仿真参数和真实分布对齐。

第三是**把反事实干预变成可学习策略**。当前干预由用户指定，例如教师提醒、团队竞争、预教育。更进一步的问题是：能否让一个策略 Agent 学习何时干预、干预谁、用什么语言干预，并用心理状态恢复和群体稳定性作为 reward。这个方向会自然连接到 educational agentic RL。
