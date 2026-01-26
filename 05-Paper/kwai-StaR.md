# Kwai-STaR: Transform LLMs into State-Transition Reasoners

arXiv-2024

**核心一句话**：这篇论文针对 LLM 在数学推理上的弱点，提出把问题解决过程看成从初始状态到最终状态的转移链条，通过小规模状态转移数据和两阶段训练，让普通 LLM 变成高效的状态转移推理器，显著提升直觉推理能力而非靠复杂推理时技巧。

---

## Key Contribution

- We provide the novel perspective of state transition to model the mathematical reasoning of LLMs and construct a state-transition dataset.
- We propose Kwai-STaR framework to enhance LLM reasoning through state transitions. Kwai-STaR effectively transforms models of various scales into STaRs, significantly improving their mathematical performance.
- Kwai-STaR achieves remarkable performance and efficiency, revealing the great potential of state-space strategies in enhancing LLM reasoning. We are actively extending our state-space strategies to broader scenarios.

- 作者引入状态转移视角，本质上是把数学推理拆成一个个小步转移，构建了一个只有 20K 规模的数据集，这比 MetaMath 那种海量改写 QA 对更高效，我注意到这能让模型学到更结构化的思考路径，而不是简单记住答案。
- Kwai-STaR 框架的核心是三步走：定义状态空间、生成转移数据、用课程式训练转化模型，这不只提升性能，还让小模型如 Mistral-7B 在 GSM8K 上从 17% 跳到 80%，证明状态转移能放大模型的内在推理潜力。
- 作者强调性能和效率双赢，比如单次推理精度媲美多采样方法，我觉得这揭示了状态空间在 LLM 推理中的潜力，但也让我怀疑是否只限于数学领域。

---

## Method

### State Space Definition

> [!question] 不合理的动机
> 作者引入状态空间是为了解决 LLM 在复杂数学问题上容易卡壳的问题，直觉上合理，因为数学解题本来就是步步推进，但为什么不直接用树搜索而要自定义 7 个动作？这是否强行复杂化了？

作者的直觉是把数学问题看成从初始问题状态到最终答案状态的转移过程，LLM 每次选一个动作推进状态，避免一次性生成长链推理出错。机制上，他们定义了状态（解题过程中的点）、动作集（7 种操作如 Formalize、Decompose 等）和转移规则，让 LLM 像 RL 代理一样逐步探索。

必须结合公式解释：公式没直接给出，但动作集如 Table 1 所示，转移是序列化的，比如从初始状态 $S_0$ 通过动作 $A1$ 到 $S1$，直到终态。各项物理意义：状态代表当前解题进度，动作是具体操作工具，确保分治原则（divide-and-conquer）。
![[98_Assets/kwai-StaR.png]]

> [!tip]
> 这里的设计精妙在于 Verify 和 Backtrack 动作，能让模型自我纠错，这比纯 CoT 更鲁棒。
> ![[98_Assets/kwai-StaR-1.png]]
>
> *注意看图中这个指令示例，它展示了如何用动作序列指导数据生成，对应了状态转移的实际流程。*

> [!warning] 异议
> - 作者没解释为什么选这 7 个动作而不是更多或更少，比如为什么不加合并子问题的动作？
> - 反直觉的是，通常 LLM 推理是自由文本，这里强制结构化转移，是否会限制创造性？原文没消融实验证明每个动作的必要性。
> - 状态空间是否完整？复杂问题可能有循环转移，但作者没提。

### State-Transition Data Construction

> [!question] 不合理的动机
> 作者建数据集是为了教弱模型掌握状态转移，因为先进 LLM 虽能跟指令，但小模型不行，这合理，但为什么只用 GSM8K 训练集生成 20K 数据，而不扩展？这是否太窄？

作者直觉是高质量小规模数据胜过海量低质，通过两阶段生成：第一阶段用简化动作集生成正确案例，第二阶段加 Verify/Backtrack 纠错生成拒收对。机制是提示强大 LLM（如 GPT-4o）模拟学生/老师角色，输出状态转移路径，形成 20K 正确 +3K 纠错实例。

![[98_Assets/kwai-StaR-2.png]]

*图中上方部分展示了数据构建流程，从正确到错后纠错，对应了生成拒收对的过程。*

> [!warning] 异议
> - 作者用 LLaMA-3.1-70B 和 GPT-4o 生成数据，但没提如何避免生成器偏置传给下游模型？
> - 反直觉：数据集小但有效，是否因为结构化格式？但原文没对比纯文本数据。
> - 缺失消融：没实验证明错例比例对训练的影响。

### Curricular Training Strategy

> [!question] 不合理的动机
> 作者用两阶段训练是为了渐进学习，先基础后高级，这像课程学习合理，但为什么 DPO 只用拒收对，而不混 SFT？这是否忽略了正样本强化？

作者直觉是先用简单正确案例教模型转移模式，再用难例拒收对教纠错，提升鲁棒。机制分基础 SFT 和高级 DPO：SFT 用 NTP 损失训练正确路径，DPO 用拒收对优化偏好。

必须结合公式解释：NTP 损失
$$
\mathcal{L}_{\text{NTP}} = - \sum_{t=1}^{T} \log P(y_{t} | y_{<t}; \theta)
$$
其中 $y_t$ 是下一个 token，教模型生成正确序列；DPO 损失
$$\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = - \mathbb{E}_{(x,y_a,y_r) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \left( \frac{\pi_{\theta}(y_a | x)}{\pi_{\text{ref}}(y_a | x)} \right) - \beta \log \left( \frac{\pi_{\theta}(y_r | x)}{\pi_{\text{ref}}(y_r | x)} \right) \right) \right]
$$
$y_a$ 是接受路径，$y_r$ 是拒绝，ref 是 SFT 模型，确保偏向正确转移。

![[98_Assets/kwai-StaR-2.png]]

*图中下方训练流程，对应了从 SFT 到 DPO 的渐进，注意羊驼图标代表模型演化。*

> [!warning] 异议
> - 作者没解释β超参选择，为什么默认值？
> - 反直觉：DPO 用基础模型作为 ref，是否会放大早期错误？原文没迭代 DPO 的实验。
> - 缺失消融：没证明两阶段比单阶段好。

---

### Experiments

> [!note] 重要指标解释
> 用 maj@1 和 maj@8 表示多数投票精度，maj@1 是单次推理，maj@8 是 8 采样投票；GSM-Hard 是更难变体，测试泛化。

- **定性总结**：Kwai-STaR 在小模型上提升巨大，如 Mistral-7B GSM8K 从 17% 到 80%，证明状态转移高效，但大模型如 LLaMA3.1-8B 提升较小，暗示天花板效应。
- **关键分析**：Ablation 隐含在比较中，状态转移数据比 MetaMathQA 高效（小规模胜大），单次精度媲美多采样，来源是结构化学习纠错能力。

![[98_Assets/kwai-StaR-3.png]]

> [!warning] 漏洞
> - 没比更强 Baseline 如 o1 模型，或完整 MCTS。
> - 数据集挑 GSM8K/GSM-Hard，数学窄域，没跨域测试。
> - 结果没可视化，可能是 cherry-picked，没报告方差。

---

### Future

作者提到扩展状态空间到更广场景，如一般推理；我作为专家认为值得挖掘潜在方向：
1. 自动化状态空间设计，用 LLM 生成动作集而非手动；
2. 结合 RLHF 进一步优化转移策略；
3. 测试在代码生成或规划任务上的泛化，探索多模态状态。
