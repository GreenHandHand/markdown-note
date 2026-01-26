# CodePlan: Unlocking Reasoning Potential in Large Language Models by Scaling Code-form Planning

ICLR-2025

**核心一句话**：这篇论文想说明一件事：不是模型不会推理，而是**自然语言规划这条路本身就很吵**。作者把规划过程直接换成“写代码”，用可执行、可组合、可扩展的代码结构，把 LLM 的隐性推理能力逼出来。

---

## Key Contribution

- Introduce Code-form Planning (CodePlan) to replace natural language reasoning.
- Demonstrate that scaling code-based planning improves reasoning performance across tasks.
- Show strong gains without additional finetuning on multiple reasoning benchmarks.

- **核心贡献不是“又一种 prompting 技巧”，而是一次表征空间的切换**：从模糊、歧义多的自然语言规划，切换到高度结构化、可执行的代码规划。
- **所谓 scaling 不是指模型变大，而是规划空间变大**：代码天然支持函数抽象、循环、变量复用，这让中间推理可以被不断“堆高”，而不是在一段 CoT 里憋死。

---

## Method

整体逻辑非常清晰，可以概括为：  
- **Motivation：自然语言推理太吵**  
- **Intuition：代码天然就是为复杂规划而生的中间表示**  
- **Mechanism：让 LLM 学会先写代码形式的 plan，再执行/解释结果**

---

### Code-form Planning (CodePlan)

> [!question] 不合理的动机？
> 作者真的证明了“问题在表示，而不是模型能力”吗？还是只是找到了一个更容易 exploit 模型的 trick？

**Intuition（一句话直觉）**  
如果人类在做复杂问题时都会画流程图、写伪代码，那为什么要逼 LLM 全程用自然语言自言自语？

**它到底在干什么（人话版）**  
作者不再让模型输出一大段“首先…其次…所以…”，而是要求模型输出一段**结构化代码**，代码本身就包含了推理步骤。比如：

- 用变量保存中间结论
- 用函数封装子问题
- 用循环表达递归或枚举

这些在自然语言 CoT 里都非常别扭，但在代码里是“本职工作”。

![[98_Assets/CodePlan.png]]

结合图来看，这里最关键的不是 execution，而是 **planning stage 的外显化**：  
模型先生成 CodePlan，再基于 CodePlan 得到最终答案。

> [!tip] Aha moment  
> 这里非常妙的一点是：**代码是否真的被执行并不重要**。重要的是，代码强迫模型在生成阶段就把推理结构摆正。

---

### Why Code Scales Better than Natural Language

> [!note] 一个容易被忽略的点  
> 这篇论文真正的关键词其实是 scaling，而不是 code。

**Intuition**  
自然语言推理很难“越写越复杂”，因为上下文会爆、指代会乱、模型会开始胡言乱语。  
代码不一样，它天生就是为复杂系统设计的。

**具体体现在哪里**

- 自然语言 CoT：长度 ↑ → 噪声 ↑ → 推理稳定性 ↓  
- CodePlan：长度 ↑ → 结构 ↑ → 推理可控性 ↑

![[figure2.png]]

图里可以明显看到，随着 planning depth 增加，CodePlan 的性能是单调提升的，而 NL-CoT 很快撞墙。

> [!warning] 潜在猫腻  
> - 这里的 scaling 很大程度依赖模型本身的 code pretraining  
> - 如果是 code 能力较弱的模型，这个优势是否还成立？

---

### Execution vs Non-execution

> [!question] 原文没有细说  
> 到底是“代码结构”在起作用，还是“可执行反馈”在起作用？

作者做了一个很关键但容易被忽略的对比：  
**不执行代码，只把它当结构化文本，效果也很好。**

这说明：  
- 真正有价值的是 **planning representation**
- execution feedback 只是锦上添花，而不是必要条件

> [!tip] 值得学习的设计  
> 这一步实际上是在拆解“reasoning ≠ execution”，非常重要。

---

## Experiments

> [!note] 指标说明  
> 多数任务用的是 pass@k / accuracy，但真正该关注的是 **随 planning depth 的趋势变化**。

**定性总结**

- CodePlan 在 GSM8K、StrategyQA、BIG-Bench Hard 等任务上全面优于 NL-CoT
- 优势在中高难度问题上尤为明显
- 小模型收益更大，说明这是“能力释放”，不是“堆算力”

**Ablation 里最有信息量的点**

- 去掉代码结构，仅保留长文本，性能明显下降
- 限制代码复杂度，性能随之下降

> [!warning] 漏洞与保留意见  
> - baseline 主要是 NL-CoT，没有覆盖 tree-of-thought / graph-based reasoning  
> - prompt 工程痕迹仍然较重  
> - 是否对“非算法型推理”同样有效，还不够明确

---

## Future

作者提到的方向：

- 更复杂的 code abstraction
- 与 external tools / interpreters 结合

**我认为更值得挖的方向**

- CodePlan 作为 **中间表示接口**，用于 multi-agent reasoning
- 学习型 planner：让模型学会什么时候该写什么样的代码结构
- 把 CodePlan 当作 latent program，引入 verification / constraint checking

> [!tip] 个人判断  
> 这篇论文真正的价值不在于 code，而在于它重新定义了“推理的载体是什么”。
