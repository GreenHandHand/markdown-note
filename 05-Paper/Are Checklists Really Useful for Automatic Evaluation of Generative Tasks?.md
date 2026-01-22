---
tags:
  - todo
---


# Are Checklists Really Useful for Automatic Evaluation of Generative Tasks?

> [!info] EMNLP 2025

**核心一句话**：这篇论文质疑“用 checklist（检查清单）自动评估大模型生成结果”是否真的有效——作者发现，checklist 并非总是有用，甚至很多和人类评价低相关的 checklist 条目其实和人类专家写的条目高度重合，暴露出人类评价本身可能就不够客观。

---

## Key Contribution

- We find that selective checklist use sometimes improves evaluation outcomes, suggesting that omitting checklists can be justified in specific settings.
- We show that no universally optimal checklist generation method exists, as usefulness varies significantly depending on the evaluation model and use case.
- We find that even checklist items with low correlation to human evaluations often overlap with human-written ones, indicating they may still capture valid criteria. This highlights the subjective nature of human evaluations and calls for more objective evaluation design.

- **选择性使用 checklist 可能比全量使用效果更好**，尤其在 pairwise comparison（两两比较）任务中。这意味着不是所有问题都需要细粒度 checklist，有些简单任务直接让 LLM 判断反而更准。
- **不存在“最好用”的 checklist 生成方法**：baseline、specify、self-refine 等策略在不同模型/任务上表现不一，说明 checklist 的有效性高度依赖上下文。
- **最反直觉的发现**：那些“去掉后反而提升与人类评分相关性”的 checklist 条目（即“负向条目”），居然有大量和人类专家写的条目语义重合。这暗示问题可能不在 checklist 本身，而在人类评价标准模糊、主观性强。

---

## Method

### RQ1: Checklists 是否必要？——引入“选择性使用”策略

> [!question] 动机是否合理？
> 作者假设：当 LLM 自己多次评估结果不一致时，说明这个问题“难判”，才需要 checklist 辅助。这个直觉很自然——把 checklist 当成“疑难病例会诊工具”。

具体做法：
- 对每个样本，先让评估模型（如 GPT-4o）在**无 checklist**条件下重复评估 10 次。
- 定义“不一致性”指标：
  - Pairwise：少数派得票数 $x_{\text{pairwise}}$（比如 7:3，$x=3$）
  - Direct scoring：10 次打分的标准差 $x_{\text{direct}}$
- 设定阈值 $k$，仅当 $x \geq k$ 时才启用 checklist。

我注意到这里有个巧妙设计：**不是按问题类型判断是否需要 checklist，而是按模型自身行为动态决定**。这比预设“数学题需要 checklist、开放问答不需要”更灵活。

![[Figure 3.png]]
*图 3a 显示，在 pairwise 任务中，对 GPT-4o、Qwen2.5-32B 等模型，“选择性使用”（曲线）在某些 k 值下确实超过了 “All”（全用）和 “None”（不用）。但图 3b 的 direct scoring 任务中几乎没提升——说明 checklist 对“打分”帮助有限。*

> [!warning] 异议
> - 为什么 pairwise 能受益而 direct scoring 不行？作者没深挖机制。我猜是因为 pairwise 本质是排序，checklist 提供的二元判断（yes/no）更容易转化为胜/负；而 direct scoring 需要连续打分，checklist 的离散信号可能反而干扰模型校准。
> - 阈值 $k$ 是人工调的，缺乏理论依据。有没有可能用模型不确定性（如 entropy）替代投票分歧？

### RQ2: 如何生成有用的 checklist？

作者对比 6 种生成策略，核心差异在于**控制 checklist 的粒度与数量**：

| 方法 | 核心思想 | 我的理解 |
|------|--------|--------|
| Baseline | 生成 yes/no 问题，紧扣任务要求，避免模糊 | 最朴素的 prompt |
| Specify | 在 baseline 基础上，要求 checklist 考虑“可能的正确答案” | 让条目更具体，比如不只问“是否正确”，而问“是否答出 8849m” |
| Length×0.5 / ×1.5 | 调整 checklist 条目数量为 baseline 的 0.5 或 1.5 倍 | 测试“越多越好”是否成立 |
| Self-refine | 先生成 checklist → 用它评估 → 根据反馈迭代优化 checklist | 类似 self-critique，提升 checklist 质量 |
| Ticking | 复现 Cook et al. (2024) 的 prompt（但去掉了示例） | 作为 SOTA 方法的代表 |

> [!tip] Aha moment
> **“Specify” 方法之所以有效，是因为它把抽象标准（如“内容准确”）转化成了可验证的事实点（如“提到珠峰 8849m”）**。这本质上是在做 *evaluation grounding* ——把主观判断锚定到客观事实。

![[Figure 5.png]]
*图 5 展示了正/负向 checklist 条目的例子。注意 closed question（如翻译题）的负向条目往往是冗余的（“bonsoir 是否适合晚上语境？”——其实只要翻译对就行）；而 open question（如写邮件）的正向条目则聚焦关键要素（“是否礼貌？”、“是否明确请假日期？”）。*

> [!warning] 缺失实验
> 作者没测试“人类专家 vs 自动生成 checklist”的绝对质量差异。如果人类 checklist 本身就有噪声，那 overlap 高也不代表自动生成的好。

### RQ3: 哪些 checklist 条目真正对齐人类评价？

作者用 **ablation（消融）** 方法定义“正向/负向条目”：
- 正向条目：移除后，模型评分与人类的相关性 **下降**
- 负向条目：移除后，相关性 **上升**

直觉很简单：**看每个条目对最终 alignment 的边际贡献**。

公式上，定义：
$$
\Delta \bar{s}_{\text{abl}} = |\bar{s}_{\text{gold}} - \bar{s}_{\text{all}}| - |\bar{s}_{\text{gold}} - \bar{s}_{\text{abl}}|
$$
- 若 $\Delta \bar{s}_{\text{abl}} < 0$ → 移除后误差变大 → **正向条目**
- 若 $\Delta \bar{s}_{\text{abl}} > 0$ → 移除后误差变小 → **负向条目**

![[Figure 4.png]]
*图 4 显示，即使在“负向 checklist”中，也有约 40% 的条目是真正有害的（红色区域）。但注意纵轴 frequency 很高，说明大部分负向条目的危害其实很小（集中在 0 附近）。*

> [!question] 关键盲区
> 作者用 Qwen2.5-7B-it 做 ablation，但不同评估模型对同一 checklist 的敏感度可能不同。用更强的模型（如 GPT-4o）做 ablation，结果会变吗？

---

### Experiments

> [!note] 指标选择
> - Pairwise 用 **Accuracy**（多数投票胜率），tie 算 0.5 分——很务实，反映实际部署效果。
> - Direct scoring 用 **Krippendorff’s Alpha** 而非 Pearson/Spearman，因为它是衡量评分者间一致性的更鲁棒指标，尤其适合 Likert 量表。

- **定性总结**：checklist 的价值高度任务依赖。在 pairwise 任务中，**只要用 checklist 就比不用强**（Table 1 中 "None" 常是最差）；但在 direct scoring 中，选错生成策略（如 Baseline）反而拖后腿。
- **关键发现**：bootstrap 检验显示，**selective checklist 在 pairwise 中有统计显著提升（20/48 cases），但在 direct scoring 中完全无效**。

![[Table 4.png]]
*表 4 揭示惊人事实：InFoBench 中 274 个 checklist 条目里，超过一半（17+29=46）同时出现在人类和自动生成的清单中。说明 LLM 能抓住人类认为重要的点——哪怕这些点对提升相关性没用。*

> [!warning] 漏洞
> - 数据集局限：只用了英文数据（LLMBar + InFoBench），跨语言/跨文化场景未验证。
> - 模型覆盖虽广（7B–32B），但都是 instruction-tuned 模型，没测试 base 模型或非主流架构。
> - “人类评价”被当作 gold standard，但作者自己也承认其主观性（相关系数仅 0.35–0.52）——这动摇了整个评估根基。

---

### Future

**作者提及方向**：
- 重新设计更客观的人类评价协议
- 结合人类与自动生成 checklist 提升可靠性

**我认为值得深挖的方向**：
1. **Checklist 的“认知负荷”效应**：给评估模型太多 checklist 条目是否会分散注意力？能否用 attention 机制动态加权条目重要性？
2. **从 checklist 到 rubric**：当前 checklist 是二元 yes/no，未来可探索生成带权重的评分 rubric（如 FLASK 的 skill-set 思路）。
3. **人类评价的“去主观化”**：既然 checklist 条目和人类标准重合但相关性低，或许该用 checklist **反过来规范人类评价**——让人类按 checklist 打分，而非自由发挥。
