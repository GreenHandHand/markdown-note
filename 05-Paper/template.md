# template

## 结构抽象

### 1️⃣ 元信息区（Metadata）

```markdown
# {Paper Title}

{Conference / Journal}-{Year}

本文着重于：{一句话概括论文核心思想 / 方法定位}
```

**设计意图**

- 标题 = Paper 唯一标识
- 第二行 = venue + 年份（方便 Obsidian 搜索 / 时间排序）
- “本文着重于” = 你个人的理解入口（不是 abstract）

---

### 2️⃣ 核心概念注解（Concept Notes，可选）

```markdown
> [!note] {概念名}
> {用通俗但技术准确的语言解释该概念}
```

**特点**

- 不是每篇都必须有
- 专门用来解释：

  - inversion-free
  - training-free
  - latent optimization
  - self-attention leakage
- 这是你笔记**“教学价值”最高**的部分

---

### 3️⃣ Key Contribution（强制）

```markdown
# Key Contribution

- {英文原文贡献点 1}
- {英文原文贡献点 2}
- ...

- {对应的中文解释 1（扩写 + 技术消歧）}
- {对应的中文解释 2}
- ...
```

**这是你风格里非常关键的一点：**

- **不删英文原文**
- **中文不是翻译，而是“技术再解释”**
- 英文负责“对齐作者”，中文负责“对齐自己未来的理解”

---

### 4️⃣ Method（主体部分）

```markdown
# Method

![[{method_overview_figure}.png]]

![[{method_detail_figure}.png]]
```

#### 4.1 方法总览 / 子模块标题

```markdown
## {核心方法模块名}
```

先写**目标 / 设计动机**：

```markdown
目标是在 {约束条件} 下同时实现：

1. {目标 1}
2. {目标 2}
3. {目标 3}
```

---

#### 4.2 输入与整体流程

```markdown
输入预处理：

1. {步骤 1}
2. {步骤 2}
3. {步骤 3}
```

- 你习惯用**编号列表**
- 不急着讲公式，先讲“工程流程”

---

#### 4.3 核心思想拆解（必选）

每一个关键机制，都用 **三级标题 + 解释**：

```markdown
### {机制名}

{为什么要这样设计？}

{作者的直觉 / 核心假设}

{如果有公式，给公式 + 解释}
```

如：

- Pixel-manipulated latents as anchor
- Obtaining delta edit direction
- Feature-preserving source branch
- Leak-Proof Self-Attention
- Editing Guidance with Latents Optimization

---

#### 4.4 公式说明（局部）

```markdown
> [!note] {符号 / 操作名}
> $$
> {公式}
> $$
```

**原则**

- 公式只在“概念节点”出现
- 永远配一句**语义解释**

---

### 5️⃣ Experiment（结果解读）

```markdown
# Experiment

{总体评价一句话}

![[{experiment_figure}.png]]

{你对结果的定性判断（不是复述指标）}
```

---

### 6️⃣ Future Work / Open Questions（允许为空）

```markdown
# Future Work

{作者是否提及}

{你认为真正值得继续做的方向（可选）}
```

## 你的任务

```text
你是一个研究型阅读助手，擅长将非结构化的论文草稿笔记与原文摘录，
整理为高质量、结构清晰、适合 Obsidian 的论文阅读记录。

请遵循以下要求：

【输入】
- 我将提供：
  1. 非结构化的阅读草稿（可能包含原文、中文理解、零散想法）
  2. 可能混杂英文原文段落、公式、图片引用说明

【输出目标】
将其整理为一篇结构化的 Obsidian Paper Reading Note，结构和风格必须满足：

1. 使用 Markdown
2. 结构必须包含（按顺序）：
   - 标题 + 会议年份
   - “本文着重于：”的一句话总结（偏方法定位）
   - Key Contribution（保留英文 + 给出扩展中文解释）
   - Method（分模块，重视设计动机与流程）
   - Experiment（定性总结）
   - Future Work（如无则明确说明）

3. 写作风格要求：
   - 不要照抄 abstract
   - 中文部分是“技术解释”，不是直译
   - 允许适度扩写，但不得引入论文中不存在的方法
   - 公式必须保留，并配文字解释
   - 使用 Obsidian 的 callout（> [!note]）解释关键概念

4. Method 部分要求：
   - 明确整体目标
   - 清晰拆分子模块
   - 对关键设计（如 anchor、delta、branch、attention）解释“为什么”

5. 如果原文和草稿中存在重复或冲突表述：
   - 以“更利于理解方法机制”的版本为准
   - 不要保留明显冗余

【输入内容如下】
---
{在这里粘贴你的非结构化草稿 + 原文}
---

请直接输出最终整理好的 Obsidian 阅读记录，不要解释你的整理过程。
```
