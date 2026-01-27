# # Interactive and Expressive Code-Augmented Planning with Large Language Models

ACL-2024

**核心一句话**：这篇论文针对大语言模型在复杂长程规划任务中易出错、难处理模糊问题的痛点，提出了 REPL-Plan，一种基于 LLM-REPL（增强型交互式代码环境）的规划方法，通过“递归生成子任务 REPL+ 动态代码交互”实现自上而下的灵活规划，既保留了代码的结构化优势，又解决了纯代码方法的刚性局限。

---

## Key Contribution

- REPL-Plan: a novel approach for planning using LLM-REPLs, which are an extension of REPLs (e.g. language shells, code notebooks). LLM-REPLs enable a LLM to make decisions in a top-down way that is both dynamic and expressive.
- Strong performance across challenging environments (ALFWorld, WebShop, Real-World Web Tasks) with robust handling of long-horizon tasks and complex observations.
- Ablation studies verifying the necessity of recursive subtask spawning, code expressivity, and error correction capabilities.

- 首次将 REPL（交互式代码环境）的“边写边执行边修正”特性引入 LLM 规划，解决了纯代码增强方法“一次性编码”的刚性问题——我注意到这本质是把人类写代码的迭代模式赋予了 LLM，让规划过程从“静态脚本执行”变成“动态交互优化”。
- 提出 LLM-REPL 的递归生成机制，让模糊子问题（如主观决策、非结构化数据解读）和复杂主任务解耦，既保留代码的结构化优势（循环、变量传递），又发挥 LLM 处理模糊问题的能力——这比之前 THREAD 的“匿名子任务”更灵活，子 REPL 可复用且上下文传递更清晰。
- 在真实 web 导航任务（4k-20k tokens 观测）中验证了 scalability，证明代码 expressivity 是处理超长观测和复杂循环任务的关键——之前的方法（如 THREAD）在长观测下失效，而 REPL-Plan 通过代码切片处理观测，解决了 LLM 长上下文理解的短板。

---

## Method

### 核心动机：纯代码增强规划的三大痛点

> [!question] 方法动机的合理性
> 作者指出纯代码增强规划存在三个问题：模糊子问题难以用代码解决、自底向上编码需要精准预判、代码易出错。这三个痛点是否真实存在？
> 我的判断：确实是实际问题——比如“选最符合用户需求的商品”这种主观决策，代码无法用规则穷举；而一次性写对长流程代码对人类都难，更别说 LLM。

作者的核心直觉是：**把人类写代码的“交互式迭代 + 自上而下分解”模式迁移到 LLM 规划中**——人类用 REPL 写代码时，会分步执行、即时纠错、把复杂功能拆成函数；LLM 也应该具备这种能力，既用代码结构化规划，又通过交互动态调整。

### 核心模块：LLM-REPL

> [!note] REPL 基础概念
> REPL（Read-Eval-Print-Loop）是交互式代码环境（如 Jupyter），特点是“输入一行代码→执行→输出结果→循环”，支持即时纠错和分步开发。

LLM-REPL 是对传统 REPL 的扩展，核心新增**递归子 REPL 生成**和**上下文传递机制**，结构如下：

$$
\text{LLM-REPL} = \text{传统REPL功能} + \text{递归子REPL生成} + \text{上下文传递原语}
$$

#### 关键原语（Primitive Functions）

- `[subtask](args)`：生成子 LLM-REPL，用于处理子任务（如模糊问题、重复操作），子 REPL 可复用且独立维护变量状态。
- `get_args()`：子 REPL 从父 REPL 获取参数，解决上下文传递问题。
- `answer(a)`：子 REPL 向父 REPL 返回结果，执行权交回父 REPL。
- `act(a)`：与环境交互（如点击网页、移动物体），是规划任务的核心动作接口。
- `get_obs()`：获取环境观测（支持超长观测的字符串处理）。

#### 工作流程（结合图 1）

![[Figure 1.png]]
1. 主 REPL 接收任务（如“找两个苹果放在边桌上”），LLM 逐行编写代码，调用 `find_and_take_obj` 函数（未定义）。
2. 触发 `NameError`，主 REPL 生成子 REPL`find_and_take_obj`，通过 `get_args()` 获取目标物体和位置列表。
3. 子 REPL 执行“遍历位置→检查是否关闭→寻找苹果→拿起”的逻辑，通过 `act(a)` 与环境交互，实时接收观测反馈（如“柜子 2 是关着的”）。
4. 子 REPL 完成后通过 `answer('done')` 返回结果，主 REPL 继续执行“放置苹果”的逻辑，重复上述过程直到任务完成。

> [!warning] 设计异议
> - 作者为什么用 `NameError` 触发子 REPL 生成？这是否会与真正的变量未定义错误混淆？原文附录 A.1 提到通过自定义 `REPLNameError` 区分，但这种“错误驱动”的设计是否比显式声明子任务更高效？
> - 子 REPL 与父 REPL 不共享变量状态，必须通过 `get_args()` 和 `answer()` 传递——这增加了代码编写负担，为什么不设计可选的共享状态机制？

### 核心方法：REPL-Plan

REPL-Plan 本质是“LLM-REPL 的集合”，通过以下特性实现高效规划：
1. **代码表达性（Code-Expressive）**：支持循环、函数调用、变量存储等完整代码功能，可处理重复任务（如遍历 10 个商品）和复杂逻辑。
2. **动态性（Dynamic）**：每步代码执行后即时获取反馈，LLM 可修正错误（如代码 bug、观测误解），无需重写整个规划。
3. **自上而下分解（Top-Down）**：通过递归子 REPL 将复杂任务拆分为小任务，如“网页商品筛选”拆分为 `filter_search`→`filter_page`→`item_matches`（图 2）。

![[Figure 2.png]]
图 2 展示了网页商品筛选的任务分解：主 REPL 调用 `filter_search` 子 REPL，后者又调用 `filter_page` 和 `item_matches` 子 REPL，通过 `get_args()` 传递商品描述，`answer()` 返回匹配结果。这种分解让长流程任务的代码更简洁，且每个子 REPL 可独立优化。

> [!tip] 高光设计
> 子 REPL 的复用机制——全局 REPL 池存储之前任务的子 REPL（如 `check_requirements`），新任务可直接调用，实现少样本学习。这比之前的方法（如 THREAD）的“一次性子任务”更高效，尤其适合同类场景的规划。

---

## Experiments

### 实验设置

- **环境**：ALFWorld（文本家居环境）、WebShop（电商导航）、Real-World Web（真实电商网站导航，4k-20k tokens 观测）。
- **基线**：Reflexion、ReAct、THREAD、RAP 等主流 LLM 规划方法。
- **关键指标**：成功率（SR）、匹配分数（Score，WebShop 中衡量商品属性匹配度）、专家分数占比（Real-World Web）。

> [!note] 指标解释
> Real-World Web 使用“专家分数占比”而非绝对成功率，因为真实网页结构多变，专家完成度是更合理的参照——这比单纯的成功率更能反映方法的鲁棒性。

### 核心结果分析

1. **ALFWorld**：REPL-Plan 达到 97.0% SR，超过最佳基线 THREAD（95.5%）。
   - 关键原因：自上而下分解和动态纠错能力——两者都能分解任务，但 REPL-Plan 的代码表达性让变量跟踪更准确（如记录已找到的苹果位置），减少 hallucination。
2. **WebShop**：
   - k=3（Top-3 商品）：THREAD 略优（49% SR），REPL-Plan 47% SR——简单任务中，文本分解（THREAD）与代码分解（REPL-Plan）差异不大。
   - k=10（Top-10 商品）：REPL-Plan 的 Top-20 策略达到 52% SR，而 THREAD 仅 21%——代码的循环和变量传递能力是处理大量商品遍历的关键。
3. **Real-World Web**：
   - 简单任务：REPL-Plan 与 ReAct 均为 86.7%，THREAD 仅 13.3%——THREAD 无法处理超长观测。
   - 复杂任务：REPL-Plan 39.6%，ReAct 17.6%，THREAD 0%——代码表达性让循环遍历和模糊属性判断（如“12 页/分钟打印速度”）更高效。

### 消融实验关键发现（Table 4）

- **Buggy Demonstrations**：注入代码 bug 后，性能仅轻微下降（GPT-4o-mini 从 44%→40%）——证明动态纠错能力有效。
- **No-Subtask-REPLs**：禁用递归子 REPL 后，性能腰斩（GPT-3.5 从 52%→24%）——自上而下分解是核心。
- **Zero-shot Subtask-REPL**：移除关键子 REPL 后，性能大幅下降（GPT-3.5 从 52%→28%）——少样本子 REPL 复用对性能至关重要。

> [!warning] 实验漏洞
> - WebShop 的 k=10 设置中，THREAD 的 Top-20 策略未成功实现（作者提到 THREAD 会无限循环），是否是实现问题而非方法本身的局限？
> - Real-World Web 仅测试了 5 个电商网站，数据集规模较小，是否存在过拟合？
> - 未与最新的代码增强方法（如 CodeAct）对比，这些方法是否也具备类似的交互能力？

---

## Future

### 作者提及的方向

1. 扩展到分布外场景（Out-of-Distribution），提升 LLM 对未见过的子任务的零样本代码生成能力。
2. 优化效率：自动判断纯代码可解决的子任务，无需 LLM 干预，减少交互开销。
3. 微调 LLM 以更好地适应 REPL-Plan 范式，而非仅依赖上下文学习。

### 专家潜在方向

1. **多模态扩展**：将 LLM-REPL 应用于视觉 - 语言规划（如机器人导航），用代码处理结构化的视觉观测（如物体坐标），LLM 处理模糊决策（如“选择适合的工具”）。
2. **强化学习结合**：用强化学习优化子 REPL 的代码生成，而非依赖专家演示，提升在无标注场景的性能。
3. **错误类型自适应**：当前对所有 `NameError` 都生成子 REPL，可设计自适应机制——区分“子任务声明”和“真正的代码错误”，提升效率。
4. **跨领域 REPL 复用**：构建通用子 REPL 库（如“文本筛选”“网页点击”），支持跨任务快速迁移，减少少样本演示需求。Interactive and Expressive Code-Augmented Planning with Large Language Models

arXiv-2024

**核心一句话**：这篇论文针对大语言模型在复杂长程规划任务中易出错、难处理模糊问题的痛点，提出了 REPL-Plan——一种基于 LLM-REPL（增强型交互式代码环境）的规划方法，通过“递归生成子任务 REPL+ 动态代码交互”实现自上而下的灵活规划，既保留了代码的结构化优势，又解决了纯代码方法的刚性局限。

---

## Key Contribution

- REPL-Plan: a novel approach for planning using LLM-REPLs, which are an extension of REPLs (e.g. language shells, code notebooks). LLM-REPLs enable a LLM to make decisions in a top-down way that is both dynamic and expressive.
- Strong performance across challenging environments (ALFWorld, WebShop, Real-World Web Tasks) with robust handling of long-horizon tasks and complex observations.
- Ablation studies verifying the necessity of recursive subtask spawning, code expressivity, and error correction capabilities.

// 扩写 + 技术消歧，对齐你的理解
- 首次将 REPL（交互式代码环境）的“边写边执行边修正”特性引入 LLM 规划，解决了纯代码增强方法“一次性编码”的刚性问题——我注意到这本质是把人类写代码的迭代模式赋予了 LLM，让规划过程从“静态脚本执行”变成“动态交互优化”。
- 提出 LLM-REPL 的递归生成机制，让模糊子问题（如主观决策、非结构化数据解读）和复杂主任务解耦，既保留代码的结构化优势（循环、变量传递），又发挥 LLM 处理模糊问题的能力——这比之前 THREAD 的“匿名子任务”更灵活，子 REPL 可复用且上下文传递更清晰。
- 在真实 web 导航任务（4k-20k tokens 观测）中验证了 scalability，证明代码 expressivity 是处理超长观测和复杂循环任务的关键——之前的方法（如 THREAD）在长观测下失效，而 REPL-Plan 通过代码切片处理观测，解决了 LLM 长上下文理解的短板。

---

## Method

### 核心动机：纯代码增强规划的三大痛点

> [!question] 方法动机的合理性
> 作者指出纯代码增强规划存在三个问题：模糊子问题难以用代码解决、自底向上编码需要精准预判、代码易出错。这三个痛点是否真实存在？
> 我的判断：确实是实际问题——比如“选最符合用户需求的商品”这种主观决策，代码无法用规则穷举；而一次性写对长流程代码对人类都难，更别说 LLM。

作者的核心直觉是：**把人类写代码的“交互式迭代 + 自上而下分解”模式迁移到 LLM 规划中**——人类用 REPL 写代码时，会分步执行、即时纠错、把复杂功能拆成函数；LLM 也应该具备这种能力，既用代码结构化规划，又通过交互动态调整。

### 核心模块：LLM-REPL

> [!note] REPL 基础概念
> REPL（Read-Eval-Print-Loop）是交互式代码环境（如 Jupyter），特点是“输入一行代码→执行→输出结果→循环”，支持即时纠错和分步开发。

LLM-REPL 是对传统 REPL 的扩展，核心新增**递归子 REPL 生成**和**上下文传递机制**，结构如下：

$$
\text{LLM-REPL} = \text{传统REPL功能} + \text{递归子REPL生成} + \text{上下文传递原语}
$$

#### 关键原语（Primitive Functions）

- `[subtask](args)`：生成子 LLM-REPL，用于处理子任务（如模糊问题、重复操作），子 REPL 可复用且独立维护变量状态。
- `get_args()`：子 REPL 从父 REPL 获取参数，解决上下文传递问题。
- `answer(a)`：子 REPL 向父 REPL 返回结果，执行权交回父 REPL。
- `act(a)`：与环境交互（如点击网页、移动物体），是规划任务的核心动作接口。
- `get_obs()`：获取环境观测（支持超长观测的字符串处理）。

#### 工作流程（结合图 1）

![[Figure 1.png]]
1. 主 REPL 接收任务（如“找两个苹果放在边桌上”），LLM 逐行编写代码，调用 `find_and_take_obj` 函数（未定义）。
2. 触发 `NameError`，主 REPL 生成子 REPL`find_and_take_obj`，通过 `get_args()` 获取目标物体和位置列表。
3. 子 REPL 执行“遍历位置→检查是否关闭→寻找苹果→拿起”的逻辑，通过 `act(a)` 与环境交互，实时接收观测反馈（如“柜子 2 是关着的”）。
4. 子 REPL 完成后通过 `answer('done')` 返回结果，主 REPL 继续执行“放置苹果”的逻辑，重复上述过程直到任务完成。

> [!warning] 设计异议
> - 作者为什么用 `NameError` 触发子 REPL 生成？这是否会与真正的变量未定义错误混淆？原文附录 A.1 提到通过自定义 `REPLNameError` 区分，但这种“错误驱动”的设计是否比显式声明子任务更高效？
> - 子 REPL 与父 REPL 不共享变量状态，必须通过 `get_args()` 和 `answer()` 传递——这增加了代码编写负担，为什么不设计可选的共享状态机制？

### 核心方法：REPL-Plan

REPL-Plan 本质是“LLM-REPL 的集合”，通过以下特性实现高效规划：
1. **代码表达性（Code-Expressive）**：支持循环、函数调用、变量存储等完整代码功能，可处理重复任务（如遍历 10 个商品）和复杂逻辑。
2. **动态性（Dynamic）**：每步代码执行后即时获取反馈，LLM 可修正错误（如代码 bug、观测误解），无需重写整个规划。
3. **自上而下分解（Top-Down）**：通过递归子 REPL 将复杂任务拆分为小任务，如“网页商品筛选”拆分为 `filter_search`→`filter_page`→`item_matches`（图 2）。

![[Figure 2.png]]
图 2 展示了网页商品筛选的任务分解：主 REPL 调用 `filter_search` 子 REPL，后者又调用 `filter_page` 和 `item_matches` 子 REPL，通过 `get_args()` 传递商品描述，`answer()` 返回匹配结果。这种分解让长流程任务的代码更简洁，且每个子 REPL 可独立优化。

> [!tip] 高光设计
> 子 REPL 的复用机制——全局 REPL 池存储之前任务的子 REPL（如 `check_requirements`），新任务可直接调用，实现少样本学习。这比之前的方法（如 THREAD）的“一次性子任务”更高效，尤其适合同类场景的规划。

---

## Experiments

### 实验设置

- **环境**：ALFWorld（文本家居环境）、WebShop（电商导航）、Real-World Web（真实电商网站导航，4k-20k tokens 观测）。
- **基线**：Reflexion、ReAct、THREAD、RAP 等主流 LLM 规划方法。
- **关键指标**：成功率（SR）、匹配分数（Score，WebShop 中衡量商品属性匹配度）、专家分数占比（Real-World Web）。

> [!note] 指标解释
> Real-World Web 使用“专家分数占比”而非绝对成功率，因为真实网页结构多变，专家完成度是更合理的参照——这比单纯的成功率更能反映方法的鲁棒性。

### 核心结果分析

1. **ALFWorld**：REPL-Plan 达到 97.0% SR，超过最佳基线 THREAD（95.5%）。
   - 关键原因：自上而下分解和动态纠错能力——两者都能分解任务，但 REPL-Plan 的代码表达性让变量跟踪更准确（如记录已找到的苹果位置），减少 hallucination。
2. **WebShop**：
   - k=3（Top-3 商品）：THREAD 略优（49% SR），REPL-Plan 47% SR——简单任务中，文本分解（THREAD）与代码分解（REPL-Plan）差异不大。
   - k=10（Top-10 商品）：REPL-Plan 的 Top-20 策略达到 52% SR，而 THREAD 仅 21%——代码的循环和变量传递能力是处理大量商品遍历的关键。
3. **Real-World Web**：
   - 简单任务：REPL-Plan 与 ReAct 均为 86.7%，THREAD 仅 13.3%——THREAD 无法处理超长观测。
   - 复杂任务：REPL-Plan 39.6%，ReAct 17.6%，THREAD 0%——代码表达性让循环遍历和模糊属性判断（如“12 页/分钟打印速度”）更高效。

### 消融实验关键发现（Table 4）

- **Buggy Demonstrations**：注入代码 bug 后，性能仅轻微下降（GPT-4o-mini 从 44%→40%）——证明动态纠错能力有效。
- **No-Subtask-REPLs**：禁用递归子 REPL 后，性能腰斩（GPT-3.5 从 52%→24%）——自上而下分解是核心。
- **Zero-shot Subtask-REPL**：移除关键子 REPL 后，性能大幅下降（GPT-3.5 从 52%→28%）——少样本子 REPL 复用对性能至关重要。

> [!warning] 实验漏洞
> - WebShop 的 k=10 设置中，THREAD 的 Top-20 策略未成功实现（作者提到 THREAD 会无限循环），是否是实现问题而非方法本身的局限？
> - Real-World Web 仅测试了 5 个电商网站，数据集规模较小，是否存在过拟合？
> - 未与最新的代码增强方法（如 CodeAct）对比，这些方法是否也具备类似的交互能力？

---

## Future

### 作者提及的方向

1. 扩展到分布外场景（Out-of-Distribution），提升 LLM 对未见过的子任务的零样本代码生成能力。
2. 优化效率：自动判断纯代码可解决的子任务，无需 LLM 干预，减少交互开销。
3. 微调 LLM 以更好地适应 REPL-Plan 范式，而非仅依赖上下文学习。

### 专家潜在方向

1. **多模态扩展**：将 LLM-REPL 应用于视觉 - 语言规划（如机器人导航），用代码处理结构化的视觉观测（如物体坐标），LLM 处理模糊决策（如“选择适合的工具”）。
2. **强化学习结合**：用强化学习优化子 REPL 的代码生成，而非依赖专家演示，提升在无标注场景的性能。
3. **错误类型自适应**：当前对所有 `NameError` 都生成子 REPL，可设计自适应机制——区分“子任务声明”和“真正的代码错误”，提升效率。
4. **跨领域 REPL 复用**：构建通用子 REPL 库（如“文本筛选”“网页点击”），支持跨任务快速迁移，减少少样本演示需求。
