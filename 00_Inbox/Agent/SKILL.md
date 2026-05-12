---
tags:
  - Agent
---

# Agent Skills 简介

> [!definition] Agent Skills
> Agent Skills 是一种用于扩展 Agent 能力的轻量级开放格式。它将任务说明、执行步骤、输出格式、示例、脚本和参考资料组织为可加载的文件目录。

Agent Skills 的提出背景来自 Agent 应用场景的扩展。Agent 逐渐参与代码开发、文档处理、数据分析、企业流程和长期协作任务。这些任务通常需要明确的背景、步骤、格式、检查项和项目约束。Skills 用文件形式保存可复用的任务说明，使这些经验可以被共享、维护和迭代。

> [!info] 提出背景
> Anthropic 于 2025 年 10 月发布 Agent Skills 相关工程文章，提出用文件和文件夹为 Agent 组织专用能力。随后，Agent Skills 被整理为开放标准。OpenAI 在 ChatGPT、Codex 和 API 文档中也采用 Skills 机制，将其用于可复用、可共享、可版本化的 workflow 文件包。

Skills 保存的是**程序性知识**。程序性知识关注任务的完成方式，包括操作步骤、执行顺序、质量标准和结果格式。例如，代码审查流程、论文阅读流程、表格处理规范、报告生成格式和知识图谱维护规则，都可以整理为 Skill。

> [!definition] 程序性知识
> 程序性知识是关于任务完成方法的知识，通常表现为步骤、规则、检查项和操作顺序。

从系统结构看，Skill 位于用户意图和工具调用之间。用户提出任务后，Agent 可以根据 Skill 中的说明决定如何分解任务、读取材料、调用工具、检查结果并组织输出。

> [!tip] 相关概念
> - Tool 提供可调用能力，例如搜索网页、运行代码、读写文件。
> - Memory 保存历史信息和用户偏好。
> - Prompt 提供当前对话中的指令。
> - Skill 保存某类任务的执行方法。
> - Workflow 描述更完整的任务流程，Skill 可以作为 workflow 的组成单元。

Skills 的意义主要体现在三个方面：
1. 减少重复提示词，将常用任务说明从单次对话中抽取出来。
2. 提高执行一致性，使 Agent 在相同任务中遵循稳定步骤和格式。
3. 支持共享和版本管理，使个人、团队或组织的经验能够以文件形式沉淀和更新。

> [!example] 典型应用
> - 代码审查 Skill 可以规定类型标注、异常处理、边界条件、测试覆盖率和提交格式的检查流程。
> - 论文阅读 Skill 可以规定研究问题、方法假设、实验设计、评价指标和局限性的提取顺序。
> - 知识图谱整理 Skill 可以规定概念节点拆分、关系建立、重复内容发现和缺失信息补全的处理流程。

简单来说，Skill 用于告诉 Agent 在某类任务中应当怎样执行。它把任务流程、领域经验和输出要求整理成可加载的文件，使 Agent 可以在需要时按照这些规则工作。

> [!warning] 使用边界
> Skill 不应写成冗长的背景资料库。核心文件应保持简洁，长篇参考材料应放入辅助文件。  
> Skill 不应替代工具定义。工具负责执行外部操作，Skill 负责规定任务流程。  
> Skill 不应包含不必要的全局规则。过宽的触发条件会导致 Agent 在无关任务中加载错误流程。

## 编写 Skill

得益于 LLM 的文本理解能力，设计 Skill 不需要复杂的编程门槛。设计者需要理解 Skill 的目录结构、元数据字段和正文组织方式，并将任务经验整理为 Agent 可以读取和执行的说明。

一个 Skill 目录一般由以下内容构成：

```text
my-skill/
├── SKILL.md          # Required: metadata + instructions
├── scripts/          # Optional: executable code
├── references/       # Optional: documentation
├── assets/           # Optional: templates, resources
└── ...               # Any additional files or directories
````

`SKILL.md` 是 Skill 的核心文件。该文件必须包含 YAML frontmatter 和 Markdown 正文。Frontmatter 用于描述 Skill 的元数据，正文用于描述 Skill 的具体使用方法。根据 Agent Skills 规范，frontmatter 必须包含 `name` 和 `description` 两个字段。

```markdown
---
name: skill-name
description: A description of what this skill does and when to use it.
---

# Instructions

Describe how the agent should complete the task.
```

- `name` 是 Skill 的名称。该字段用于标识 Skill，必须与父目录名称一致。*规范要求 `name` 长度为 1 到 64 个字符，只能包含小写字母、数字和连字符，不能以连字符开头或结尾，也不能包含连续连字符。*
- `description` 是 Skill 的功能和适用场景描述。*该字段用于帮助 Agent 判断当前任务是否需要使用该 Skill。规范要求 `description` 非空，长度不超过 1024 个字符，并建议同时说明 Skill 能做什么，以及何时使用。*

> [!note] description 的写法
> `description` 应包含具体任务关键词。较好的描述会同时覆盖任务对象、操作类型和触发场景。例如，PDF 处理 Skill 的描述可以包含 PDF、表格抽取、表单填写、文件合并等关键词。

Markdown 正文用于保存 Skill 的主要说明。规范不限制正文格式，但建议包含逐步说明、输入输出示例和常见边界情况。Agent 激活某个 Skill 后，会加载完整的 `SKILL.md`，因此正文应保持聚焦。较长的参考材料应拆分到辅助文件中。

> [!tip] 正文推荐内容
> - 任务适用范围
> - 执行步骤
> - 输入与输出要求
> - 输出格式
> - 示例
> - 常见边界情况
> - 需要读取的辅助文件路径

辅助目录用于承载正文之外的资源。最佳实践如下：
- `scripts/` 存放可执行代码，适合放置数据处理、格式转换、批量检查等脚本。
- `references/` 存放额外文档，适合放置长篇规范、术语表、API 说明和领域资料。
- `assets/` 存放静态资源，适合放置模板、图片、样例数据和配置文件。

Agent Skills 采用**渐进式披露**机制。Agent 启动时通常只加载所有 Skills 的 `name` 和 `description`。当某个 Skill 被激活后，再加载完整 `SKILL.md`。如果任务需要更详细的资料，Agent 再读取 `scripts/`、`references/` 或 `assets/` 中的相关文件。规范建议 `SKILL.md` 正文控制在 5000 tokens 以内，主文件控制在 500 行以内。

> [!note] 渐进式披露
> 渐进式披露指 Agent 先读取 Skill 的名称和描述，再按任务需要读取详细说明和附加资源。  
> 该机制用于降低上下文成本，并帮助 Agent 在大量 Skills 中选择当前任务需要的内容。

> [!warning] 文件引用
> 在 `SKILL.md` 中引用其他文件时，应使用相对路径。文件引用应从 Skill 根目录出发，例如 `references/REFERENCE.md` 或 `scripts/extract.py`。规范建议避免过深的嵌套引用链，减少 Agent 查找资料时的上下文和路径复杂度。

> [!tip] 可选字段
> 除 `name` 和 `description` 外，规范还定义了若干可选字段。这些字段用于补充版权、运行环境、扩展元数据和工具权限。多数简单 Skills 只需要 `name` 和 `description`。
>
> | 字段              | 是否必需 | 作用                                     |
> | --------------- | ---: | -------------------------------------- |
> | `license`       |    否 | 说明 Skill 的许可证，或指向随包提供的许可证文件            |
> | `compatibility` |    否 | 说明运行环境要求，例如目标产品、系统依赖、网络访问需求            |
> | `metadata`      |    否 | 保存规范之外的额外键值信息，例如作者、版本号                 |
> | `allowed-tools` |    否 | 声明预批准使用的工具，属于实验字段，不同 Agent 实现的支持程度可能不同 |

创建 Skill 时，通常先确定任务范围，再编写 `description`，随后设计正文流程，并按需要补充脚本、参考资料和模板。完成后，应使用典型任务测试 Skill 是否会被正确触发，以及 Agent 是否能按照正文流程完成任务。
