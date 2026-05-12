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

> [!tip] 可选字段与实现扩展
> 除 `name` 和 `description` 外，Agent Skills 规范、Claude Code、Codex 和 OpenCode 还提供了一些可选字段和实现扩展。通用 Skill 应优先依赖 `name`、`description`、正文说明和辅助文件；平台相关字段需要按具体运行环境判断是否可用。
>
> | 字段或能力 | 来源 | 作用 |
> |---|---|---|
> | `license` | Agent Skills 规范 / OpenCode | 说明 Skill 的许可证，或指向随包提供的许可证文件。 |
> | `compatibility` | Agent Skills 规范 / OpenCode | 说明运行环境要求，例如目标产品、系统依赖、网络访问需求。 |
> | `metadata` | Agent Skills 规范 / OpenCode | 保存规范之外的额外键值信息，例如作者、版本号、适用团队、工作流类型。OpenCode 要求其为 string-to-string map。 |
> | `allowed-tools` | Agent Skills 规范 / Claude Code | 声明 Skill 激活时可免确认使用的工具。Claude Code CLI 支持该字段，但它用于预批准工具，不会限制其他工具的可见性；若要禁止某些工具，需要使用权限配置。 |
> | `when_to_use` | Claude Code 扩展 | 补充说明 Skill 的触发条件，例如典型请求、触发短语和适用场景。Claude 会把它追加到 `description` 中用于技能列表展示。 |
> | `argument-hint` | Claude Code 扩展 | 在自动补全中提示该 Skill 期望的参数形式，例如 `[issue-number]` 或 `[filename] [format]`。 |
> | `arguments` | Claude Code 扩展 | 定义命名位置参数。Skill 正文中可以使用 `$ARGUMENTS`、`$ARGUMENTS[N]` 或 `$N` 引用用户传入的参数。 |
> | `disable-model-invocation` | Claude Code 扩展 | 禁止 Claude 自动触发该 Skill，使其只能通过 `/skill-name` 手动调用。适合需要明确用户确认的流程。 |
> | `user-invocable` | Claude Code 扩展 | 控制 Skill 是否出现在 `/` 菜单中。设置为 `false` 时，该 Skill 更适合作为后台知识或被其他机制调用。 |
> | `context: fork` | Claude Code 扩展 | 让 Skill 在隔离的子上下文中运行。Skill 正文会作为子 Agent 的任务输入，不继承主对话历史。 |
> | `agent` | Claude Code 扩展 | 与 `context: fork` 配合使用，用于指定执行该 Skill 的子 Agent 类型，例如 `Explore` 或 `Plan`。 |
> | 动态上下文注入 | Claude Code 扩展 | 在 Skill 正文中使用命令占位语法，将命令输出提前插入到提示内容中。常用于读取 git diff、PR 信息、环境状态等实时上下文。 |
> | `agents/` | Codex 扩展 | Codex Skill 目录中可以包含 `agents/` 子目录。例如 `agents/openai.yaml` 可用于描述外观和依赖等 Codex 相关配置。 |
> | 显式调用 | Claude Code / Codex / OpenCode | 用户可以直接指定使用某个 Skill。Claude Code 使用 `/skill-name`，Codex CLI/IDE 支持 `/skills` 或通过 `$` 提及 Skill，OpenCode 通过原生 `skill` 工具加载。 |
> | 隐式调用 | Claude Code / Codex / OpenCode | Agent 根据 `description` 判断当前任务是否匹配某个 Skill。描述越清晰，误触发和漏触发概率越低。 |
> | 安装范围 | Claude Code / Codex / OpenCode | Skill 可以放在用户级、项目级、插件级或组织级目录中。不同平台的扫描路径和优先级不同。 |
> | 权限配置 | Claude Code / OpenCode | Claude Code 支持 `allowed-tools` 进行 Skill 级工具预批准。OpenCode 的 `SKILL.md` 只识别少数字段，权限主要通过 `opencode.json` 配置，权限项包括 `read`、`edit`、`bash`、`task`、`skill`、`webfetch`、`websearch` 等。 |
> | 技能选择预算 | Codex | Codex 初始上下文只放入 Skill 的名称、描述和路径，并对技能列表占用的上下文设置预算。Skill 被选中后才读取完整 `SKILL.md`。 |
> | 版本管理 | OpenAI API | OpenAI API 将 Skill 作为带版本的文件包管理，支持创建新版本、设置默认版本、引用 curated skills 和 inline skills。 |
>
> 这些扩展能力分属不同实现，不应全部视为 Agent Skills 开放规范的必需部分。编写跨平台 Skill 时，应把核心流程写在 `SKILL.md` 正文中，把平台相关能力作为可选增强。

创建 Skill 时，通常先确定任务范围，再编写 `description`，随后设计正文流程，并按需要补充脚本、参考资料和模板。完成后，应使用典型任务测试 Skill 是否会被正确触发，以及 Agent 是否能按照正文流程完成任务。

## 读取流程

本节描述 Skill 从可用文件到实际参与任务的读取过程。

Skill 的读取流程采用**渐进式披露**机制。
- Harness 不会在初始阶段把所有 Skill 内容都放入上下文，而是先注入轻量摘要。Agent 看到摘要后，根据用户任务判断是否需要某个 Skill。
- 只有当 Skill 与当前任务相关时，Agent 才会读取完整 `SKILL.md`，并在执行过程中按需读取辅助文件。

完整流程可以概括为：
1. Harness 扫描可用 Skill 目录，并解析其中的 `SKILL.md`。
2. Harness 将 Skill 的 `name`、`description` 等摘要信息注入 Agent 上下文。
3. Agent 根据用户任务和 Skill 摘要判断是否需要使用某个 Skill。
4. Agent 选择 Skill 后，Harness 提供完整 `SKILL.md` 内容。
5. Agent 按照 `SKILL.md` 执行任务，并在需要时请求读取辅助资源。
6. Harness 根据权限和路径规则提供脚本、参考资料、模板等文件。
7. Agent 结合 Skill 说明、辅助资源和可用工具生成结果。

```mermaid
flowchart LR
    A[Harness 扫描 Skill 目录] --> B[解析 SKILL.md 的元数据]
    B --> C[注入 name 与 description]
    C --> D[Agent 判断任务是否匹配 Skill]
    D --> E[读取完整 SKILL.md]
    E --> F[按需读取辅助资源]
    F --> G[执行任务并生成结果]
````

在 Skill 开放标准中，初始上下文通常只需要包含 `name` 和 `description`。这两个字段用于让 Agent 识别当前有哪些 Skills，以及每个 Skill 大致适用于什么任务。Codex 的实现还会提供 `path`，使 Agent 可以在选中 Skill 后定位完整文件。

> [!tip]
> - `name` 用于区分 Skill。
> - `description` 用于判断适用场景。
> - `path` 用于定位完整 Skill 文件。
>
> `description` 越清晰，Agent 越容易正确选择 Skill。

Skill 被选中后，Agent 才会读取完整 `SKILL.md`。此时进入上下文的是具体执行说明，包括任务步骤、输出格式、注意事项和辅助文件路径。辅助文件不会默认全部读取，只有在任务需要时才会被加载。

> [!warning] 常见问题
> - `description` 写得过宽，容易导致 Skill 被误用。
> - `description` 写得过窄，容易导致 Skill 无法被选中。
> - `SKILL.md` 写得过长，会增加上下文负担。
> - 辅助文件路径不清晰，会增加 Agent 查找资料的难度。

> [!note] 具体实现
> Claude Code 和 Codex 都采用这种分层读取思路，但实现细节不同。
> - Claude Code 支持自动触发和显式调用。显式调用通常通过 `/skill-name` 完成。Claude Code 还支持工具预批准、参数传入、动态上下文注入和子 Agent 执行等扩展能力。
> - Codex 会将 Skill 的 `name`、`description` 和 `path` 放入初始上下文。当 Skills 数量较多时，Codex 会控制初始 Skill 列表的上下文预算。Codex 支持显式调用和隐式匹配，并在 Skill 被选中后读取完整 `SKILL.md`。

> [!note] 上下文注入方式
> 从开源实现看，Harness 通过修改 system prompt、上下文消息、工具列表、文件读取结果和工具返回结果，使模型在合适阶段看到 Skill 的摘要、正文和辅助资源。
>
> - **在系统提示中注入 Skill 摘要**  
> Harness 可以在 system prompt 或类似的全局上下文中加入可用 Skill 列表。列表通常只包含 `name`、`description` 和路径信息。模型在生成下一步响应时，可以根据这些摘要判断当前任务是否需要某个 Skill。
>
> - **在工具列表中提供 Skill 加载能力**  
> 一些实现会把 Skill 加载做成工具，例如 `skill`、`use_skill` 或 `read_skill`。模型在看到 Skill 摘要后，可以生成工具调用，请求加载某个 Skill。工具执行结果会作为新的上下文消息返回给模型，其中包含完整 `SKILL.md`。
>
> - **通过文件读取结果注入完整 Skill**  
> 如果模型可使用文件读取工具，Harness 可以只在摘要中提供 Skill 路径。模型判断某个 Skill 相关后，请求读取该路径下的 `SKILL.md`。文件内容随后作为工具返回结果进入上下文。
>
> - **通过显式调用直接插入 Skill 正文**  
> 当用户通过命令或菜单指定 Skill 时，Harness 可以跳过模型的自动选择过程，直接把该 Skill 的完整说明插入后续模型调用的上下文中。此时 Skill 正文通常以系统消息、开发者消息或工具结果的形式出现，具体位置取决于实现。
>
> - **通过包装块标记 Skill 内容边界**  
> 完整 Skill 正文通常不会裸露地拼接进上下文，而是被包装成结构化文本块。包装块会标明 Skill 名称、文件位置和相对路径规则。这样可以让模型区分 Skill 说明、用户输入和普通文件内容。
>
> - **通过参数替换生成本次调用文本**  
> 支持参数的实现会在注入前处理占位符。Harness 先把用户传入的参数替换进 Skill 正文，再把替换后的文本放入上下文。模型看到的是已经绑定本次任务参数的 Skill 说明。
>
> - **通过动态命令输出补充实时上下文**  
> 部分实现允许 Skill 正文声明需要执行的命令。Harness 在模型调用前运行这些命令，并把输出结果插入 Skill 文本中。例如当前 git diff、文件列表、项目状态等信息可以作为上下文的一部分进入模型输入。
>
> - **通过按需文件读取注入辅助资源**  
> `SKILL.md` 可以引用参考资料、模板或脚本路径。模型在后续步骤中请求读取这些文件，Harness 再把文件内容作为工具结果追加到上下文。这样可以避免辅助资源在初始阶段全部进入模型输入。
>
> - **通过工具执行结果注入脚本输出**  
> 如果 Skill 要求运行脚本，模型会生成相应工具调用。Harness 执行脚本后，将标准输出、错误信息或生成文件路径作为工具结果返回。模型后续依据这些结果继续完成任务。
>
> - **通过权限配置影响可用工具集合**  
> 一些实现会在 Skill 激活后调整工具权限或预批准规则。该过程不一定直接表现为模型可见文本，但会影响后续工具调用是否允许执行、是否需要确认、是否被拒绝。
>
> - **通过子上下文隔离 Skill 执行**  
> 支持子 Agent 或 fork context 的实现，会为某次 Skill 调用构造一段新的模型输入。该输入包含任务、Skill 正文和必要资源。子上下文完成后，主上下文只接收结果或摘要。
>
> 因此，Skill 注入可以理解为对模型输入序列的分层构造：初始调用放入摘要，选中后放入完整说明，执行中再追加文件内容、脚本输出、工具结果和权限相关上下文。
