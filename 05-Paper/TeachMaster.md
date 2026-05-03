# TeachMaster: Generative Teaching via Code

> [!note] ACL Industry Track 2026

**核心一句话**：作者面向教育视频生产成本高、周期长、修改困难的问题，提出以代码作为中间表示的多 Agent 生成流程，将教学意图逐步转化为讲稿、页面蓝图、Manim 动画代码、旁白音频和可渲染视频，并通过调试、同步、布局修复和人工编辑保证生产稳定性。

---

## Key Contribution

- **We propose Generative Teaching, a novel paradigm that shifts the educator’s role from manual creator to high-level director.**
- **We introduce TeachMaster, a multi-agent framework that utilizes code as an intermediate semantic medium.**
- **Extensive experiments and real-world deployment across diverse disciplines validate TeachMaster.**

作者的基本意图是降低教育视频生产的人工成本。传统在线课程依赖脚本撰写、课件设计、动画制作、录制和后期编辑，流程长、成本高、更新慢。作者把教师定位为教学意图的提出者，把具体制作流程交给系统执行。

论文中的技术承载点是 **code-centric workflow**。代码在这里承担三类功能：第一，作为视觉内容的生成表示；第二，作为可执行对象用于调试和渲染；第三，作为可编辑对象支持人工修改。这个设计使系统能够避免直接生成像素视频带来的不可控性。

从贡献形式看，论文更接近工程系统论文。它的价值主要体现在完整流程、部署规模和成本效率，而不是单一算法创新。多 Agent 编排是系统组织方式，Manim/Python 代码生成是实际生产链路中的核心表示。

---

## Method

### Overall Motivation

> [!question] 作者为什么要引入这个系统？
> 教育视频需要同时满足教学结构、视觉表达、旁白解释和音画同步。直接使用视频生成模型可以生成画面，但难以保证结构清晰、内容可改和局部修复。作者因此选择把视频生成拆成多个可控阶段，并把代码作为中间表示。

作者将 TeachMaster 定义为一个从教学意图到教学视频的多阶段生产系统。输入是 lecture outline 或 keywords，输出是视频和讲稿。

$$
O = F(k, \Phi, f_{human})
$$

其中，$k$ 表示课程大纲或关键词，$\Phi$ 表示可选配置，$f_{human}$ 表示人工介入，$O={V_{out}, L_{out}}$ 表示最终视频和讲稿。

---

### System Architecture

**Intuition**：作者的直觉是，教育视频生产可以拆成有依赖关系的流水线，每个阶段处理一种明确的产物。

TeachMaster 的流程分为三部分：

1. **Content Planning**：生成讲稿并拆分为页面级蓝图。
2. **Presentation Generation**：将页面蓝图转为视觉代码、旁白和音频。
3. **Quality Validation**：修复代码错误、同步音画、调整布局，并支持人工介入。

![[98_Assets/TeachMaster.png]]

图 2 是论文的主架构图。上半部分是用户视角，包括课程创建、历史记录、任务详情、视频下载和分享。下半部分是系统框架，包括内容规划、展示生成和质量验证三个阶段。这个图同时表达产品工作流和 Agent 工作流，符合 Industry Track 对可部署系统的写法。

> [!note] 结构观察
> 图 2 中的 Agent 分工以产物为边界，而不是以模型能力为边界。composition agent 产出讲稿，pagination agent 产出页面单元，coding agent 产出代码，narration agent 产出旁白，debugging、synchronization 和 layout agent 负责后处理。

> [!warning] 论证不足
> 论文没有提供系统性消融来证明这些 Agent 拆分的必要性。缺少对比对象包括单 Agent 直接生成 Manim 代码、无分页模块、无同步模块、无布局修复模块等。因此，实验结果能说明系统整体有效，但不能明确归因到具体 Agent 设计。

---

### Content Planning

**Intuition**：作者先生成讲稿和页面蓝图，是为了让后续视觉生成有稳定的内容结构。

内容规划阶段由 composition agent 和 pagination agent 完成。composition agent 将输入的大纲扩展为完整讲稿，并根据目标时长进行调整。

$$
L_{out}=R(E(S(k)),t)
$$

其中，$S(k)$ 表示语义骨架生成，$E(\cdot)$ 表示内容扩展，$R(\cdot,t)$ 表示根据目标时长 $t$ 进行长度修正，$L_{out}$ 是生成讲稿。

随后，pagination agent 将讲稿划分为页面级单元。作者使用 Chain-of-Agents 思路处理长文本，将讲稿拆成多个片段并由局部 Agent 处理。

$$
F_{CoA}(L_{out})=\bigcup_{i=1}^{n}F_i(D_i)={P_i}_{i=1}^{n}
$$

其中，$D_i$ 是讲稿片段，$F_i$ 是局部分页 Agent，$P_i$ 是页面级蓝图。

> [!question] 页面蓝图没有明确定义
> 论文没有给出 $P_i$ 的结构。它可能是自然语言 storyboard，也可能是结构化页面描述。后续代码生成、旁白生成和同步都依赖 $P_i$，因此这是影响复现的重要缺口。

> [!note] 与智慧教材生成的对应关系
> $P_i$ 可以类比为教材生成中的阶段规划单元。若迁移到智慧教材场景，页面蓝图应扩展为章节、节、知识点、教学目标、学习者状态和已生成内容共同组成的规划状态。

---

### Presentation Generation

**Intuition**：作者将视觉生成和旁白生成分开处理，再通过代码和上下文信息建立两者之间的对应关系。

这一节在论文中称为 **Presentation Generation**。从实现细节看，其核心产物是视觉代码。论文在 Method 中使用抽象符号 $C_i$ 表示代码，Implementation Details 中说明系统通过 Manim 引擎渲染可执行 Python 脚本。

#### Routing Agent and Coding Agent

视觉生成公式为：

$$
C_i=F_{vis}(P_i,Mode)
$$

其中，$P_i$ 是页面蓝图，$C_i$ 是生成的视觉代码，$Mode \in {Standard, ImageEnhanced}$。Standard 模式用于公式、几何图形、流程关系等可程序化表达的内容；ImageEnhanced 模式用于需要写实图像或复杂视觉素材的页面。

![[98_Assets/TeachMaster.png]]

从图 2 的 Presentation Generation 模块可以看到，routing agent 决定生成路径，coding agent 生成代码，narration agent 生成旁白。这个流程说明作者将视觉表达视为可路由的生成任务。

> [!warning] Method 对 Manim 表示的说明不足
> Manim 代码是系统可控性的主要来源，但论文没有在 Method 中展开说明代码规范、对象命名、事件锚点、可编辑边界和 API 约束。若 $C_i$ 是自由形式代码，后续同步和布局修复会面临较高不稳定性。

> [!question] ImageEnhanced 路径缺少细节
> 作者提到需要写实素材时会触发 image-enhanced coding agent，但没有说明图像如何生成、如何插入 Manim 场景、如何保证图像内容和教学语义一致。

#### Code as Intermediate Representation

在 TeachMaster 中，代码承担了视觉表示层的作用。一个页面的 Manim 代码通常会包含对象、坐标、颜色、层级、动画顺序和时间控制等信息。与直接生成视频相比，代码可以被执行、调试、局部修改和人工编辑。

这一设计支撑了后续三个能力：

- **可执行性**：代码运行失败后可以得到错误信息。
- **可编辑性**：教师可以通过自然语言或直接代码修改视频。
- **可同步性**：动画事件可以与旁白时间进行对齐。

> [!note]
> 我注意到，作者把 code 称为 intermediate semantic medium，但论文展示的其实是工程上的可执行表示。它是否具有稳定的语义结构，取决于代码生成规范是否足够严格。论文对此没有充分展开。

---

### Narration and Audio Generation

**Intuition**：旁白需要知道当前页面讲什么、前文讲到哪里、画面中实际出现了什么。

旁白生成公式为：

$$
T_i=F_{narr}(P_i,T_{i-1},C_i)
$$

其中，$P_i$ 是当前页面蓝图，$T_{i-1}$ 是前一页旁白，$C_i$ 是当前页视觉代码，$T_i$ 是当前页旁白。

这个设计用于保持两种一致性：一是相邻页面之间的术语和叙事连续性，二是旁白内容和视觉对象之间的对应关系。

随后，TTS agent 将旁白转换为音频，并计算语速：

$$
A_i,r_i=F_{tts}(T_i)
$$

其中，$A_i$ 是音频，$r_i$ 是 speaking rate。$r_i$ 会被后续 synchronization agent 用于确定动画触发时间。

> [!tip] 值得保留的设计
> 旁白生成读取 $C_i$，可以减少画面和讲解脱节的问题。对于教学内容，这比单独生成脚本再配画面更稳定。

> [!question] 代码可读性假设
> 该设计隐含一个前提：narration agent 能够从 $C_i$ 中准确理解画面内容。如果代码过长、对象命名混乱或动画逻辑复杂，这一前提可能不稳定。

---

### Quality Validation

**Intuition**：作者把质量控制设计成代码层面的闭环，使生成结果可以被检测和修复。

质量验证包含 debugging、synchronization 和 layout 三个 Agent。

#### Debugging Agent

$$
C_i^{debug}=F_{debug}(C_i,Error(C_i))
$$

debugging agent 通过渲染结果发现语法错误或运行错误，并把错误信息反馈给模型修复代码。若多次修复失败，系统会启用 fallback，将复杂元素替换为标准模板。

> [!note] 这一模块解决的问题
> 代码生成的不稳定性是实际部署中的常见问题。debugging agent 使系统能够从失败的渲染中恢复，降低人工介入频率。

> [!warning] 质量边界
> 该模块主要处理代码能否运行，不能保证教学内容正确、讲解顺序合理或知识依赖完整。

#### Synchronization Agent

$$
C_i^{sync}=F_{sync}(C_i^{debug},T_i,r_i)
$$

synchronization agent 根据调试后的代码、旁白和语速，在代码中插入时间控制逻辑，使画面变化和语音讲解对齐。

> [!question] Event anchors 没有说明来源
> 论文提到 synchronization agent 利用代码中的 event anchors，但没有说明这些 anchors 是由 coding agent 生成，还是由后处理模块解析得到。若 anchors 不稳定，音画同步的可控性会受影响。

#### Layout Agent

布局修复公式为：

$$
O_i=F_{detect}(C_i^{sync})
$$

$$
\Omega_i=F_{retrieve}(O_i,dir_{h,v})
$$

$$
C_i^{layout}=F_{layout}(C_i^{sync},\Omega_i)
$$

其中，$O_i$ 表示检测到的重叠或遮挡，$\Omega_i$ 表示调整后的坐标，$dir_{h,v}$ 表示水平和垂直搜索方向，$C_i^{layout}$ 是布局修复后的代码。

> [!warning] 动态布局检测不清楚
> Manim 视频包含对象移动和变化。论文没有说明 overlap 检测针对最终帧、关键帧还是完整动画过程。若只检测静态帧，可能无法覆盖动画过程中的遮挡。

#### Human-in-the-loop

$$
C_i^{final}=F_{human}(C_i^{layout})
$$

系统支持两类人工介入：自然语言修改和直接代码编辑。这个设计保留教师对教学逻辑和专业内容的控制权，也符合教育内容生产中需要人工审核的现实需求。

---

## Experiments

### Evaluation Setup

> [!note] 重要指标解释
> 论文的评测目标是系统产物质量和生产效率，而不是学生学习效果。作者使用 GPT-5.2 对开放式生成结果进行 1 到 10 分评分，并使用 300 个视频上的 3 位专家偏好判断验证人机一致性。作者报告专家与 GPT-5.2 的总体一致率为 81.71%。

评测包含三组指标：

1. **Video Generation Quality**：视觉清晰度、视觉丰富度、教学逻辑、图文对应和事实准确性。
2. **Educational Script Quality**：叙事连贯性、准确性、完整性和一致性。
3. **Cross-modal Semantic Alignment**：语义覆盖、指代准确性和视觉语言对称性。

baseline 包括：

- **Sora 2**：代表端到端视频生成模型。
- **Human-Crafted**：代表人工制作视频。

> [!warning] Baseline 设置偏向系统叙事
> Sora 2 适合作为端到端视频生成参照，但它不适合作为代码驱动教学视频生产系统的强 baseline。更直接的对比应包括单 Agent Manim 代码生成、模板化课件视频系统、无质量验证模块版本，以及其他 code-to-video 系统。

---

### Video Generation Quality and Efficiency

![[98_Assets/TeachMaster-1.png]]

表 1 比较视频质量和效率。Human 的 Overall 为 8.29，TeachMasterGemini 为 7.91，TeachMasterQwen 为 7.59，Sora 2 为 7.57。TeachMasterGemini 的质量接近人工，并高于 Sora 2。

效率方面，Human 的 Ratio 为 24.46，Sora 2 为 12.80，TeachMasterGemini 为 2.46，TeachMasterQwen 为 3.47。Ratio 表示生产 1 分钟视频所需时间。作者据此说明 TeachMaster 在效率上显著优于人工制作和端到端视频生成。

> [!warning] 对比解释需要谨慎
> 表 1 中 Sora 2 的总视频时长为 0.25 分钟，而 Human 和 TeachMaster 都是 30 分钟以上。这说明 Sora 2 在长课程生产任务上不适配，但不能直接推出 TeachMaster 在所有视频生成场景中优于 Sora 2。

---

### Educational Script Quality

![[98_Assets/TeachMaster-2.png]]

表 2 显示 TeachMasterGemini 的 Overall 为 8.95，高于 Human 的 8.84。TeachMasterQwen 为 8.34，明显高于 Sora 2 的 4.39。Sora 2 在 Completeness 上得分很低，说明端到端视频生成难以覆盖完整教学脚本。

这一结果与系统设计一致。TeachMaster 先做讲稿生成和分页规划，因此脚本完整性和结构连贯性更容易维持。

> [!warning] 缺少脚本模块消融
> 论文没有比较仅使用 composition agent 和 pagination agent 的脚本质量，也没有说明后续视频生成模块是否影响脚本质量。因此，表 2 主要证明内容规划流程有效，不能证明完整多 Agent 系统的所有模块都必要。

---

### Cross-modal Semantic Alignment

![[98_Assets/TeachMaster-3.png]]

表 3 比较语义覆盖、指代准确性和视觉语言对称性。TeachMasterQwen 的 Overall 为 8.79，TeachMasterGemini 为 8.44，高于 Human 的 8.13 和 Sora 2 的 6.65。

这组结果与方法设计有较强对应关系。TeachMaster 的 narration agent 读取页面蓝图、前序旁白和视觉代码，因此旁白和画面之间更容易形成对应关系。代码中间表示为跨模态同步提供了可操作对象。

> [!note] 这组指标与方法关系最直接
> 相比视频质量和脚本质量，cross-modal alignment 更能体现代码中间表示和 code-aware narration 的作用。

> [!warning] 仍需更细粒度验证
> 论文没有展示音画不同步的失败案例，也没有报告 synchronization agent 修复前后的对比。因此，表 3 支持整体效果，但无法量化同步模块本身的贡献。

---

### Deployment and User Feedback

![[98_Assets/TeachMaster-4.png]]

图 3 展示教师和学生反馈，包括备课时间减少、视频覆盖教学意图、系统相对传统方法的优势，以及学生希望增加的内容类型。作者报告超过 75.2% 的页面无需人工干预，其余页面平均 1.88 轮交互完成。

![[98_Assets/TeachMaster-5.png]]

图 4 展示学科分布。作者报告系统已服务超过 1000 名教育者，生成超过 30000 分钟教育内容，覆盖 40 多个学科，并声称 45 小时课程生产成本约为 83.70 美元，约为传统课程制作成本的 0.3%。

> [!note] Industry Track 证据
> 部署规模、用户反馈和成本估算是这篇论文的重要证据。这些结果说明系统有实际应用价值。

> [!warning] 用户反馈不能替代学习效果评测
> 教师感知到备课时间减少，学生感知到视频清晰，并不等同于学习效果提升。论文没有提供 pre-test/post-test、长期保持、迁移任务或个性化学习效果实验。

---

### Case Study

![[98_Assets/TeachMaster-6.png]]

图 5 展示监督学习、人工智能导论、分子生物学和语言学四类案例。案例体现了 Manim 适合表达公式、结构关系、几何示意和流程动画。对于抽象数学、计算机科学和部分自然科学内容，代码驱动动画具有较好的表达优势。

> [!warning] 案例展示存在选择性
> 论文展示的是成功案例。对于失败案例，例如布局拥挤、图像素材不准确、旁白引用错误或动画节奏不自然，论文没有提供系统分析。

---

## Critical Reading

### 1. 论文性质

这篇论文的主要贡献属于工程系统层面。作者将已有能力组合成一个可部署流程，包括 LLM 生成讲稿、LLM 生成代码、Manim 渲染、TTS、代码调试、音画同步、布局修复和人工编辑。

其研究价值主要体现在三个方面：

- 将教育视频生产转化为可执行代码生成问题。
- 将多模态视频生产拆分为可验证的流水线。
- 通过真实部署和成本估算展示系统实用性。

### 2. Method 的主要不足

方法部分对中间表示说明不足。论文强调 code-centric，但没有充分定义以下内容：

- $P_i$ 的结构。
- $C_i$ 的代码规范。
- Manim API 使用范围。
- event anchors 的格式和来源。
- layout 检测的具体对象。
- image-enhanced 模式的图像生成和对齐方式。
- fallback 机制的触发条件和模板类型。

这些细节会影响系统可复现性，也会影响读者判断多 Agent 编排是否具有稳定的技术贡献。

### 3. Evaluation 的主要不足

实验能够支持 TeachMaster 作为整体系统的有效性，但缺少模块级归因。

主要不足包括：

- 缺少 ablation study。
- 缺少单 Agent Manim 代码生成 baseline。
- 缺少相近系统对比。
- Sora 2 baseline 与 TeachMaster 的任务形态不完全一致。
- LLM-as-judge 指标缺少更细粒度人工标注。
- 用户反馈没有转化为学习效果评估。

## Future

### 作者方向

作者希望进一步推动 Generative Teaching，使教师从重复性内容生产中释放出来，将更多精力投入个性化教学指导。系统层面可能继续扩展生成规模、学科覆盖和人工编辑体验。

### 可继续挖掘的方向

1. **中间表示规范化**
   明确定义 storyboard、visual object graph、event anchor graph 或教材 planning state schema。
2. **模块级消融实验**
   分别验证 pagination、routing、code-aware narration、synchronization 和 layout repair 的贡献。
3. **更公平的 baseline**
   加入单 Agent Manim 代码生成、模板化视频生产、已有 code-to-video 系统和无后处理版本。
4. **从感知质量走向学习效果**
   引入 pre-test/post-test、延迟测试、迁移任务、认知负荷和错误概念修正效果。
