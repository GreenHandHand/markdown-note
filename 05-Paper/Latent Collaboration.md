# Latent Collaboration in Multi-Agent Systems

ICML-2026

**核心一句话**：作者想把多智能体协作从显式文本通信推进到连续隐空间协作。核心方法是让每个 Agent 用最后一层 hidden state 生成 *latent thoughts*，再把这些思考结果通过 layer-wise KV cache 传给下一个 Agent，减少文本解码和重复编码带来的信息压缩与推理开销。论文声称该方法 training-free，并在 9 个 benchmark 上同时提升准确率、降低 token 使用量、加速推理。

---

## Key Contribution

- Pure latent collaboration among LLM agents
- Auto-regressive latent thoughts generation through last-layer hidden states
- Latent working memory transfer via layer-wise KV caches
- Training-free input-output alignment for hidden-state reuse
- Lower token usage and faster inference under sequential and hierarchical MAS settings

作者的真正贡献点可以压缩成一句话：**他把 MAS 中最贵、最容易丢信息的文本中介拿掉，让 Agent 直接交换内部状态**。原文的切入点很清楚：现有 MAS 主要靠自然语言承载每个 Agent 的思考和通信，而 latent reasoning 和 latent communication 过去大多分开研究，作者试图把两者合成一个完整系统。

我注意到这篇文章的立意比普通多 Agent workflow 强一些。普通 MAS 论文常常围绕角色、流程、prompt 编排做设计，这篇文章直接动了通信媒介本身。它的关键假设是：LLM 的 hidden states 已经包含比文本更丰富的语义结构，因此把 hidden states 当成协作语言，可以降低文本化带来的损耗。作者把这个假设拆成三个原则：**Reasoning Expressiveness**、**Communication Fidelity**、**Collaboration Complexity**。

> [!warning] 核心风险
> 这篇论文最强的 claim 是 *lossless* 与 *more expressive*。这两个词很容易让人误解。它的 lossless 更多是在 Transformer KV cache 等价复用意义上的 lossless，依赖模型结构、cache 对齐和相同计算路径。它的 expressive 更多来自一个计数式理论假设，真实任务中是否等价于更强推理能力，还需要实验和消融支撑。

---

## Method

### Problem Setup: TextMAS 的通信瓶颈

**作者的直觉是**：TextMAS 让 Agent 把内部思考先写成文本，再让下一个 Agent 重新编码这些文本，这一步既慢，又可能把连续语义压缩成离散 token。

在标准 MAS 中，作者采用两个代表性结构作为实验底座。第一个是 sequential MAS，由 planner、critic、refiner、solver 串联；第二个是 hierarchical MAS，由 code、math、science 等专家 Agent 独立推理，再交给 summarizer 聚合。作者没有提出新的 Agent 拓扑，而是把同样的 MAS 结构中的通信介质替换成 latent representation。

![[98_Assets/Latent Collaboration.png]]

图 2 其实说明了一个关键点：作者没有试图证明某个 workflow 最优，而是拿两个常见 workflow 测试 latent collaboration 是否能作为通用替换件。这一点让方法更像一个 *communication layer*，而不是一个特定任务 pipeline。

> [!question] 这里的研究问题是否足够干净？
> 作者的问题是 **Can MAS achieve pure latent collaboration?**。这个问题很直接，但也有一个隐含跳跃：如果 latent 协作比文本协作更好，原因可能来自减少 token、减少冗余解码、KV cache 复用、最终 Agent 输入更短等多种因素。论文后面做了一些拆解，但仍然需要更细的控制实验来区分这些因素。

---

### Auto-regressive Latent Thoughts Generation

**作者的直觉是**：模型每一步生成 token 前，最后一层 hidden state 已经包含下一步思考的信息；与其把它投影成词，不如直接把这个 hidden state 当成下一步思考输入。

标准 token generation 是：

$$
f_\theta(x_{t+1}\mid x_{\le t})=\mathrm{softmax}(h_t W_{out})
$$

其中 $h_t$ 是当前位置最后一层 hidden state，$W_{out}$ 是 LM head，把 hidden state 映射到 vocabulary logits。作者的 latent generation 跳过 softmax 和 token sampling，把 $h_t$ 经过对齐后放回输入序列，继续跑 Transformer。这样重复 $m$ 次，得到：

$$
H=[h_{t+1},h_{t+2},...,h_{t+m}]
$$

这些连续向量就是作者称为 *latent thoughts* 的东西。原文明确说，每个 Agent 会用问题 $q$ 和自己的 instruction prompt 得到输入 embedding，然后自回归地产生 $m$ 个 last-layer hidden states。

![[98_Assets/Latent Collaboration-1.png]]

看图 3 左侧，Agent $A_1$ 先从 token embedding layer 得到 $e_1,...,e_t$，经过 Transformer 得到 $h_t$，随后通过 input-output alignment 得到下一步输入 embedding。这个过程重复 $m$ 次，生成一串 latent thoughts。图中橙色块表示最后一层 hidden states，蓝色块表示对齐后的输入向量。

> [!tip] 这个设计的 Aha moment
> 这一步把推理长度从 **文本 token 数** 改成 **latent step 数**。如果一个 latent step 能承载多维连续语义，那么几十个 latent step 可能替代几千甚至上万个文本 token 的 CoT 轨迹。这正是后面速度和 token 节省的来源之一。

> [!warning] 这里有一个非常危险的地方
> last-layer hidden state 原本是给 LM head 使用的，它不天然等价于 input embedding。直接把 $h_t$ 塞回浅层输入，可能造成分布漂移。作者意识到了这个问题，所以引入 input-output alignment。

---

### Input-Output Distribution Alignment

**作者的直觉是**：hidden state 要重新作为输入，就需要把它拉回 token embedding 所在的分布区域，否则模型后续层看到的是异常输入。

作者用一个线性映射 $W_a$ 做对齐：

$$
e = h W_a
$$

并给出近似形式：

$$
W_a \approx W_{out}^{\dagger} W_{in}
$$

这里 $h$ 是 last-layer hidden state，$e$ 是对齐后的输入 embedding，$W_{out}^{\dagger}$ 是 $W_{out}$ 的伪逆，$W_{in}$ 是输入 embedding 矩阵。直观理解是：如果 $W_{out}$ 把 hidden state 映射到词表空间，那么用伪逆可以把 hidden state 先拉回一个与词表语义相容的空间，再映射到输入 embedding 空间。

更细一点，作者在 Appendix 中把这个问题写成最小化：

$$
\min_{W_a}|\beta W_{out}W_a-W_{in}|_F^2
$$

它的闭式解是：

$$
W_a=\frac{1}{\beta}(W_{out}^{\top}W_{out})^{-1}W_{out}^{\top}W_{in}
$$

为了数值稳定，作者进一步加入 ridge regularization：

$$
W_a=\frac{1}{\beta}(W_{out}^{\top}W_{out}+\lambda I)^{-1}W_{out}^{\top}W_{in}
$$

这个矩阵只计算一次，然后在所有 latent reasoning steps 中复用。

![[98_Assets/Latent Collaboration-2.png]]

图 6 很重要。作者展示了未对齐的 $h_t$ 会明显偏离原始输入 embedding $e_t$ 的分布；经过 $W_a$ 后，$e_{t+1}$ 回到接近输入 embedding 的区域。图 7 进一步显示，加上 $W_a$ 后 ARC-Challenge、ARC-Easy、GSM8K 的准确率提升 2.3% 到 5.3%。

> [!question] 我对这个线性对齐仍然保留疑问
> 作者使用 $W_{out}$ 和 $W_{in}$ 构造 $W_a$，这个设计简洁，并且 training-free。但它默认 output hidden state 到 input embedding 的错位可以被一个全局线性映射修正。对于不同层深、不同任务、不同推理阶段，这个全局线性映射是否稳定，论文主要用可视化和下游增益说明，还没有完全解释其边界。

---

### Expressiveness of Latent Thoughts

**作者的直觉是**：连续向量的语义容量高于离散 token，因此相同长度的 latent sequence 可以承载更多思考信息。

作者给出的核心定理是：在 Linear Representation Hypothesis 下，如果长度为 $m$ 的 latent thoughts 可以被文本无损表达，那么文本长度至少需要：

$$
\Omega\left(\frac{d_h m}{\log |V|}\right)
$$

其中 $d_h$ 是 hidden dimension，$m$ 是 latent steps，$|V|$ 是词表大小。作者进一步说 latent thoughts generation 可以达到：

$$
O\left(\frac{d_h}{\log |V|}\right)
$$

倍的表达效率优势，并给出 Qwen3-4B、8B、14B 对应 235.7、377.1、471.4 倍的估算。

这里的证明依赖一个很强的表示假设：hidden embedding 可以写成一组语义基的线性组合，每个语义系数为 $0,\pm1$。因此一个 hidden embedding 的可能语义状态数量是 $3^{d_h}$，长度为 $m$ 的 latent sequence 就有 $3^{d_hm}$ 种状态。要用长度 $m'$ 的文本序列覆盖这些状态，需要满足 $|V|^{m'}\ge 3^{d_hm}$，于是得到下界。

> [!warning] 理论证明的价值与局限
> 这个证明更像是 **容量上界层面的 argument**。它说明连续 hidden state 的可表达状态空间更大，但没有直接证明模型在真实任务中会有效使用这些状态。真实推理质量还取决于 latent trajectory 是否可控、是否稳定、是否能被后续 Agent 正确读取。

---

### Thoughts Transfer via Latent Working Memory

**作者的直觉是**：如果一个 Agent 的思考已经体现在所有层的 KV cache 中，那么后续 Agent 可以直接读取这些 cache，而无需把思考先写成文本再重新编码。

作者定义 Agent $A_1$ 的 latent working memory 为所有层的 KV cache 集合：

$$
M_{A_1}={(K_{A_1,cache}^{(l)},V_{A_1,cache}^{(l)})\mid l=1,2,...,L}
$$

其中：

$$
K_{A_1,cache}^{(l)}=[K_{A_1,1}^{(l)},...,K_{A_1,t+m}^{(l)}]
$$

$$
V_{A_1,cache}^{(l)}=[V_{A_1,1}^{(l)},...,V_{A_1,t+m}^{(l)}]
$$

这个 memory 同时包含原始输入上下文和 $A_1$ 新生成的 latent thoughts。后续 Agent $A_2$ 在生成自己的 latent thoughts 前，把 $A_1$ 的 layer-wise KV cache 拼接到自己的 KV cache 前面，于是 $A_2$ 的生成会同时条件化于 $A_1$ 的 working memory 和自身内部表示。

![[98_Assets/Latent Collaboration-1.png]]

图 3 右侧是这篇文章最关键的结构。$A_1$ 的 KV cache 被收集成 latent working memory，然后按层传给 $A_2$。$A_2$ 的 cache 不是只包含自己的输入，还拼接了 $A_1$ 的 input + latent。这样后面的 Agent 能访问前面 Agent 的内部推理轨迹，而不用读一段自然语言解释。

作者提出 Theorem 3.3，说明接收 latent working memory 与直接输入前序 Agent 输出在计算结果上等价。证明思路并不神秘：如果 cache 来自同一个模型、同一输入、同一计算路径，那么当前层 attention 看到的 keys 和 values 一样，后续 deterministic computation 也会得到一样的 hidden state。

> [!tip] 这个模块的真正技术价值
> KV cache 传递同时解决两个问题：一是保留信息，二是避免重复计算。直接传 hidden states 仍然可能需要后续模型重新构造每层 attention 所需的 K/V；传 KV cache 则把后续 Agent 需要 attend 的内容提前准备好。

> [!warning] lossless 的前提非常强
> 作者在 Appendix 里承认，为了 training-free 和简化，他们假设所有 Agent 共享相同形状的 Transformer layers。若要支持异构 Agent，需要引入 trainable adapter 做跨模型 latent representation 对齐。
> 我注意到这直接限制了方法的现实适用范围。很多 MAS 的价值恰好来自异构模型，例如强推理模型、代码模型、视觉模型、工具模型组合。LatentMAS 要进入这类系统，adapter 可能会从辅助组件变成核心组件。

---

### End-to-End Pipeline and Complexity

**作者的直觉是**：中间 Agent 全部在 latent space 内思考和通信，只有最后一个 Agent 解码出自然语言答案，这样系统级成本会显著下降。

整个 pipeline 是：$A_1$ 生成 latent thoughts，传递 $M_{A_1}$；$A_2$ 继承 memory 后继续生成 latent thoughts，传递 $M_{A_2}$；这个过程一直持续到最后一个 Agent，最后一个 Agent 才输出文本答案。

作者给出 LatentMAS 每个 Agent 的复杂度：

$$
O((d_h^2m+d_hm^2+d_htm)L)
$$

其中 $t$ 是当前 Agent 的输入长度，$m$ 是 latent thought 长度，$L$ 是层数。TextMAS 若要达到同等表达能力，复杂度会出现与 $d_h/\log |V|$ 相关的放大项，并额外承担 vocabulary decoding 成本。

> [!warning] 复杂度分析的核心依赖
> 复杂度优势依赖一个前提：较短 latent steps 的表达能力可以替代大量 text tokens。这个前提在理论上来自 Theorem 3.1，在实验中通过速度、token 和准确率结果间接支持。但如果某些任务必须显式搜索、符号计算或外部工具调用，latent step 是否仍能替代长文本轨迹，需要单独验证。

---

## Experiments

### Setup

作者在 9 个 benchmark 上评估，包括 GSM8K、AIME24、AIME25、GPQA-Diamond、MedQA、ARC-Easy、ARC-Challenge、MBPP-Plus、HumanEval-Plus，覆盖数学与科学推理、常识推理、代码生成。模型使用 Qwen3 4B、8B、14B 和 Llama3 3B、8B。baseline 包括 Single LLM、Sequential TextMAS、Hierarchical TextMAS。

> [!note] 指标解释
> 作者看三个指标：**Accuracy** 衡量任务正确率，**Token** 衡量系统总输出 token，**Speed** 衡量端到端推理时间。这里的重点不是单纯追求 accuracy，而是看 latent communication 能否同时提升质量和效率。

### Main Results

![[98_Assets/Latent Collaboration-3.png]]

Figure 1 给出最强的总览：在 hierarchical MAS 下，LatentMAS 相比 single model 和 text-based MAS，平均准确率提升 13.3%，平均 token 减少 83.7%，平均推理速度提升 4.3 倍。

![[98_Assets/Latent Collaboration-5.png]]

在主实验中，作者总结 LatentMAS 相比 single-model baseline 在 sequential 和 hierarchical 设置下平均提升 14.6% 和 13.3%，相比 text-based MAS 进一步提升 2.8% 和 4.6%。同时，它比 TextMAS 平均快 4 倍和 4.3 倍，token 使用量减少 70.8% 和 83.7%。

我注意到这里最稳定的优势其实是 **效率**，而非所有任务上的准确率。Table 1 中有若干单元格 LatentMAS 比 TextMAS 略低，例如 Qwen3-8B 的 ARC-E、ARC-C，Qwen3-14B 的 ARC-C 等。也就是说，论文总体趋势成立，但不应把它理解为每个任务每个模型规模都严格提升。

> [!warning] 结果解读
> 这篇文章最强证据是 token 和 speed 的大幅下降。accuracy 的提升有价值，但相对 TextMAS 的平均增益更温和。若后续复现，应优先检查：相同最大输出长度、相同温度、相同 prompt template、相同 vLLM 加速设置下，TextMAS 是否还有进一步压缩空间。

---

### In-depth Analyses

#### Latent thoughts 是否真的有语义

**作者的直觉是**：如果 latent thoughts 真能代替文本推理，它们至少应该落在和文本 token embeddings 相近的语义区域。

![[98_Assets/Latent Collaboration-6.png]]

作者比较 LatentMAS 新生成 last-layer embeddings 和 TextMAS token-by-token response embeddings。Figure 5 显示两者在 embedding space 中高度重叠，且 LatentMAS 的点覆盖范围更大。作者据此认为 latent thoughts 与文本推理语义一致，并且具有更高表示多样性。

Appendix 进一步用平均 pairwise cosine similarity 量化多样性。LatentMAS 在 Qwen3-4B、8B、14B 上的相似度均低于 TextMAS，说明生成表示更分散、更少塌缩。

> [!question] 这里我会继续追问
> embedding 分布重叠只能说明 latent thoughts 落在相近区域，不能直接说明它们表达了正确推理步骤。更强的证据应当包括 latent trajectory 与可验证中间状态的对应关系，或者在可控任务中对 latent step 进行因果干预。

#### latent steps 的深度

**作者的直觉是**：latent steps 太少，思考不够；latent steps 太多，会引入冗余或漂移。

![[98_Assets/Latent Collaboration-7.png]]

Figure 8 显示 latent steps 从 0 增加时，性能通常上升，在 40 到 80 步左右达到峰值，之后平台化或下降。作者最后选择中等 latent step budget，追求 accuracy-efficiency trade-off。

> [!note] 这个实验很关键
> 它说明 latent reasoning 不是免费午餐。更多 hidden-state rollout 会带来更多内部计算，也可能累积噪声。这个现象和普通 CoT 长度类似：适度推理有帮助，过长推理会增加错误传播。

#### Latent reasoning 与 latent communication 的消融

**作者的直觉是**：LatentMAS 的收益来自两个部分同时成立，一个是 Agent 内部用 latent steps 推理，另一个是 Agent 之间用 latent working memory 通信。

![[98_Assets/Latent Collaboration-8.png]]

Appendix 的 hybrid ablation 显示，*Latent Reasoning + Text Communication* 和 *Text Reasoning + Latent Communication* 都低于完整 LatentMAS。以 Qwen3-8B 为例，LatentMAS 在 GSM8K、MBPP+、MedQA 上分别达到 93.8、74.6、75.3，两个 hybrid 版本均更低。

> [!tip] 这个消融做得比较对
> 它没有只比较完整方法和 TextMAS，而是拆开 reasoning medium 与 communication medium。这个实验直接回答了一个重要问题：收益是否只来自少解码 token。结果显示，单独替换一个部分不够，完整 latent collaboration 才最强。

---

### Debug Mode and Interpretability

**作者的直觉是**：纯 latent 系统很难审计，因此需要一个旁路，把中间 latent thoughts 对应的文本探针暴露出来。

作者提出 *debug mode*：每个 Agent 同时生成 latent thoughts 和一段并行文本响应。latent thoughts 继续传给下一个 Agent，文本响应只作为 probe，帮助研究者检查中间错误。作者在 GSM8K 上做了 100 个 debug text 与最终答案的相关性分析，最终答案正确时，中间 debug text 也正确的比例为 96.2%；最终答案错误时，中间 debug text 错误的比例为 90.0%。

case study 展示了 debug mode 下错误与正确案例。错误案例中，Refiner 给出错误数值关系，最终 Solver 继承了这个错误。正确案例中，中间 Agent 的文本 probe 保持一致，最终得到正确答案。

> [!warning] debug text 不是 latent thought 本身
> 我注意到 debug mode 的解释性仍然是间接的。它证明并行文本与最终答案高度相关，但不能完全证明文本 probe 忠实还原了 latent thoughts。它更像一个工程上可用的错误定位工具，而不是严格的 latent interpretability 方法。

---

## Critical Reading

### 1. 论文的核心直觉很强，但理论 claim 偏激进

这篇论文最有价值的直觉是：**多 Agent 协作的瓶颈可能来自文本通信，而不是 Agent 数量或角色设计**。如果每个 Agent 的内部状态本来就包含丰富语义，那么反复写成文本再读回去会造成计算浪费和语义压缩。

但 *lossless* 和 *expressive* 的理论证明有较强前提。lossless 建立在 KV cache 等价复用和相同模型计算路径上；expressiveness 建立在 hidden semantic basis 的计数假设上。理论可以支撑方法动机，但还不足以独立证明真实任务上的推理优势。

### 2. 方法对同构模型更自然，对异构 MAS 仍不完整

作者承认当前为了 training-free，假设 Agent 具有相同 Transformer layer shape。异构 Agent 需要 trainable adapter 对齐 latent representations。

这会影响方法的扩展性。真实 MAS 往往混合不同模型和工具，例如 coding model、math model、vision model、retriever、verifier。若都要通过 adapter 对齐，LatentMAS 会从 training-free 框架变成需要额外训练的 latent protocol learning 问题。

### 3. 效率优势比准确率优势更可靠

实验上最稳定的收益是 token usage 和 speed。accuracy 也有提升，但存在若干局部下降。作者的结论应理解为：在相同 MAS 架构下，latent collaboration 通常能以更低成本达到相近或更高性能，而不是每个任务都严格优于 TextMAS。

### 4. 需要补充更强 baseline

我会希望看到这些 baseline：

- **Compressed TextMAS**：中间 Agent 只输出短摘要或结构化要点，检查 LatentMAS 是否仍有优势。
- **Hidden-state only transfer**：只传最后层 hidden states，不传 layer-wise KV cache，区分 KV 机制的贡献。
- **KV sharing baseline**：传 prefill KV 或输入上下文 KV，检查 newly generated latent thoughts 的贡献。
- **Verifier-enhanced TextMAS**：给 TextMAS 加强最终验证，判断 LatentMAS 的准确率优势是否来自最终 Agent 更短上下文，还是来自 latent 信息本身。
- **Heterogeneous MAS**：混合 Qwen、Llama 或 specialized model，测试 adapter 成本和稳定性。

---

## Future

作者自己提到两个方向。第一，扩展到 heterogeneous agents，需要 layer mapping 或 trainable adapter。第二，把 text-based MAS 中的 post-training 范式迁移到 LatentMAS，用来优化 latent collaboration protocol。

我认为更值得挖掘的方向有四个。

1. **Learned latent communication protocol**
   当前 $W_a$ 是一次性线性对齐，KV transfer 也是直接拼接。下一步可以让系统学习什么时候传、传多少、传哪些层的 KV，而不是全量传递。

2. **Causal interpretability for latent thoughts**
   debug mode 只能做相关性解释。更强方向是对 latent steps 做 ablation、patching、steering，验证某个 latent segment 是否对应具体推理子目标。

3. **Memory compression for latent MAS**
   多 Agent 连续传 KV cache 会导致 memory 长度增长。可以研究 latent working memory 的压缩、裁剪、检索与遗忘机制。这与长期 Agent memory 很接近。

4. **Heterogeneous latent alignment**
   真实 Agent 系统很难只用同构模型。跨模型、跨模态、跨工具的 latent alignment 可能是 LatentMAS 进入实际 MAS 的关键瓶颈。
