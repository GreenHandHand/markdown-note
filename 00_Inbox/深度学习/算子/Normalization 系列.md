---
tags:
  - 深度学习
---

# Normalization

随着神经网络从浅层模型发展为深层模型，训练过程逐渐受到数值尺度和优化稳定性的限制。每一层的参数更新都会改变后续层接收到的输入分布，使中间激活的幅度、梯度传播的状态以及损失曲面的局部形态持续变化。当网络层数增加时，这种变化会在层间不断累积，使模型对参数初始化、学习率设置和优化器超参数更加敏感，也更容易出现训练缓慢、梯度不稳定或激活值异常放大的现象。

Normalization 相关的方法，可以看作是对深层网络内部表示进行数值约束的一类方法。它通过在网络中显式引入统计变换，将中间激活调整到更可控的尺度范围内，从而改善前向传播中的数值稳定性和反向传播中的梯度条件。对于优化过程而言，这类约束能够降低不同层之间尺度变化带来的干扰，使参数更新更容易在稳定区域内进行，并提高模型使用较大学习率训练的可行性。

从更一般的角度看，normalization 层承担的是一种训练过程中的结构性调节作用。它并不改变任务目标函数本身，却会改变模型参数化后的优化形态。通过控制中间表示的尺度，normalization 可以缓解深层网络中由层间耦合带来的训练困难，使模型容量的增加更容易转化为可训练的表达能力。因此，在深度学习的发展中，normalization 逐渐成为连接模型结构设计、数值稳定性和优化效率的重要机制。

## Batch Normalization

Batch Normalization 是一种基于 mini-batch 统计量的中间激活归一化方法。它通常被插入到神经网络的线性变换或卷积之后、非线性激活之前，用当前 mini-batch 中样本的统计量对激活值进行标准化，再通过可学习的仿射参数恢复模型需要的表达尺度。其基本作用是让每一层接收到的输入保持在更稳定的数值范围内，从而改善深层网络训练过程中的优化条件。[[00_Inbox/深度学习/现代卷积神经网络#批量规范化|这里有一些以前的内容]]。

设某一层的中间激活为：

$$
x = \{x_1, x_2, \dots, x_m\}
$$

其中 $m$ 表示 mini-batch 中参与统计的样本数量。Batch Normalization 首先计算当前 mini-batch 的均值和方差：

$$
\begin{align}
\mu_B &= \frac{1}{m}\sum_{i=1}^{m}x_i \\
\sigma_B^2 &= \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2
\end{align}
$$

接着对激活值进行标准化：

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

其中 $\epsilon$ 是一个很小的常数，用于避免分母为零并提高数值稳定性。

标准化后的激活值会再经过一个可学习的仿射变换：

$$
y_i = \gamma \hat{x}_i + \beta
$$

其中 $\gamma$ 和 $\beta$ 是可学习参数。它们的作用是让模型在需要时恢复或调整归一化后的表示分布。没有这两个参数时，归一化会强制限制每一层输出的均值和方差；加入 $\gamma$ 和 $\beta$ 后，模型可以根据任务学习合适的尺度和平移量。

在卷积神经网络中，Batch Normalization 通常按通道进行统计。对于形状为 $x \in \mathbb{R}^{N \times C \times H \times W}$ 的输入，BN 会对每个通道 $C$ 分别计算统计量，统计范围覆盖 batch 维度和空间维度：

$$
(N, H, W)
$$

因此，每个通道拥有独立的 $\mu_B$、$\sigma_B^2$、$\gamma$ 和 $\beta$。这种设计与 CNN 的通道语义相匹配：同一通道通常表示同一类特征响应，不同样本和不同空间位置可以共同估计该通道的激活分布。

训练阶段和推理阶段的 Batch Normalization 行为不同。
- 训练时，BN 使用当前 mini-batch 的均值和方差进行归一化；同时，它会维护一组 running mean 和 running variance，用于估计整个训练数据分布中的统计量。
- 推理时，模型通常不再使用当前输入 batch 的统计量，而是使用训练过程中累计得到的 running statistics。PyTorch 的 BatchNorm 默认会在训练阶段跟踪 running mean 和 running variance，并在 eval 阶段使用这些估计值进行归一化。

这一训练/推理差异可以写成：

训练阶段：

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

推理阶段：

$$
\hat{x} = \frac{x - \mu_{\text{running}}}{\sqrt{\sigma_{\text{running}}^2 + \epsilon}}
$$

其中 $\mu_{\text{running}}$ 和 $\sigma_{\text{running}}^2$ 来自训练过程中对多个 mini-batch 统计量的滑动更新。

> [!note] 历史背景
> Batch Normalization 由 Ioffe 和 Szegedy 在 2015 年提出。原论文将其动机归结为缓解 Internal Covariate Shift，即网络训练过程中中间层输入分布持续变化的问题。论文报告 BN 可以加速训练，使模型能够使用更高学习率，并在某些情况下减少对 Dropout 的依赖。
>
> BN 原始解释强调 Internal Covariate Shift，但后续研究对这一解释提出了修正。Santurkar 等人的研究认为，BN 的有效性与稳定每层输入分布的关系并不充分；更核心的影响可能在于让优化景观更加平滑，使梯度行为更加稳定、可预测，从而加速训练。

> [!note] 目前的共识
> BN 的实际收益主要体现在优化层面。它能稳定激活尺度，缓解深层未归一化网络中激活和梯度随深度失控的问题，使模型更容易使用较大学习率训练。Bjorck 等人的经验研究也指出，BN 很大程度上通过支持更大学习率来带来更快收敛和更好泛化。

> [!tip] 工程注意事项
> BN 对 batch size 较敏感。mini-batch 太小时，均值和方差估计会变得不稳定；训练数据分布和推理数据分布差异较大时，running statistics 也可能成为误差来源。因此，在小 batch、变长序列、自回归生成和分布迁移较强的场景中，BN 的使用需要更加谨慎。

## Layer Normalization

Batch Normalization 的有效性建立在 mini-batch 统计量相对可靠这一前提上。
- 在图像任务中，BN 常处理 `[BatchSize, Channels, H, W]` 大小的张量，同一 batch 内的样本和空间位置可以为每个通道 `C` 提供较稳定的统计估计。
- 但在序列建模中，batch size、序列长度、padding 方式和自回归推理状态都会改变统计条件，使 batch 统计量更容易受到输入组织方式的影响。

Layer Normalization 将统计范围从 batch 维度转移到单个样本的特征维度。对于 Transformer 中的输入 `[BatchSize, SeqLen, Hidden]`，LN 通常在 `Hidden` 维上计算统计量，因此归一化计算不依赖 batch 内其他样本，训练阶段和推理阶段也使用相同的计算过程。

> [!note]
> 从这一角度看，LN 解决的是 BN 对 batch 统计量的依赖问题，尤其适合 batch size 不稳定、序列长度可变和逐 token 推理的场景。
>
> 不过，LN 并没有完全替代 Batch Normalization。二者对应不同的统计假设和模型结构：
> - Batch Normalization 更适合利用 CNN 中相对稳定的通道统计
> - Layer Normalization 更适合约束单个 hidden vector 的特征尺度
>
> 因此，LayerNorm 的出现可以理解为 normalization 方法从视觉网络中的跨样本统计，转向序列模型中的样本内特征统计。

Layer Normalization 的输入通常可以看作一个特征向量 $x \in \mathbb{R}^{C}$，其中 `C` 表示需要归一化的特征维度。LN 会在这个向量内部计算均值 $\mu$ 和方差 $\sigma^{2}$，然后进行标准化：

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}}
$$

其中 $\varepsilon$ 是用于数值稳定的小常数。标准化之后，LN 会引入可学习的仿射参数：

$$
y = \gamma \hat{x} + \beta
$$

其中 $\gamma, \beta \in \mathbb{R}^{C}$ 分别表示逐特征维度的缩放和平移参数。它们允许模型在归一化之后重新学习合适的表示尺度和偏移。

这种计算方式使每个 token 的归一化只依赖自身的 hidden vector，不依赖 batch 中的其他样本，也不依赖其他 token 的 hidden vector。因此，LayerNorm 的行为在训练和推理阶段保持一致，不需要维护 BatchNorm 中的 running mean 和 running variance。

> [!note]
> BatchNorm 和 LayerNorm 都可以写成标准化形式：
>
> $$
> \hat{x} = \frac{x - E[X]}{\sqrt{Var[X] + \epsilon}}
> $$
>
> 二者的主要区别在于 $E[X]$ 和 $Var[X]$ 的统计范围。
> - 对于 CNN 中的 BatchNorm，输入通常是 `[BatchSize, Channels, H, W]`。BN 会为每个通道分别计算 $E[X]$ 和 $Var[X]$，统计范围覆盖 `[BatchSize, H, W]`。因此，每个通道共享一组均值和方差。
> - 对于 Transformer 中的 LayerNorm，输入通常是 `[BatchSize, SeqLen, Hidden]`。LN 会为每个 token 的 hidden vector 分别计算 $E[X]$ 和 $Var[X]$，统计范围覆盖 `Hidden`。因此，每个 token 拥有自己的一组均值和方差。
>
> 简单说，BatchNorm 的统计量来自同一通道在 batch 和空间位置上的分布；LayerNorm 的统计量来自同一个 token 内部的 hidden 特征维度。

> [!note] 提出背景
> Layer Normalization 由 Ba、Kiros 和 Hinton 在 2016 年提出。它的关键出发点是摆脱 mini-batch 统计量对归一化结果的影响，使同一个样本在不同 batch 组织方式下得到一致的归一化计算。这一特点使 LN 更适合 RNN、Transformer 和自回归语言模型。

> [!tip] 实现注意事项
> 手写 LayerNorm 时，最容易出错的是归一化维度。对于 LLM 中常见的 `[BatchSize, SeqLen, Hidden]`，均值和方差通常沿最后一维 `Hidden` 计算，并保留维度用于广播。
>
> 因此，归一化统计量的形状通常是：
>
> ```text
> input:  [BatchSize, SeqLen, Hidden]
> mean:   [BatchSize, SeqLen, 1]
> var:    [BatchSize, SeqLen, 1]
> output: [BatchSize, SeqLen, Hidden]
> ```
>
> 如果没有保留最后一维，后续广播过程容易出现形状错误，或者产生不符合预期的广播行为。

> [!note] Transformer 中的 Norm 位置
> Transformer 中 Norm 的位置会影响训练稳定性。
> - Post-LN 结构通常写作：
> $$
> x_{l+1} = \mathrm{LN}(x_l + F(x_l))
> $$
>
> - Pre-LN 结构通常写作：
> $$
> x_{l+1} = x_l + F(\mathrm{LN}(x_l))
> $$
>
> Post-LN 将归一化放在残差相加之后，形式上更接近原始 Transformer 的常见写法。Pre-LN 将归一化放在子层计算之前，使 residual path 更直接，有利于深层模型中的梯度传播。
>
> 现代 LLM 更常采用 Pre-LN 或 Pre-RMSNorm 结构。

## RMS Normalization

RMS Normalization 可以看作 Layer Normalization 的进一步简化。LayerNorm 会对 hidden vector 同时进行均值中心化和尺度归一化，而 RMSNorm 只保留尺度归一化部分。它不再减去均值，只根据向量的均方根大小调整整体尺度。

对于一个特征向量 $x \in \mathbb{R}^{C}$，RMSNorm 首先计算它的均方根：

$$
\mathrm{RMS}(x) = \sqrt{\frac{1}{C}\sum_{i=1}^{C}x_i^2 + \epsilon}
$$

然后使用该数值对输入向量进行归一化：

$$
\hat{x} = \frac{x}{\mathrm{RMS}(x)}
$$

最后通过可学习参数进行缩放：

$$
y = \gamma \hat{x}
$$

其中 $\gamma \in \mathbb{R}^{C}$ 是逐特征维度的可学习缩放参数。与 LayerNorm 常见形式相比，RMSNorm 通常不使用偏置项 $\beta$。它关注的是控制 hidden vector 的整体尺度，而非调整其均值位置。

> [!note] RMSNorm 与 LayerNorm 的核心差异
> LayerNorm 的标准化形式可以写成：
> $$
> \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
> $$
>
> RMSNorm 的标准化形式可以写成：
> $$
> \hat{x} = \frac{x}{\sqrt{\frac{1}{C}\sum_{i=1}^{C}x_i^2 + \epsilon}}
> $$
>
> 二者都用于控制 hidden vector 的数值尺度。区别在于，LayerNorm 同时执行均值中心化和尺度归一化；RMSNorm 只执行尺度归一化。因此，RMSNorm 的计算路径更短，也更适合需要高频执行 normalization 的大模型场景。

> [!note] 为什么可以去掉均值中心化
> RMSNorm 的设计基于一个观察：在许多深层模型中，控制激活向量的尺度往往比强制其均值为零更加关键。LayerNorm 中的均值中心化会改变向量的整体偏移，而 RMSNorm 保留了这一偏移信息，只对向量长度进行约束。
>
> 因此，RMSNorm 可以被理解为一种更轻量的尺度控制机制。它减少了均值计算和中心化操作，在保持训练稳定性的同时降低了一部分计算开销。

> [!note] 提出背景
> RMSNorm，全称 Root Mean Square Layer Normalization，由 Biao Zhang 和 Rico Sennrich 在 2019 年提出，论文标题为 *Root Mean Square Layer Normalization*。

> [!tip] 混合精度中的计算
> 在 LLM 训练和推理中，输入 hidden state 可能是 `float16` 或 `bfloat16`。为了提高数值稳定性，RMSNorm 通常会先将输入转成 `float32` 计算 RMS，再将输出转回原始 dtype。现代基本所有的 normalization 操作都会进行同样的处理。

> [!note]- 为什么 RMSNorm 使用 RMS，而不是标准差？
> RMSNorm 的关键选择是：保留输入向量本身，只对它的整体尺度进行约束。对于一个 hidden vector $x \in \mathbb{R}^{H}$，LayerNorm 和 RMSNorm 使用了不同的尺度定义。
>
> LayerNorm 使用中心化后的波动尺度：
>
> $$
> \mathrm{Std}_{H}(x) = \sqrt{D_H[x] + \epsilon}
> $$
>
> $$
> \hat{x} = \frac{x - E_H[x]}{\mathrm{Std}_{H}(x)}
> $$
>
> RMSNorm 使用未中心化的整体能量尺度：
>
> $$
> \mathrm{RMS}_{H}(x) = \sqrt{E_H[x^2] + \epsilon}
> $$
>
> $$
> \hat{x} = \frac{x}{\mathrm{RMS}_{H}(x)}
> $$
>
> 二者的关系可以由二阶矩分解给出：
>
> $$
> E_H[x^2] = D_H[x] + E_H[x]^2
> $$
>
> 这里的 $D_H[x]$ 表示 hidden 维度上的方差，$E_H[x]^2$ 表示均值项的平方。因此，标准差只描述向量围绕自身均值的波动幅度，RMS 同时包含波动项和均值偏移项。
>
> 这个差异解释了 RMSNorm 为什么使用 RMS。LayerNorm 在分子中已经减去了均值，因此使用标准差作为分母是匹配的；RMSNorm 在分子中保留了原始 $x$，均值偏移仍然属于向量整体能量的一部分，因此使用包含均值项的 RMS 更合适。
>
> 如果在 RMSNorm 中直接使用标准差作为分母，分母只反映中心化后的波动幅度，无法反映原始向量的整体幅度。当 hidden vector 各维度存在共同偏移时，标准差可能很小，但原始向量本身的数值仍然很大，此时直接计算 $x / \mathrm{Std}_{H}(x)$ 会导致输出被异常放大。
>
> 例如：
>
> $$
> x = [10.0,\ 10.1,\ 9.9]
> $$
>
> 这个向量的均值接近 $10$，标准差很小，但整体幅度接近 $10$。如果使用标准差缩放原始 $x$，输出会被放大到很大的数值；如果使用 RMS，分母会接近 $10$，输出尺度会保持在稳定范围内。
>
> 因此，RMSNorm 的设计可以理解为：放弃均值中心化后，使用 RMS 来度量原始 hidden vector 的整体能量。这样既保留了均值方向的信息，又能控制向量尺度。它保留了 normalization 中对训练稳定性很关键的 re-scaling 作用，同时省去了均值中心化相关的计算。
