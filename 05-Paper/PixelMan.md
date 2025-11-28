AAAI-2025

PixelMan: Consistent Object Editing with Diffusion Models via Pixel Manipulation and Generation

本文着重于：提出了 PixelMan，一种无需反演 (inversion-free) 和训练 (training-free) 的方法。通过像素控制和生成，在 16 步内实现与现有预训练 Text2Img 扩散模型一致的物品编辑。

> [!note] inversion-free
> 传统基于扩散模型的图像编辑方法 (论文中提到的 DDIM) 通常需要先将图像反演 (inversion) 到隐空间的噪声轨迹，而这个过程计算开销大，步骤多。
> 
> 本文提出的方法绕过了 Inversion 步骤，提升效率。

# Key Contribution

- We propose to perform pixel manipulation for achieving consistent object editing, by creating a pixel-manipulated image where we copy the source object to the target location in the pixel space. At each step, we always anchor the target latents to the pixel-manipulated latents, which reproduces the object and background with high image consistency, while only focusing on generating the missing “delta” between the pixel-manipulated image and the target image to be generated. 
- We design an efficient three-branched inversion-free sampling approach, which finds the delta editing direction to be added on top of the anchor, i.e., the latents of the pixel-manipulated image, by computing the difference between the predicted latents of the target image and pixel-manipulated image in each step. This process also facilitates faster editing by reducing the required number of inference steps and number of Network Function Evaluations (NFEs).
- To inpaint the manipulated object’s source location, we identify a root cause of many incomplete or incoherent inpainting cases in practice, which is attributed to information leakage from similar objects through the SelfAttention (SA) mechanism. To address this issue, we propose a leak-proof self-attention technique to prevent attention to source, target, and similar objects in the image to mitigate leakage and enable cohesive inpainting. 
- Our method harmonizes the edited object with the target context, by leveraging editing guidance with latents optimization, and by using a source branch to preserve uncontaminated source K, V features as the context for generating appropriate harmonization effects (e.g. lighting, shadow, and edge blending) at the target location.
- 我们提出通过**像素级操作**（pixel manipulation）来实现一致的对象编辑：具体而言，我们构建一张像素操作图像，在其中将源对象直接复制到目标位置。在每一步去噪过程中，我们将目标潜在表示（target latents）始终锚定于该像素操作图像对应的潜在表示，从而高保真地复现对象与背景；同时，模型只需专注于生成像素操作图像与最终目标图像之间的缺失“增量”（delta）。
- 我们设计了一种**高效、无需反演**（inversion-free）的三分支采样框架。该方法在每一步通过计算目标图像与像素操作图像的预测潜在表示之差，显式地估计出需叠加在锚点（即像素操作图像的潜在表示）之上的增量编辑方向。这一机制不仅提升了编辑质量，还显著减少了所需的推理步数和网络函数调用次数（NFEs），从而加速编辑过程。
- 为修复被移动对象留下的源区域，我们识别出现有方法中导致**修复不完整或不连贯**的一个根本原因：自注意力（Self-Attention, SA）机制会从图像中的相似对象处泄露信息。为此，我们提出一种**防泄漏自注意力**（leak-proof self-attention）技术，通过阻止空洞区域对源对象、目标对象及其他相似对象的关注，有效抑制信息泄露，实现连贯、干净的修复。
- 我们的方法能够将编辑后的对象与目标上下文自然融合：一方面，通过**基于潜在表示优化的编辑引导**（editing guidance with latents optimization）施加语义约束；另一方面，引入一个**源分支**（source branch），从中提取未受干扰的源对象 Key 和 Value 特征，并将其作为上下文注入目标分支，以生成恰当的协调效果（例如光照、阴影和边缘融合）。

# Method

![[PixelMan.png]]

![[PixelMan-1.png]]

## Three-Branched Inversion-Free Sampling

目标是在高效率前提下同时实现：

1. 物体与背景内容的一致性保留
2. 物体与背景的视觉和谐
3. 对物体移除后留下的空洞进行连贯的修复。

输入预处理：

1. 三个独立分支：*source branch, pixel-manipulated branch, target branch*，每个分支在流程中独立 (各自维护不同的噪声隐变量)，但是使用同一个网络 (参数相同)。
2. 初始输入是一个图像 $I_{src}$ 和一个称为 Pixel-Manipulated Image 的图像 $I_{man}$
	- $I_{man}$ 是一个人工构造的“像素级操作图像”
		- 在**物体移动/缩放**任务中，从 $I_{src}$ 中裁剪物体，经过插值后粘贴到新位置，覆盖原图得到 $I_{man}$
		- 在物体粘贴任务中，从参考图像 $I_{ref}$ 中获取物体，粘贴到 $I_{src}$ 的目标位置，形成 $I_{man}$
3. 利用 VAE 将它们转换为隐空间编码 $z_{0}^{man}$ 和 $z_{0}^{src}$

### Pixel-manipulated latents as anchor

由于 $z_{0}^{man}$​ 已通过像素操作将物体置于目标位置并保留原始背景，可作为编辑结果的“一致性锚点”。然而，该表示缺乏真实世界的视觉协调性（如光照、阴影、空洞修复等），因此需引入一个校正项 $\Delta z$，使得最终输出为 $z_{0}^{out}=z_{0}^{man} + \Delta z$，其中 $\Delta z$ 专门用于实现 harmonization 与 inpainting。

### Obtaining delta edit direction

整个算法的核心是计算 $\Delta z$ 的值。这部分，需要先忽略细节。

总的来说，作者使用了三个分支来合作计算 $\Delta z$。对于每一个时间步 $t$ 来说：

1. 生成噪声 $\epsilon \in \mathcal{N}(0, I)$。
2. **pixel-manipulated branch**：对 $z_{0}^{man}$ 进行 FDP 操作 (根据时间步 $t$ 加噪声)，得到 $z_{t}^{man}$。利用 UNet 预测噪声，执行去噪操作 RGP 得到 $\hat{z}_{0}^{man}$。
3. **source branch**：细节略过，但是这一步可以得到 $K_{src}$ 和 $V_{src}$
4. **target branch**：与 **pixel-manipulated branch**类似，但是
	- 在采样开始时 $(t=T)$ 时，使用 $z_{0}^{man}$ 作为 $z_{0}^{tgt}$ 的初始估计，也就是说使用 $z_{0}^{man}$ 初始化 $z_{0}^{tgt}$。
	- 在之后的步骤中，使用上一步得到的 $z_{0}^{out}$ 作为 $z_{0}^{tgt}$ 的估计，用于生成 $z_{t}^{tgt}$。
	- 使用 FDP 操作，得到 $z_{t}^{tgt}$
	- 通过 **latent optimization** 操作，对 $z_{t}^{tgt}$ 进行更新
	- 通过 UNet + $K_{src}, V_{src}$ 预测噪声，通过去噪操作 RGP 得到 $\hat{z}_{0}^{tgt}$
5. 计算 $z_{0}^{out} = z_{0}^{man} + \overbrace{(\hat{z}_{0}^{tgt} - \hat{z}_{0}^{man})}^{\Delta z} \times M$, 其中 $M$ 是 Mask，防止 $\Delta z$ 对目标区域的影响。在去噪的最后基本，不使用 $M$，允许 $\Delta z$ 对细节进行修改。

> [!note] FDP
> $$
> z_{t}=\sqrt{ \bar{\alpha}_{t} }\times z_{0} + \sqrt{ 1-\bar{\alpha}_{t} }\times\epsilon
> $$

### Feature-preserving source branch

该步骤描述了 source branch 的细节内容。

1. 生成噪声，对 $z_{0}^{src}$ 进行 FDP 操作。
2. 使用 UNet 预测噪声，但是与之前不同的是，这里不需要预测的噪声结果，而是将 UNet 网络中的 self-attention 得到的 $K_{src}$ 和 $V_{src}$ 注入 (直接替换) target branch 的 UNet 中。

作者认为，这里的 $K_{src},V_{src}$ 保存了原始图像的细节，包括光照、阴影等，有利于 inpainting 和 harmonization。

### Leak-Proof Self-Attention

为了防止 Self-Attention 中对于 Mask 的区域的信息泄露 (在 Target branch 中对于目标区域进行了 Mask) ，作者将 Self-Attention 注意力矩阵中关于 to-be-edited 的区域置为了最小值 (-inf)。

### Editing Guidance with Latents Optimization

作者采用一种高效的 inference-time 优化策略：在每个时间步 $t$，直接对 target branch 的 noisy latent $z_{t}^{tgt}$​ 应用梯度下降，梯度来自一个预定义的能量函数 $\mathcal{E}(z_{t}^{tgt}, z_{t}^{man})$。该能量函数借鉴自 DragonDiffusion，用于约束 inpainting 区域的连贯性与物体 - 背景的和谐性，通常基于预训练视觉模型（如 DINO/CLIP）计算特征对齐损失。

与传统 Energy Guidance（EG）方法不同，PixelMan **不修改预测噪声**$\epsilon$，而是**直接优化​**$z_{t}$，从而避免了 EG 所需的“time travel”（即重复 DDIM inversion），显著降低计算开销（NFE）。该策略在理论上等价于推理阶段的梯度下降（如 GSN），兼具效率与效果。

# Experiment

从结果上来看，PixelMan 取得了非常好的效果。

![[PixelMan-2.png]]

可以在较少的步骤的情况下得到较好的结果。

# Futrue Work

作者没有提到 Future Work。