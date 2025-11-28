---
tags:
  - ICCV
---

ICCV-2025

SpatialCrafter: Unleashing the Imagination of Video Diffusion Models  for Scene Reconstruction from Limited Observations

本文着重于：从稀疏 (甚至单视角) 观测图片中实现高质量的 3D 场景重建，并合成逼真的新视角。

![[SpatialCrafter.png]]

# Key Contributions

- We introduce a framework that effectively utilizes the physical-world knowledge embedded in video diffusion models to provide additional plausible observations for sparse-view scene reconstruction, thus reducing the ambiguity of sparse view scene reconstruction. 
- To address the scale ambiguity problem that occurs in joint training across datasets, we develop a unified scale estimation approach for trajectory calibration. This solves the performance degradation problem, thus enabling effective multi-dataset training. 
- We combine monocular depth priors with semantic features extracted from the video latent space, and directly regress 3D Gaussian primitives through a feed-forward manner. Meanwhile, we propose a hybrid architecture integrating Mamba blocks with Transformer blocks to efficiently handle long-sequence feature interactions.
- 我们提出一个框架，有效利用视频扩散模型中蕴含的物理世界先验知识，为稀疏视角场景重建提供额外的合理观测，从而显著降低重建的歧义性。
- 为解决跨数据集联合训练中的尺度模糊问题，我们设计了一种统一的尺度估计方法用于轨迹校准，有效缓解了性能下降问题，使得多数据集联合训练成为可能。
- 我们将单目深度先验与从视频隐空间提取的语义特征相结合，以端到端前馈方式直接回归出 3D 高斯图元；同时，我们提出一种融合 Mamba 模块与 Transformer 模块的混合架构，高效处理长序列特征交互。

# Method

![[temp-01.png]]

## Camera-Conditioned Video Generation

### Scale Alignment

目标：**把无尺度（或任意尺度）的估计结果，对齐到一个有真实物理单位的尺度上。**

方法：使用 VGGT 估计初始相机参数 (外参矩阵，包括平移和旋转向量) 和每个视频帧的深度图 $d_{v}$。利用 Metric3D 为同一帧生成一个度量深度图 (metric depth map) $d_{m}$，作为真实尺度的参考。

为了将 $d_{v}$ 对齐到 $d_{m}$ 上，使用一个全局缩放因子 $s$ 进行对齐。

- $s$ 定义为 IPR(Inter-Percentile Range) 的比值。具体来说就是两个分位数的差值：

$$
\begin{align}
IPR_{0.8,0.2}(d) &= \text{percentile}(d, 0.8) - \text{percentile}(d, 0.2) \\
s &= IPR_{0.8,0.2}(d_{v}) / IPR_{0.8, 0.2}(d_{m})
\end{align}
$$

并用这个 $s$ 来调整相机外参：$P_{scaled}=[R|s\cdot T]$ (只调整平移，因为选择与尺度无关)

## Camera Injection

目标：**将相机姿态（内参 K、外参 $[R∣T]$）作为条件注入视频扩散模型，以实现视角可控的生成，同时避免全模型微调。**

作者使用 ray embeddings 和 depth warping frames 来表示相机信息 (用于控制扩散模型)。

- ray embeddings：对图像中每个像素对应的 3D 射线，用其 **Plücker 坐标** $(o \times d, d)\in \mathbb{R}^6$ 表示。
- depth warping frame：利用预训练的深度估计模型 (Moge) 将参考转换为点云，在目标视角下，通过相机参数重新投影点云，作为从新视角看到的场景几何先验。

最后，利用一个可以训练的 Encoder 模型将相机信息注入到视频扩散模型中，而不是微调整个模型。损失函数是：

$$
\mathcal{L} = \mathbb{E}_{z,z_{0},\epsilon,C,t}\left[ \lVert \epsilon-\epsilon_{\theta}(z_{t};z_{0},t,\phi(C)) \rVert _{2}^2  \right] 
$$

其中 $\phi(C)$ 是相机控制编码，$z_{0}$ 表示参考帧的隐空间表示，$t$ 表示时间步。

### Epipolar Feature Aggregation

目标：**解决扩散模型 3D 视频生成的一致性问题**

方法：对于第 $i$ 帧的像素 $p=(u, v)$，计算其与第 $k$ 帧的关联：

$$
l_{ik}(p)=F_{ik} \cdot \tilde{p}
$$

其中 $\tilde{p}$ 是齐次坐标 $(u, v, 1)^{T}$，$F_{ik}$ 是基础矩阵 (fundamental matrix)

$$
F_{ik}=K_{k}^{-T}E_{ik}K_{i}^{-1}
$$

其中 $K_{i}, K_{k} \in \mathbb{R}^{3\times 3}$ 是相机内参，$E_{ik}$ 是本质矩阵 (essential matrix)。

之后计算每一个像素的距离，并通过一个阈值转换为注意力掩码，以此限制几何对有效区域的关注。

### Sparse-View Setting

目标：**适应稀疏视角问题**

为了更好的适应稀疏视角问题，作者将问题建模为了**插帧问题**(在仅给定**起始帧和结束帧**的情况下，生成中间视角的连贯视频。)。为了最大化预训练模型的先验，作者结合了第一帧和最后一帧的隐空间表征，然后将两帧提取的 CLIP 嵌入向量进行拼接，进行交叉注意力特征注入。

具体来说就是使用了下面的损失函数：

$$
\mathcal{L}=\mathbb{E}_{z,z_{0},z_{n},\epsilon,C,t}\left[ \lVert \epsilon-\epsilon_{\theta}(z_{t};z_{0},z_{n},t,\phi(C)) \rVert ^2_{2} \right] 
$$

其中 $z_{0}$ 和 $z_{n}$ 分别是第一帧和最后一帧的隐空间嵌入表示。在 infer 阶段，使用 VGGT 来直接得到相机的内参和外参。

总得来说，扩散模型的 UNet 输入变为了：

$$
\epsilon_{\theta}(z_{t}; \underbrace{[z_{0},z_{n}]}_{\text{latent prior}}, \underbrace{t}_{\text{time}}, \underbrace{\phi(C)}_{\text{camera injection}})
$$

### 总结

该部分的目标是从输入视频（如图中 Input View 所示，黄线为相机轨迹）中生成任意新视角的视频帧，支持稠密或稀疏视角输入。

![[temp-01.png]]

Cam-Conditioned Generation 分为两大分支：

1. **Cam-Conditioned Generation**
    - 对于每个目标生成视角（Target View），使用 VGGT 估计其相机参数（内参、外参）。
    - 构造 **Ray Maps** 或 **Warped Frames** 作为几何表示。
    - 通过一个**可学习的 Condition Encoder** 将其编码为 $\phi(C)$，注入扩散模型，实现视角可控生成。
2. **Sparse-View Setting**
    - 当输入仅为少数帧（如首尾帧）时，系统将其作为参考帧（Ref. View）。
    - 使用 **VAE Encoder** 提取其 latent 表示 $z_{0},z_{n}$​​，拼接后注入 cross-attention 层中。
3. **Epipolar Feature Aggregation**
    - 引入 epipolar 特征约束，计算任意两帧间像素的极线。
    - 构建**注意力掩码**，限制跨帧 attention 只能在几何有效区域内进行，提升 3D 一致性。
4. **Video Diffusion Model**
    - 接收上述所有条件（几何、latent、语义、mask）。
    - 通过多步去噪生成中间帧的 latent。
    - 经 VAE Decoder 解码为真实图像，输出新视角视频。

## Video-based Scene Reconstruction

### Latent Feature Fusion

目标：**由于从生成图片中重建比较困难，因此设计了信息更多的输入**

作者将给定的视频隐空间表示 $z\in \mathbb{R}^{T\times H\times W\times C}$ 和相机位姿的 ray embeddings 编码 $p \in \mathbb{R}^{T\times H\times W\times 6}$，将其进行序列化 (patch)。

- 将视频隐空间表示实行**空间 patch 化**，转换为 $z_{t}\in \mathbb{R}^{N\times d_{z}}$
- 将 ray embeddings 实行 3D-patch 化，转换为 $p_{t}\in \mathbb{R}^{N\times d_{p}}$
- 为了结合更加确切的几何信息，作者还利用单目深度估计模型，生成了深度图像 $D\in \mathbb{R}^{T\times H\times W\times 1}$，并通过一个 depth encoder 将其转换为 $d_{t}\in \mathbb{R}^{N\times d_{d}}$。

最后，将三种 token 拼在一起，得到最终的输入 $x=[z_{t}; p_{t}; d_{t}]\in \mathbb{R}^{N\times(d_{z}+d_{p}+d_{d})}$，然后通过一层线性层投影为维度更低的 $x'\in \mathbb{R}^{N\times d}$。

在模型方面，作者使用了 Mamba，双向扫描的 Mamba 块。作者认为 Mamba 在处理密集重建任务中，由于时间复杂度 $O(L)$ 优于 transformer 模型的 $O(L^{2})$，且提供了接近的性能，因此更加优秀。

### Gaussian Decoding

目标：**将特征转换为每个像素的高斯参数**

作者设计了一个轻量化的模型，由 3D-DeConv 层组成，将输入转换为高斯特征图 $G\in \mathbb{R}^{(T\times H\times W)\times 12}$，其中 12 个通道是

- RGB 颜色, 3
- 每个通道的缩放因子, 3
- 旋转四元数, 4
- 透明度, 1
- 沿射线的距离, 1

最后的输出是所有图像合并后的 3D Gaussians

### Training Objective

训练过程中，作者通过有监督的方法，利用已知的视角和渲染的图片训练模型。损失函数由三个部分组成：

$$
\mathcal{L}_{recon}=\lambda_{1}\mathcal{L}_{mse}+\lambda_{2}\mathcal{L}_{perc}+\lambda_{3}\mathcal{L}_{depth}
$$

其中 $\mathcal{L}_{mse}$ 表示像素级的平方误差，$\mathcal{L}_{perc}$ 表示透视损失，$\mathcal{L}_{depth}$ 保证深度一致性。

## Experiment

数据方面，作者融合了三个数据集：RealEstate-10K, ACID, DL3DV-10K，总计 (67,477 + 11,075 + 10,510) 个训练数据和 (7,289 + 1,972) 条测试数据。

消融实验表明，每个模块都对最终结果起到了促进作用。

# Future Work

加入动态场景的重建。