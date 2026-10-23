---
aliases:
  - 偏好学习
tags:
  - 深度学习
---

# DPO

[[RLHF]] (Reinforcement Learning from Human Feedback) 通过人类对模型回答的偏好，引导语言模型生成更符合人类期望的内容。经典的 RLHF 流程首先利用人类偏好数据训练一个**奖励模型** (Reward Model)，然后将该奖励模型作为优化目标，使用 PPO 等强化学习算法调整语言模型的生成策略。

具体而言，给定偏好数据集，
$$
\mathcal{D}=\left\{ (x, y_{w},y_{l}) \right\} 
$$
其中 $y_{w}\succ y_{l}$ 表示人类偏好 $y_{w}$，RLHF 首先学习一个能够反映人类偏好的奖励函数 $r_{\phi}(x,y)$，然后通过强化学习优化语言模型的生成策略:
$$
\max_{\pi}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi(\cdot\mid x)}\left[ r_{\phi}(x,y) \right] -\beta \mathbb{E}_{x\sim \mathcal{D}}\left[ D_{\text{KL}}(\pi(\cdot\mid x)\Vert \pi_{\text{ref}}(\cdot\mid x)) \right] 
$$
