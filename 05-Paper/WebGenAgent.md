# WebGenAgent

arxiv, 2025.11

这篇主要关于 Web 生成，针对目前网页生成缺乏视觉反馈的问题。简单来说就是用 VLM 指导 Coding Agent 进行网页生成。

主要贡献如下：
1. 提出了 WebGen-Agent，使用屏幕截图和一个 GUI 测试 Agent 来提供反馈，来指导生成网页。
2. 提供了 Step-GRPO 方法，利用截图和 GUI Agent 分数来指导 GRPO 的训练。

