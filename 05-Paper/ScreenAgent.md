# ScreenAgent

2024 IJCAI

主要贡献：构建了一个与真实计算机屏幕交互的环境（在人机交互方法的创新）。在这个环境中，Agent 通过观察屏幕，输出鼠标和键盘操作（通过 function call）来操作图形用户界面。

使用 vlm 模型。

论文贡献：
1. 给出一个 RL 环境，允许 VLM 和真实计算机交互。
2. 

为了引导 VLM Agent 与计算机屏幕持续交互，构建了一个 plan-exec-reflect 循环（类似 ReAct)。
1. *输入 Task+Screen，输出 plan list*：输入任务，划分子任务。(这里的子任务不是具体到操作，而是先做什么，再做什么)。
2. 执行每个子任务（具体到每次点击，涉及到了 function call)。
	1. Acting(*输入 subtask+screen，输出 action list*)：模型根据当前子任务输出操作，计算机执行操作。
	2. Reflecting(*输入 subtask+screen，输出 need retry|continue|need replan*)：模型根据当前子任务和执行操作之后的状态，判断是否执行成功。
