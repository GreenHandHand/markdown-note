# 训练一个 GPT2

本文记录训练一个性能接近 GPT2 (112M) 的网络中涉及到的一个内容。这里不包含 tokenizer 和数据集处理的问题，直接使用了 GPT2 的 tokenizer 和 fineweb-edu 数据集。

> [!cite]
> 参考内容：[karpathy/build-nanogpt: Video+code lecture on building nanoGPT from scratch (github.com)](https://github.com/karpathy/build-nanogpt?tab=readme-ov-file)

## GPT2 模型结构

本次实验首次编写的内容是 GPT2 的网络结构。我们可以使用 Huggingface 加载 GPT2 模型，并输出查看其结构。
```python
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')
display(model)
```
可以得到类似下面的输出内容：
```python
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2SdpaAttention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
```
从上面的内容可以看出，GPT2 的网络结构如下：
1. 词嵌入层 (`wte`, word token embedding)
2. 位置编码 (`wpe`, word position embedding)
3. Dropout 层
4. 12 层 transformer 的 Decoder
5. 层归一化 (`ln_f`)
6. 输出层

为了可以直接读取预训练的 GPT2 的网络参数，我们在实现该网络结构的时候，最好可以按照其命名规则来实现。

### 整体结构

首先实现其整体的结构，我们按照前面的分析，依次创建即可。
```python
class GPT2(nn.Module):
    """GPT2 Model Scratch"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # word token embedding
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # word position embedding
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

> [!note]
> 1. 这里创建了一个 `Dataclass`，用于保存一些 GPT2 模型的配置内容。这样的方式在后续的诸多步骤中可以简化很多的代码。
> 2. 在第 8 行中，我们使用了 pytorch 中的 `ModuleDict` 模块，该模块允许我们创建一个按名字索引的模块字典。这为管理大型模型提供了更多的灵活性。
> 3. 在构建 12 层的 transformer 时，我们使用了 `nn.ModuleList` 来构建 12 层的 GPT2Block，用于堆叠 Transformer 结构。`ModuleList` 的使用确保了层之间的顺序排列，是构建深度网络的重要工具。

### Transformer 模块

我们将每层 transformer 单独使用 `GPTBlock` 实现。其代码如下：
```python
class GPT2Block(nn.Module):
    """Decoder Block"""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

GPT2Block 实现了 Transformer Decoder 中的一个完整的层。它包含两个主要部分：自注意力层和前馈网络层，并通过残差连接与 LayerNorm 实现信息的流动。这种设计允许每层保留输入特征，同时通过注意力机制和前馈网络不断优化输出。

Transformer 中最重要多头注意力我们单独使用 `CausalSelfAttention` 实现。

```python
class CausalSelfAttention(nn.Module):
    """Causal Self Attention Block"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), "n_embd should be divisible by n_head"

        # key, query, value projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch_size, seq_len, n_embd
        batch_size, seq_len, n_embd = x.size()

        # qkv projection
        qkv: torch.Tensor = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        k = k.view(batch_size, seq_len, self.n_head, n_embd // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(batch_size, seq_len, self.n_head, n_embd // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(batch_size, seq_len, self.n_head, n_embd // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

		# flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        y = self.c_proj(y)
        return y
```

> [!note]
> 1. PyTorch 官方 GPT2 实现使用了一维卷积 (Conv1D) 来实现 Q、K、V 投影，这与我们采用的全连接层 (Linear) 实现方式不同。Conv1D 实质上是一个带有滑动窗口的线性层，我们这里改为全连接层，简化实现，并更符合传统的 Transformer 实现思路。唯一的区别在于参数组织形式，读取参数时需进行转置。
> 2. 实现中使用的 Flash Attention 来实现 dot product 操作 (即第 37 行中的 `scaled_dot_product_attention`)，该操作是内存友好的，并且速度也更快。其中的 `is_causal` 参数表示需要因果遮掩 (causal mask)，防止模型在生成任务中看到未来的 token。

> [!tip] Flash Attention^[ https://arxiv.org/abs/2205.14135 ]
> Flash Attention 是一种优化方案，它通过降低对显存的需求来提升内存利用率，同时加快了计算速度。与传统的注意力机制相比，Flash Attention 在处理长序列时能够显著减少内存占用，尤其适合大规模的自回归模型训练。

最后是 MLP 层的实现：
```python
class MLP(nn.Module):
    """GPT MLP module"""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.c_fc(x))
        x = self.c_proj(x)
        return x
```

> [!note]
> 全连接层中主要需要提及的是激活函数。在 GPT2 中，我们使用 GELU 激活函数^[[GELU — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)]，该函数是 RELU 函数的变体，其在 $x<0$ 时仍然由非常微弱的梯度信息，可以一定程序的防止 RELU 在深度网络中遇到的神经元死亡的情况。
>
> 在 GPT2 的原始实现中，GELU 激活函数被采用。然而，由于 PyTorch 早期版本没有直接支持标准的 GELU，而是使用了一种基于 `tanh` 的近似计算，这种近似在计算上更高效，且与标准 GELU 的行为非常接近。这个近似形式在大规模模型训练中帮助降低了计算复杂度，但实际上对模型性能的影响极小。因此，即使在 PyTorch 后来的版本支持了标准 GELU，一些实现（如 GPT2 的早期版本）仍然保留了 `tanh` 近似计算。这是一个历史遗留问题，虽然不影响结果，但表明了深度学习框架中优化选择的演变。

### 前向传播

最后，我们将在 GPT2 模块中实现前向传播函数。这实际上就是读取输入，添加位置编码，并传入网络的不同模块，得到前向传播函数：
```python
def forward(self, idx: torch.Tensor) -> torch.Tensor:
	batch_size, seq_len = idx.size()
	assert (
		seq_len <= self.config.block_size
	), f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"

	# forward the token and position embeddings
	pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
	# position embeddings of shape (T, n_embd)
	pos_emb = self.transformer.wpe(pos)
	# token embeddings of shape (batch_size, seq_len, n_embd)
	tok_emb = self.transformer.wte(idx)

	# broadcast align dim 0
	x = tok_emb + pos_emb  # (batch_size, seq_len, n_embd)

	# forward the blocks of the transformer
	for block in self.transformer.h:
		x = block(x)

	x = self.transformer.ln_f(x)
	logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

	return logits
```

> [!tip]
> 由于我们是基于已有的模型进行复现尝试，因此直接使用我们构建的模型去读取预训练的参数是一种非常有效的验证方法。这里我们可以尝试读取不同版本的 GPT2 模型，并检查输出内容。

### 参数初始化

良好的参数初始化应当让模型平等的预测出每个 token。这样设置的模型可以在训练的初期快速找到正确的方向，而不是花费大量的时间修改由于不平均的初始化导致的偏见。一种较为有效的初始化方法为 [[00_Inbox/深度学习/多层感知机#参数初始化|Xavier]]，这种初始化方法根据模型的参数数量进行初始化，使得模型的参数分布尽可能的平均。在 GPT2 的原始论文中，使用下面的初始化策略：
- 线性层：
	- 权重 w 初始化为均值为 0，方差为 0.02 的正态分布。
	- 偏执 b 初始化为 0。
- 特别的，嵌入层初始化为均值为 0，方差为 0.01 的正态分布。
- 对于残差流中的线性层，初始化时方差需要额外除以残差叠加次数的平方根。
```python
class GPT2(nn.Module):
    """GPT2 Model Scratch"""

    def __init__(self, config: GPTConfig):
		# same as befor
		...

        # init the weight
        self.apply(self._init_weight)

    def _init_weight(self, module: nn.Module):
        """Init the weight."""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"): 
                std /= (2 * self.config.n_layer) ** 0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

> [!note] 残差层的初始化
> 在 GPT2 中，残差层的初始化不同于其他的线性层，由于残差流会使得所有的残差层的方差叠加，假设我们使用了 $n$ 层残差叠加，如果我们原来初始化为 $\mathbf{w}\sim\mathcal N(\mu,\sigma^{2})$，那么经过了 $n$ 次叠加，得到的最后的输出将会是 $n$ 个残差层的叠加，即
> $$
\sum\limits_{i=1}^{n}\mathbf{w}_{i}\sim \mathcal N(n\mu, n\sigma^{2})
> $$
> 为了得到与我们目标一致的分布，我们需要将均值除以 $n$，方差除以 $\sqrt{ n }$。在 GPT2 中，均值为 0，方差为 0.02，因此只需要对方差进行处理。

> [!question]
> 为了实现残差层的不同的初始化方法，在这里，Karpathy 对于所有残差层中的线性层添加了 SCALE_INIT 关键字，然后通过 `hasattr` 方法判断。我没有找到比它更好的实现方法。

> [!tip] 参数的设置
> 这里采用 0.02 作为方差实际上 xavier 方法的近似值。因此我们的初始化结果与使用 xavier 方法类似。

## 在小数据集上进行测试

直接将模型放入大型数据集中进行训练，会存在很多的未知的问题可能会导致训练崩溃。因此，在进行大规模训练之前，我们需要保证模型是正确的。此时，尝试在小数据集上进行训练，并实现一个非常低的损失是一种常用的方式。我们在这里使用莎士比亚数据集进行测试。

### 数据集分析

在进行测试前，我们可以先对这个小的数据集进行一些分析，并明确一些内容。
- 莎士比亚数据集中一共有 111,5394 个字符，按照 GPT2 的 Tokenizer 的压缩比 (1:3) 来说，大概有 30,0000 个 token。
- 莎士比亚数据集很小，而 GPT2 的词表长度为 50257，很显然有大量的 token 没有在数据集中出现，这些词的 embedding 在训练时不会被修改，且在输出中会被优化到一个很小的值。(这一点说明模型的 word embedding 层的大小不是一定要按照词表大小来进行的，实际上我们后续可以通过这个特性进行一些效率优化)

### 构建数据集

在我们的训练中，我们希望模型可以通过前面出现的词，来计算得到下一个词的概率分布。因此我们的输出实际上就是输入进行一个单位的偏移 (shift)。为了实现这样的效果，一种方式是按照下面的方式：
```python
    def next_batch(self):
        buf = self.tokens[
            self.current_position : self.current_position
            + self.batch_size * self.seq_len
            + 1
        ]
        self.current_position += self.batch_size * self.seq_len
        x = buf[:-1].view(self.batch_size, self.seq_len)
        y = buf[1:].view(self.batch_size, self.seq_len)

        # if loading the next batch would be out of bounds
        # reset the current_position from the beginning
        if self.current_position + (self.batch_size * self.seq_len + 1) > len(
            self.tokens
        ):
            self.current_position = 0
        return x, y
```

在上面的例子中，我们先将数据读取到一个长度为 `batch_size * seq_len + 1` 的缓存中，然后施加偏移并创建对应大小的张量。这是一种实现批量读入的较为简便的方式。

> [!tip]
> 更进一步的，一个比较重要的改进方式是在读取数据时添加随机性。这需要更加复杂的数据预处理，但是这样的好处是可以消除一些数据集编排不当引入的一些不必要信息，模型可能会被这些顺序信息误导。可以考虑的进一步改进包括：
> - 随机采样：可以考虑在每个 epoch 重新打乱数据，或者在每次获取 batch 时从数据集中随机抽取一个片段，避免模型对某些序列的过拟合。
> - 数据增强：除了打乱顺序，另一种方式是通过引入数据增强 (data augmentation) 技术，比如在句子中随机替换、插入 token，这有助于进一步增加模型的泛化能力。

### 使用一个批次测试训练

在训练中，我们需要计算损失函数。我们将交叉熵损失的计算加入模型的前向传播中，并与计算结果一并返回。
```python
def forward(
	self, idx: torch.Tensor, targets: torch.Tensor | None = None
) -> torch.Tensor:
	batch_size, seq_len = idx.size()
	assert (
		seq_len <= self.config.block_size
	), f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"

	# forward the token and position embeddings
	pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
	# position embeddings of shape (T, n_embd)
	pos_emb = self.transformer.wpe(pos)
	# token embeddings of shape (batch_size, seq_len, n_embd)
	tok_emb = self.transformer.wte(idx)

	# broadcast align dim 0
	x = tok_emb + pos_emb  # (batch_size, seq_len, n_embd)

	# forward the blocks of the transformer
	for block in self.transformer.h:
		x = block(x)

	x = self.transformer.ln_f(x)
	logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

	loss = None
	if targets is not None:
		loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

	return logits, loss
```

为了确保模型可以被正确训练，我们首先从莎士比亚数据集中提取出一个批次，并在该批次上进行重复训练。我们的目标是让模型在这个小规模数据上获得显著的损失下降，同时确保损失曲线平滑，不出现梯度爆炸或消失现象。一般而言，经过一定轮次的训练，模型的损失应该可以稳定降至较低的水平（如小于 0.1）。在这部分中，我们使用一个较大的学习率，例如 `3e-4`，这是 karpathy^[仓库作者,大牛] 推荐的一个测试学习率。这里我们基于下面的代码进行测试：
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()
x, y = data_loader.next_batch()
x, y = x.to(device), y.to(device)
for i in range(MAX_STEPS):
	optimizer.zero_grad()
	logits, loss = model(x, y)
	loss.backward()
	optimizer.step()
	print(f"Step {i}, Loss: {loss.item()}")
```

> [!note]
> 1. 这里使用了 `AdamW` 优化器，这是 `Adam` 的改进版本。与 `Adam` 不同，`AdamW` 解决了权重衰减与正则化问题。在 `Adam` 优化器中，权重衰减的实现不完全正确，导致权重更新过程中正则化效果受限，而 `AdamW` 则通过修正这个问题，提升了模型的泛化能力。
> 2. 如果我们的训练流程没有问题，模型在训练若干步后应当能够取得一个非常小的损失。

> [!done]
> 通过 50 轮次的训练，我们的损失下降到了接近 0.01 的水平，表明模型在这个小批次上训练是有效的。这种快速的损失下降通常是因为数据集较小，模型相对简单，且使用了较大的学习率。
> ```
> Step 0, Loss: 10.981101989746094
> Step 1, Loss: 8.416521072387695
> Step 2, Loss: 7.423924446105957
> ...
> Step 48, Loss: 0.013090801425278187
> Step 49, Loss: 0.0119443004950881
> ```

### 编写训练函数

接下来，我们将尝试在莎士比亚数据集上进行训练。下面是一个简单的训练函数例程，但是已经可以进行一次简单的训练。
```python
MAX_STEP = 50
BATCH_SIZE, SEQ_LEN = 2, 1024
train_loader = DataLoaderLite(BATCH_SIZE, SEQ_LEN)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()
for step in range(MAX_STEP):
	ts = time.time()
	optimizer.zero_grad()

	x, y = train_loader.next_batch()
	x = x.to(device)
	y = y.to(device)
	_, loss = model(x, y)
	loss.backward()
	optimizer.step()
	torch.cuda.synchronize()

	t = time.time() - ts
	token_processed = BATCH_SIZE * SEQ_LEN
	tok_per_sec = token_processed / t

	print(
		f"step {step} loss: {loss.item()} | time: {t * 1000:.0f} ms | tok/sec: {tok_per_sec:.2f}"
	)
```

> [!note]
> - 虽然上面的训练过程很简单，但是也有一些值得提到的内容。我们使用小数据集进行测试的目的除了验证模型的正确性、训练过程的正确性外，我们还将统计训练过程的耗时。训练模型的过程很漫长，为了尽可能的减少训练的时间，我们需要在测试阶段尽可能的增加我们训练的吞吐量。
> - 训练过程主要在 GPU 上进行，而 PyTorch 默认采用异步执行机制，特别是在拷贝数据到 GPU 或 GPU 计算时。这意味着 CPU 端的代码不会等待 GPU 计算完毕就继续执行。因此，在统计时间之前，必须使用 `torch.cuda.synchronize()` 来确保 GPU 计算完全结束，再进行时间统计。通常情况下，PyTorch 会在可能导致数据不一致的操作之前隐式调用 `synchronize`，但为了精确测量时间和吞吐量，我们这里需要手动调用。
> - 此外，单纯的使用时间来作为衡量效率的标准是不准确的，时间受到了 `batch_size` 和 `seq_len` 参数的影响。一种更好的方式是使用模型的吞吐量，即计算每秒处理的 token 数目。在后续的优化中，我们将比较这两个指标来评估模型的优化效果。

> [!tip]
> 在调用 `synchronize` 时，我们可以会疑问如果 pytorch 默认不阻塞进行，那么是否会出现数据不一致的情况。实际上，在 pytorch 中的实现中，在进行会导致数据不一致的操作之前会隐式的调用 `synchronize`，因此在平时的训练中这个函数不是必要的。但是在这里我们需要等待 GPU 计算完毕，并统计时间，因此需要阻塞。

> [!done]
> 运行这个测试脚本，可以得到每一步的时间消耗如下 (这是在我的 3060 笔记本上测试得到的结果)：
> ```
> step 0 loss: 10.958434104919434 | time: 1252 ms | tok/sec: 1636.36
> step 1 loss: 8.877669334411621 | time: 1442 ms | tok/sec: 1420.39
> step 2 loss: 8.381708145141602 | time: 1343 ms | tok/sec: 1524.97
> step 3 loss: 7.808971405029297 | time: 1311 ms | tok/sec: 1562.49
> step 4 loss: 7.395624160766602 | time: 1530 ms | tok/sec: 1338.89
> step 5 loss: 6.833993434906006 | time: 1243 ms | tok/sec: 1647.44
> step 6 loss: 6.258546352386475 | time: 1369 ms | tok/sec: 1496.34
> # ...
> ```
> 可以看到，损失在前几步快速下降，这符合预期，表明模型能够从初始数据中快速学习。而每步的时间消耗略有波动，吞吐量在 1300 到 1600 token/sec 之间波动。这种波动可能与 GPU 的资源分配或数据加载的效率有关。在后续优化中，数据加载和硬件的充分利用将是提升性能的关键。

## 优化

接下来是模型训练中非常重要，且非常 trick 的一部分内容。在之前的测试中，我们可以看到，在我的 3060 笔记本上，训练的速度大约是 1500 tok/s，一步训练 1024x2 的输入需要接近 1300ms。我们希望在正式进行训练时，可以增加模型的训练速度。

> [!note] 限制训练速度的因素
> 1. 机器性能：机器性能是训练速度的最主要限制因素，包括 GPU 的计算能力、内存带宽、PCIe 通道的带宽等。这些硬件参数直接决定了模型在每一步的计算速度。虽然我们难以改变硬件条件，但理解这些限制有助于我们制定更实际的优化目标。
> 2. 数据读取：数据读取是另一个重要的瓶颈，尤其当 Python 进行串行数据读取时，会造成显著的延迟，进而影响 GPU 的利用率。
> 3. 计算精度：默认情况下，PyTorch 使用 float32 进行计算，但在实际训练中不需要这么高的精度。通过降低计算精度（如使用 float16），可以大幅提高训练速度并减少显存和内存占用。
> 4. 冗余的操作：模型训练中可能会存在冗余计算或不必要的算子操作。通过分析计算图，可以找到并优化这些冗余操作。例如，可以使用 fused operations（合并操作）减少不必要的张量计算开销，或者对模型的网络结构进行裁剪和简化，尽可能减少不必要的层调用。

> [!tip]
> 这里讨论的优化主要是针对单卡训练的。在 Karpathy 的教程中，他使用了 8 张 A100 GPU 来大幅提升训练速度。但对于大多数学生或个人研究者来说，使用多张 GPU 并行训练的机会较少。我们当前的优化主要针对单张 3060 或 3080 显卡，但这些优化方法同样可以为将来多卡训练打下基础。等有机会使用更多 GPU 资源时，再深入学习和应用多卡训练的相关技术。

### 数据读取优化

在前面的数据读取操作中，我们实现的是一个串行的读取方案，每次训练都必须得到上一次训练之后的数据。这显然是不必要的。为了解决这一问题，我考虑使用 PyTorch 的 DataLoader 中的 num_workers 参数来启用多线程数据预加载。
- PyTorch 中的 DataLoader 通过 num_workers 参数启用多线程数据加载，允许在训练过程中并行读取数据。每个 worker 进程会独立加载数据，这样可以在 GPU 进行计算时，预先为下一个 batch 读取数据，从而避免 GPU 等待 CPU 数据传输的情况。
- 同时，pin_memory=True 可以将数据固定在内存中，加快从 CPU 到 GPU 的数据传输速度。

```python
class TextDataset(Dataset):
    def __init__(self, seq_len):
        with open("shakespeare.txt", "r") as f:
            self.data = f.read()

        enc = tiktoken.get_encoding("gpt2")
        self.tokens = enc.encode(self.data)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        buf = self.tokens[idx : idx + self.seq_len + 1]
        x = torch.tensor(buf[:-1], dtype=torch.long)
        y = torch.tensor(buf[1:], dtype=torch.long)
        return x, y

# usage
train_dataset = TextDataset(SEQ_LEN)
train_loader = DataLoader(
	train_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True,
	num_workers=4,
	pin_memory=True,
)
```

> [!note]
> 通常，设置 num_workers=4 是一个较好的起点，但在不同硬件条件下可能需要进行调整，以实现最佳的 CPU-GPU 数据加载效率。

> [!done]
> 我们使用相同的种子，运行可以得到如下，可以看到，
> - 前 4 个步骤中，由于初始化进程导致效率较低。
> - 在后续的步骤中，优化数据的读取方式提升了大约 25% 的速度。
>
> 在这里，优化效果较为明显，这是因为目前的 GPU 速度相较 CPU 速度差异较大。这种优化的额外好处是但是读取数据可以打乱数据的结构，同时，还可以利用一些数据增强的方式，来避免文本中存在的一些无用的顺序信息、错误的噪声信息等。
> ```
> step 0 loss: 11.011943817138672 | time: 1386 ms | tok/sec: 1477.44
> step 1 loss: 9.094559669494629 | time: 1409 ms | tok/sec: 1453.71
> step 2 loss: 8.421010971069336 | time: 1050 ms | tok/sec: 1949.98
> step 3 loss: 8.119394302368164 | time: 1037 ms | tok/sec: 1975.24
> step 4 loss: 7.748178482055664 | time: 1007 ms | tok/sec: 2033.80
> step 5 loss: 7.2708353996276855 | time: 1019 ms | tok/sec: 2009.52
> step 6 loss: 6.811398029327393 | time: 1018 ms | tok/sec: 2011.76
> step 7 loss: 6.301889896392822 | time: 1018 ms | tok/sec: 2012.00
> step 8 loss: 6.033753871917725 | time: 1019 ms | tok/sec: 2009.01
> step 9 loss: 5.728846073150635 | time: 1019 ms | tok/sec: 2010.44
> # ...
> ```

> [!note] 数据增强
> 数据顺序的随机打乱可以有效减少模型过拟合特定顺序信息的风险，尤其在语言模型中，数据的时间顺序有时会带来无用的偏差。
>
> 此外，虽然数据增强在计算机视觉任务中较为常见，在 NLP 中也可以通过诸如添加噪声、词汇替换、数据采样等方式增强数据多样性。打乱数据顺序不仅优化了模型的泛化能力，也有助于减少特定模式的重复出现，提高模型对未知数据的适应性。

### 模型参数优化

这是 GPT2 模型中的一个细节问题。在 [[#GPT2 模型结构]] 中，词嵌入层 (`wte`) 与最后的输出映射层 (`lm_head`) 的形状是完全相同的。同时，如果我们对其进行仔细检查，会发现他们的参数也是接近的。在 GPT2 的实现中，这两层使用的是同一套参数。

实际上，在 *Attention is all you need* 论文^[参考了论文 [[1608.05859] Using the Output Embedding to Improve Language Models (arxiv.org)]( https://arxiv.org/abs/1608.05859 )] 中，提及了使用权重共享的方法。这种方法可以在大幅度减少数据量的同时，增加模型的表现。

体现在我们的模型中，则需要对 GPT2 的结构进行一点小小的改动。
```python
class GPT2(nn.Module):
    """GPT2 Model Scratch"""

    def __init__(self, config: GPTConfig):
		# same as before

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
```

> [!tip]
> 通过权重共享，我们大约减少了一半的嵌入层和输出层参数量。对于 GPT2 模型的嵌入维度为 `n_embd`，词汇表大小为 `vocab_size` 时，权重共享可以节省 `vocab_size * n_embd` 的参数量。然而，这种减少主要体现在模型的存储和内存消耗上，对计算速度的影响较小，因为词嵌入层和输出层的计算复杂度并未显著降低。模型在前向传播和反向传播过程中仍然需要对嵌入向量和 logits 进行计算，因此实际的计算时间提升有限。

> [!done]
> 我们仅添加了上面的那一行代码，再次运行我们的测试脚本。
>  ```
>  step 0 loss: 10.999788284301758 | time: 1343 ms | tok/sec: 1524.80
> step 1 loss: 9.430315017700195 | time: 942 ms | tok/sec: 2172.95
> step 2 loss: 8.806062698364258 | time: 981 ms | tok/sec: 2088.23
> step 3 loss: 8.477924346923828 | time: 976 ms | tok/sec: 2098.21
> step 4 loss: 8.170611381530762 | time: 981 ms | tok/sec: 2088.03
> step 5 loss: 7.875115394592285 | time: 979 ms | tok/sec: 2092.43
> step 6 loss: 7.5635294914245605 | time: 980 ms | tok/sec: 2089.72
> step 7 loss: 7.283697605133057 | time: 978 ms | tok/sec: 2093.05
> step 8 loss: 7.0387420654296875 | time: 978 ms | tok/sec: 2094.47
> step 9 loss: 6.748325347900391 | time: 978 ms | tok/sec: 2093.09
> step 10 loss: 6.534390449523926 | time: 983 ms | tok/sec: 2083.97
> # ...
>  ```
>  我们观察到每一步的训练时间仅减少了约 20 毫秒，这主要是由于计算仍然占据了较大的时间开销，尽管参数减少带来了一定的内存和存储优化，但计算的主要瓶颈仍然是矩阵乘法和其他核心计算过程。

### 计算精度优化

#### 使用 TF32

Nvidia GPU 提供了多种浮点精度选项，但在训练神经网络时，并不总是需要最高的精度。通过调整计算精度，我们可以在牺牲少量精度的前提下，大幅提升计算速度。

> [!note] 使用 TF32 进行矩阵乘法
> Nvidia 从 A100 开始支持一种新的精度格式——TF32（TensorFloat-32），用于优化矩阵乘法。与传统的 FP32（32 位浮点）相比，TF32 会略微降低尾数的精度，从而使矩阵乘法的效率提升数倍。
> $$
\begin{aligned}
\text{FP32}&\quad\begin{array}{|c|c|cc|}
\hline\small\text{符号位 (1)} & \small\text{阶码 (8)} & \small\text{尾数 (23)} &\quad\quad \quad\\ \hline
\end{array} \\
\text{TP32}&\quad\begin{array}{|c|c|c|}
\hline\small\text{符号位 (1)} & \small\text{阶码 (8)} & \small\text{尾数 (10)}\\ \hline
\end{array}
\end{aligned}
> $$
> TF32 的表示范围与 FP32 相同，但尾数精度较低，意味着它在执行矩阵乘法时可能会牺牲一部分精度。然而，对于深度学习模型，这种精度损失通常不会对训练结果产生显著影响。因此，TF32 是一种“低成本”的加速方式。

上述内容是在 GPU 中的细节，但是实际上 PyTorch 为我们隐藏了这些细节。要启用 TF32，在 PyTorch 中只需要一行代码^[[torch.set_float32_matmul_precision — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html)]：
```python
torch.backends.cuda.matmul.allow_tf32 = True # or
torch.set_float32_matmul_precision('high') # They are almost same
```

> [!done]
> 使用第一种方法得到的结果如下：
> ```
> step 0 loss: 11.06725025177002 | time: 1089 ms | tok/sec: 1880.45
> step 1 loss: 9.476325035095215 | time: 599 ms | tok/sec: 3417.57
> step 2 loss: 8.905616760253906 | time: 545 ms | tok/sec: 3760.07
> step 3 loss: 8.617420196533203 | time: 637 ms | tok/sec: 3213.38
> step 4 loss: 8.311102867126465 | time: 553 ms | tok/sec: 3706.42
> step 5 loss: 7.998371124267578 | time: 534 ms | tok/sec: 3837.84
> step 6 loss: 7.717103004455566 | time: 543 ms | tok/sec: 3770.18
> step 7 loss: 7.419944763183594 | time: 541 ms | tok/sec: 3783.70
> step 8 loss: 7.1566081047058105 | time: 535 ms | tok/sec: 3830.88
> step 9 loss: 6.874164581298828 | time: 543 ms | tok/sec: 3772.41
> step 10 loss: 6.628359317779541 | time: 541 ms | tok/sec: 3787.63
> ```
> 使用第二种方法得到的结果如下：
> ```
> step 0 loss: 10.973276138305664 | time: 847 ms | tok/sec: 2418.39
> step 1 loss: 9.310206413269043 | time: 546 ms | tok/sec: 3753.63
> step 2 loss: 8.74912166595459 | time: 539 ms | tok/sec: 3799.10
> step 3 loss: 8.45030689239502 | time: 558 ms | tok/sec: 3673.22
> step 4 loss: 8.152305603027344 | time: 536 ms | tok/sec: 3824.04
> step 5 loss: 7.856066703796387 | time: 543 ms | tok/sec: 3773.19
> step 6 loss: 7.568639278411865 | time: 548 ms | tok/sec: 3736.37
> ```
> 启用 TF32 后，训练速度有明显提升。在这种情况下，虽然理论上 TF32 可以在 A100 等高端 GPU 上将矩阵乘法加速 8 倍，但在 3060 这样的消费级 GPU 上，我们实际获得了约 76% 的加速效果，这已经是非常显著的提升。

#### 使用 BF16

在提升矩阵乘法效率后，训练速度并未达到理论上的最大提升。这是因为模型前向传播的时间不仅取决于计算，还受到数据传输带宽的限制。即使矩阵乘法加速了，数据传输的瓶颈使得性能提升效果受到影响。为了解决这一问题，混合精度训练是一种非常有效的策略。

> [!note] BF16 混合精度训练
> GPU 的数据传输速率是性能的关键瓶颈之一。FP32（32 位浮点数）虽然提供了更高的精度，但其较大的数据量导致数据传输较慢。通过压缩到 16 位格式，可以有效减少数据传输时间。当前有两种主要的 16 位浮点格式：FP16 和 BF16。
> $$
\begin{aligned}
\text{FP16}&\quad\begin{array}{|c|c|c|}
\hline\small\text{符号位 (1)} & \small\text{阶码 (5)} & \small\text{尾数 (10)}\quad\quad \\ \hline
\end{array} \\
\text{BF16}&\quad\begin{array}{|c|c|c|}
\hline\small\text{符号位 (1)} & \small\text{阶码 (8)}\quad\quad & \small\text{尾数 (7)}\,\,\, \\ \hline
\end{array}
\end{aligned}
> $$
> FP16 在混合精度训练中曾经被广泛使用，但由于其较小的阶码范围，可能导致浮点数溢出的问题，需要使用缩放因子进行调整，容易出现 NaN（非数值）的现象。BF16 的阶码范围与 FP32 一致，尽管尾数精度较低，但能避免溢出问题，适合大部分训练任务。

在 PyTorch 中，可以通过 `autocast` 实现混合精度训练，同样只需要一行，其使用方法如下：
```python
optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
	_, loss = model(x, y)
loss.backward()
optimizer.step()
```
在前向传播中使用 BF16，因为此时不需要太高的精度，而在反向传播中则仍使用 FP32，以确保梯度计算的精确性。这种方式被称为混合精度训练，它可以显著加速训练，同时尽可能保持模型的准确性。

> [!note] BF16 与 TF32 的区别
> - BF16 会显式地改变张量的存储格式，从而优化数据传输效率。由于 16 位格式的数据占用更少的内存，数据在 GPU 之间的传输会变得更快。
> - TF32 主要用于矩阵乘法运算，它在不改变张量存储格式的情况下，隐式地对矩阵乘法的精度进行降低。因此，TF32 优化的是计算速度，而 BF16 优化的是数据传输速度。

> [!done]
> 我们可以看到，使用 BF16 对于模型训练的速度提升是非常可观的。使用 BF16 在我的机器上可以提高 83% 左右的效率。
> ```
> step 0 loss: 11.0672607421875 | time: 818 ms | tok/sec: 2503.75
> step 1 loss: 9.479621887207031 | time: 408 ms | tok/sec: 5014.38
> step 2 loss: 8.903068542480469 | time: 308 ms | tok/sec: 6647.19
> step 3 loss: 8.622482299804688 | time: 303 ms | tok/sec: 6770.20
> step 4 loss: 8.327980041503906 | time: 298 ms | tok/sec: 6867.34
> step 5 loss: 7.99932861328125 | time: 289 ms | tok/sec: 7076.96
> step 6 loss: 7.714702606201172 | time: 310 ms | tok/sec: 6598.75
> step 7 loss: 7.443271636962891 | time: 298 ms | tok/sec: 6861.22
> step 8 loss: 7.170108795166016 | time: 294 ms | tok/sec: 6973.67
> step 9 loss: 6.890647888183594 | time: 306 ms | tok/sec: 6691.81
> step 10 loss: 6.672027587890625 | time: 300 ms | tok/sec: 6829.67
> # ...
> ```
> 在计算精度方面，我们仅修改了 2 行代码，同时使用 TF32 与 BF16 的方式，就将吞吐量从原来的 2000 提升到了 7000 左右，提升了 3 倍多。由此，对模型进行适当优化可以大幅缩短训练的时间。

> [!tip]
> 尽管 BF16 和 TF32 牺牲了一定的计算精度，但对于大多数任务而言，这种精度损失并不会显著影响最终模型的性能。通过加快训练过程，我们可以在相同时间内进行更多轮次的训练，从而弥补由于精度下降带来的细微损失。因此，权衡速度与精度，混合精度训练是一个非常有利的选择。

### 编译网络

PyTorch 2.0 推出的 `torch.compile` 技术通过编译计算图来减少 Python 解释执行的开销，并优化网络执行流程，从而加速训练。`torch.compile` 自动将动态图转化为优化的静态图，并在硬件上做相应的优化以提高计算效率。

启用 `torch.compile` 非常简单，只需一行代码：
```python
model = torch.compile(model)
```

> [!tip]
> 这算是 PyTorch 在 2.0 版本推出的黑科技。

> [!done]
> 启动 `torch.compile` 后，得到效果如下：
> ```
> step 0 loss: 11.025482177734375 | time: 59287 ms | tok/sec: 34.54
> step 1 loss: 9.418464660644531 | time: 258 ms | tok/sec: 7945.88
> step 2 loss: 8.805770874023438 | time: 292 ms | tok/sec: 7018.47
> step 3 loss: 8.511810302734375 | time: 308 ms | tok/sec: 6643.03
> step 4 loss: 8.228363037109375 | time: 304 ms | tok/sec: 6746.53
> step 5 loss: 7.918922424316406 | time: 282 ms | tok/sec: 7263.48
> step 6 loss: 7.627651214599609 | time: 281 ms | tok/sec: 7295.32
> step 7 loss: 7.340198516845703 | time: 300 ms | tok/sec: 6837.00
> step 8 loss: 7.1017303466796875 | time: 292 ms | tok/sec: 7003.59
> step 9 loss: 6.819297790527344 | time: 283 ms | tok/sec: 7224.83
> step 10 loss: 6.5823516845703125 | time: 287 ms | tok/sec: 7141.85
> # ...
> ```
> 观察可以发现，启用后第一次训练迭代较慢，这是因为编译过程需要花费一定时间来构建和优化计算图。然而，后续训练步骤中速度提升了约 10%。对于 RTX 3060 这样的消费级显卡，编译带来的加速效果有限，因为显卡的带宽和计算资源较少，编译优化的收益未能完全展现。但在 A100 等专业显卡上，torch.compile 可以显著提升模型训练效率。

### 超参数优化

在 GPU 计算中，矩阵乘法通常以对齐的内存块为单位进行操作（如 4x4 或更大）。使用不规则的形状（如奇数或质数）会导致额外的边界处理开销。为了减少这些不必要的计算，我们将词表大小从 50257 调整为 50304（2 的幂次倍数），这有助于提升 GPU 计算效率。

在我们的模型中，词表的大小是一个不怎么美观的数字，因此我们将其修改一个更加合适的数字。
```python
@dataclass
class GPTConfig:
    """GPT config data class"""

    block_size: int = 1024  # max sequence length
    # vocab_size: int = 50257  # number of tokens
    vocab_size: int = 50304  # use a 'beautiful' number to make cal faster
    n_layer: int = 12  # number of layers
    n_embd: int = 768  # embedding dimension
    n_head: int = 12  # number of head
```

> [!note]
> 这个简单的改动可以有效的提高模型计算速度的 10% 到 20% 左右。在之前创建网络时，我已经默认使用了这个数字，因此这里不展示优化效果。

> [!tip]
> 在这里，我们增加了一些没有使用过的 token，这些 token 在训练中从未出现，这实际上与出现频率很低的词的处理策略是一致的，因此引入一些额外的 token 对于模型来说不会有太多影响。虽然增加了一些未使用的 token，但它们在训练中被优化为接近 0 的权重，不会影响模型的推理性能。这个调整可以减少非对齐的内存访问和边界检查，从而带来一定的性能提升，尤其是在大规模输入场景下。

### 优化器融合 (fuse)

PyTorch 的优化器支持 `fuse=True` 参数，这项技术通过将多个小的操作合并为单一的大型操作，从而减少了数据在 CPU 和 GPU 之间的传输开销。对于像 AdamW 这样的优化器，它通常需要频繁的权重更新和梯度计算，而 `fuse` 技术可以减少这些操作的开销。

对代码的改动也相当小，只需添加参数 `fuse=True` 即可。
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
```

> [!done]
> fuse 技术几乎无损的提高了模型的训练速度，所以为什么不用呢？
> ```
> step 0 loss: 10.982025146484375 | time: 19855 ms | tok/sec: 103.15
> step 1 loss: 9.36016845703125 | time: 255 ms | tok/sec: 8032.90
> step 2 loss: 8.933940887451172 | time: 250 ms | tok/sec: 8186.15
> step 3 loss: 8.611137390136719 | time: 250 ms | tok/sec: 8177.06
> step 4 loss: 8.368354797363281 | time: 265 ms | tok/sec: 7722.83
> step 5 loss: 8.014617919921875 | time: 251 ms | tok/sec: 8147.52
> step 6 loss: 7.67095947265625 | time: 252 ms | tok/sec: 8133.68
> step 7 loss: 7.3626708984375 | time: 253 ms | tok/sec: 8098.46
> step 8 loss: 7.129791259765625 | time: 258 ms | tok/sec: 7934.73
> step 9 loss: 6.884559631347656 | time: 257 ms | tok/sec: 7961.01
> step 10 loss: 6.677295684814453 | time: 253 ms | tok/sec: 8098.43
> # ...
> ```

以上，就是我们可以做到的优化措施。可以看到，训练一个 step 的延迟从原先的 1400ms 降低到了现在的 250ms，吞吐量翻了 4.5 倍。优化在深度学习模型训练中至关重要，它能够充分挖掘硬件的潜力，使模型在有限的计算资源上达到最佳性能。

## 训练细节

在优化过程中，我们主要关注如何加速模型训练，使损失函数能够更快、更稳定地下降。以下的优化方法大多参考 GPT2 和 GPT3 论文中的策略，并不深入讨论超参数的调整。

### 优化器参数

我们使用 `AdamW` 优化器，并按照以下方式进行配置：

```python
def optimizer_configuration(self, weight_decay: float, learning_rate):
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True
    )
    return optimizer
```

> [!note]
> 1. 优化器配置依赖模型，因此被设计为模型方法。
> 2. `AdamW` 优化器参数参考自 GPT2 论文，这里不深入讨论其背后的细节。
> 3. PyTorch 默认对所有参数应用 `weight_decay`，但在实践中，`LayerNorm` 层和 `bias` 参数通常不需要衰减，因为它们的作用机制与正则化不同。通过上面的实现，我们对维度小于 2 的参数禁用了权重衰减。
> 4. `torch.Tensor.numel` 方法用于计算一个张量的元素个数。
> 5. PyTorch 支持通过 `optim_groups` 的方式为不同参数设置不同的优化策略。这里我们将参数分成两个组：一组进行权重衰减，另一组不进行衰减。

> [!tip] 关于权重衰退
> `weight_decay` 是一种正则化技术，旨在防止模型过拟合。然而，对于某些参数，如 `LayerNorm` 和 `bias` 参数，进行权重衰减可能会影响模型的性能。`LayerNorm` 负责对每个层的输出进行标准化，衰减这些参数可能会破坏这一操作的稳定性。同样地，`bias` 参数通常不会显著影响模型的复杂度，因此也不需要进行衰减。

### 梯度裁剪

在训练中，我们注意到模型的损失有时会突然增大，这通常是由于梯度爆炸的原因。梯度爆炸指的是在反向传播过程中，某些梯度的值变得非常大，导致参数更新幅度过大，模型沿着错误的方向大幅优化，从而引发损失异常增大。

这种情况在 RNN 中尤为明显，而在 Transformer 等模型中也可能发生。为了解决这一问题，我们使用了梯度裁剪技术，将梯度范数限制在一定范围内，从而避免训练崩溃。

使用梯度裁剪只需在反向传播后、参数更新前添加以下代码：

```python
x, y = x.to(device), y.to(device)
optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    _, loss = model(x, y)
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
loss.backward()
optimizer.step()
```

> [!note]
> 1. 梯度裁剪的阈值为 1.0，这个值通常能有效避免梯度爆炸，但不同任务可能需要调整该阈值。
> 2. 虽然梯度爆炸在 RNN 中更为常见，但在 Transformer 模型中也时有发生，尤其是在处理大规模数据时。

### 学习率调度

为了确保梯度下降算法能正确收敛，学习率需要在训练过程中逐渐减小。一个好的学习率调度器可以帮助模型更稳定、更快地收敛。在 GPT2 的训练中，我们使用了余弦退火和 Warm up 策略：

```python
def get_lr(it: int):
    """Cosine Annealing with Warm Up"""
    if it < WARMUP_STEPS:
        return MAX_LR * (it + 1) / WARMUP_STEPS
    if it > MAX_STEP:
        return MIN_LR
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEP - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coef * (MAX_LR - MIN_LR)
```

> [!note]
> 1. 余弦退火策略来自于 [SGDR: Stochastic Gradient Descent with Warm Restarts (arxiv.org)](https://arxiv.org/abs/1608.03983)。通过模拟学习率的“冷却”过程，避免模型陷入局部最优。
> 2. Transformer 模型比传统模型更难训练，因此在训练初期我们引入 Warm up 策略，使模型从较小的学习率开始，逐步增大，以找到正确的优化方向。
> 3. 使用学习率调度器时，PyTorch 背后会自动处理每一步的学习率更新，可以通过优化器的 `param_groups` 获取当前的学习率。

### 梯度累积

GPT-2 论文中推荐的批量大小为 52488，序列长度为 1024，这显然超出了普通 GPU 的处理能力。为了在有限的显存条件下实现与大批次相同的训练效果，我们采用梯度累积技术。
```python
optimizer.zero_grad()
loss_accum = 0.0
for mini_batch_step in range(GRAD_ACCUM_STEP):
	x, y = train_loader.next_batch()
	x, y = x.to(device), y.to(device)
	with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
		_, loss = model(x, y)
	loss /= GRAD_ACCUM_STEP
	loss_accum += loss.detach()
	loss.backward()
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
lr = get_lr(step)
for param_group in optimizer.param_groups:
	param_group["lr"] = lr
optimizer.step()
```

> [!note]
> 1. 在 PyTorch 中，梯度是通过动态图机制计算的。每次调用反向传播函数后，新计算出的梯度会被累加到已有的梯度上，因此在每次更新参数之前都需要清零梯度。为了模拟较大批次的效果，我们可以分批处理数据并累计梯度，在完成一个完整批次后才执行一次参数更新。
> 2. 实现梯度累积时的一个关键点是确保损失值反映了整个批次的平均损失。由于我们实际处理的是多个小批次，因此在每次累积时应调整损失值，使其反映整个批次的真实情况。由于各个小批次的大小一致，故此调整方式为除以梯度累积步数。
> 3. 使用梯度累积技术，理论上可以达到与一次性处理大批量数据相同的效果，从而在不增加显存需求的前提下提升模型训练质量。这种做法实际上是利用更多的计算时间来替代更大的内存需求。

## 训练

在确定训练过程无误，模型正确后，我们就已经具有在大规模数据集上训练的能力了。

### 数据集处理

在正式训练之前，我们需要先准备正式训练时使用的数据集。在本次训练中，我们将使用 Fine-Web Dev 数据集，该数据集取自 Fine-Web 中的教育数据，并经过清洗，质量较高。
