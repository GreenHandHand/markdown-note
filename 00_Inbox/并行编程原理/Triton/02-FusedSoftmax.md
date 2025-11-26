---
tags:
  - Triton
  - 并行编程
---

# FusedSoftmax

本节描述一个稍微复杂一些的例子 `softmax`，其数学公式如下：

$$
\text{Softmax}(x_i) = \frac{\exp(x_{i})}{\sum _{j}\exp(x_{j})}
$$

为了计算数值稳定，一般使用下面的表达式：

$$
\text{Softmax}(x_{i}) = \frac{\exp(x_{i} - x_{\text{max}})}{\sum_{j}\exp(x_{j} - x_{\text{max}})}
$$

## 朴素实现

在 `torch` 的框架下，我们可以用下面的代码实现一个朴素的 `Softmax`（假设输入是 $M\times N$）：

```python
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch
    """
    # read MN elements ; write M elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements; write MN elements
    z = x - x_max[:, None]
    # read MN elements; write MN elements
    numerator = torch.exp(z)
    # read MN elements; write M elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements; wrote 3MN + 2M elements
    return ret
```

也就是说，在 `pytorch` 中直接实现 `Softmax`，需要从 DRAM 中读取 $5MN+2M$ 个元素，并写回 $2MN+2M$ 个元素。这些操作显然是多余的，我们可以将这整个算子融合，使得其一次性全部计算完成，这称为 `fuse softmax` 算子。

可以简单计算一下，当我们将其融合之后，对于大小为 $M\times N$ 的输入，只需要从 DRAM 中读取一次，写入一次，因此我们理论上可以加速 4 倍 ($\frac{8MN+4M}{2MN}$)。

> [!note]
> 我们也可以通过 `torch.jit.script\torch.compile` 实现这种融合操作，但是它的效果仍然不够理想。

## Triton 实现

我们仍然假设输入的大小为 $M\times N$，因此 `Softmax` 操作的过程大致可以理解为：

1. 读取输入的每一行，计算这一行的最大值 $x_{\max}$
2. 计算 $\exp (x - x_{\max})$
3. 求和并结算最后的结果

这在 Triton 中实现起来非常直观 (在 CUDA 中就比较麻烦了)：

```python
@triton.jit
def softmax_kernel(
	output_ptr,
	input_ptr,
	input_row_stride,
	output_row_stride, 
	n_rows, 
	n_cols, 
	BLOCK_SIZE: tl.constexpr,
	num_stages: tl.constexpr
):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

理解上面的代码的核心问题在于下面几个：

1. 偏移的计算：在这个问题中，我们是按行来计算 Softmax 的。

> [!tip] 
> triton 中有一个重要的限制，是每一个块必须是 2 的幂次方大小。因此，为了能够处理任意可能的形状，我们需要在内部填充每一行，并适当保护内存。