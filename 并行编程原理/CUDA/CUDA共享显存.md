---
tags:
  - 并行编程
  - CUDA
---

# CUDA 共享显存

在向量加法的例子中，每个输出值仅依赖于输入中相同位置的值。然而，实际计算任务往往更为复杂。例如，在一维模板计算中，输出值不仅依赖于当前输入元素，还需要考虑与其相邻位置的值。为了处理这种情况，线程间的通信显得尤为重要。

本节通过一维模板 (1D stencil) 计算来引入 CUDA 中的共享显存概念。

> [!info] **一维模板计算**
> 模板计算是一种在高性能计算中非常重要的技术。它通过划分计算问题为局部模板，从而高效地解决问题。模板计算中，某个值的计算依赖于其相邻位置的元素。常见的应用包括热传导方程、图卷积运算和图像滤波等。
>
> 下面是一个典型的一维模板计算的定义：
> $$
> \begin{align}
>	\mathbf{stencil}(x_{i}) &= \begin{cases}
>	\displaystyle\sum\limits_{k=-r}^{r}x_{i+k} & r \leqslant i \leqslant n - r \\[0.3em]
>	\displaystyle\sum\limits_{k=-i}^{k=r}x_{i+k} & i < r \\[0.3em]
>	\displaystyle\sum\limits_{k=-r}^{n-i}x_{i+k} & i > n - r
>	\end{cases} \\[0.5em]
> \mathbf{stencil}(X) &= [\mathbf{stencil}(x_{1}),\mathbf{stencil}(x_{2}),\cdots,\mathbf{stencil}(x_{n})]
> \end{align}
> $$
该操作类似于在一维数组上执行大小为 `2*r+1` 的卷积操作，其中 `r` 是超参数，表示模板窗口的半径。

> [!note] **线程间通信**
> 在这个例子中，每个线程的输出不仅依赖于自己的输入元素，还需要考虑与其相邻位置的输入元素。因此，线程之间必须进行通信。

## 共享显存

在每个线程块 (Block) 内，CUDA 提供了共享显存 (shared memory)，这是一个高效的内存区域，用于线程块内的线程之间进行数据共享。共享显存的访问速度比全局显存快很多，因此可以显著提升计算性能。

> [!note] **共享显存与全局显存的区别**
> - **全局显存 (Global memory)**：这是计算单元外部的 DRAM 空间，通常较大并且集成度较高。全局显存用于存储所有线程和块之间的数据，但其访问速度较慢。
> - **共享显存 (Shared memory)**：这是计算单元内部的高速缓存，访问速度远快于全局显存。它用于线程块内部的线程间通信，但共享显存的容量通常较小，通常为几 MB。
> 使用 `__shared__` 关键字可以在核函数中声明共享显存。

## 例程：一维模板函数实现

下面是使用共享显存来实现一维模板计算的例程：

```cpp
__global__ void stencil_1d(int* in, int* out){
    __shared__ int temp[BLOCK_SIZE + 2*RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + radius; // 填充 radius

    // 将输入元素写入共享显存
    temp[lindex] = in[gindex];

    // 填充部分
    if(threadIdx.x < RADIUS){
        temp[lindex - RADIUS] = 0;
        temp[lindex + BLOCK_SIZE] = 0;
    }

    // 同步函数，确保所有线程都完成数据填充
    __syncthreads();

    // 执行模板计算
    int result = 0;
    for(int offset = -RADIUS; offset <= RADIUS; offset++){
        result += temp[lindex + offset];
    }

    out[gindex] = result;
}
```

### `__syncthreads()`

在 GPU 中，所有线程是并行执行的，因此在涉及多个数据的操作时，可能会出现 [[计算机组成原理/中央处理器#数据冒险|数据冒险]] 问题。为了保证线程的同步执行，避免读取未更新的数据，CUDA 提供了 `__syncthreads()` 函数。

`__syncthreads()` 会同步一个线程块内的所有线程，确保所有线程在执行到此函数时都已经达到同步点，避免数据竞争。

> [!info] **协作组**
> 在 CUDA 中，一个线程块中的所有线程都可以视为一个协作组。协作组提供了一些方法来进行线程间同步：
>
> - `void sync()`：同步组中的所有线程。
> - `unsigned size()`：获取线程组中的线程数。
> - `unsigned thread_rank()`：获取线程在组中的编号。
> - `bool is_valid()`：检查线程组是否违反任何约束。
> - `dim3 group_index()`：获取线程块在网格中的编号。
> - `dim3 thread_index()`：获取线程在块中的编号。

---
< [[并行编程原理/CUDA/CUDA基础|CUDA基础]] |
