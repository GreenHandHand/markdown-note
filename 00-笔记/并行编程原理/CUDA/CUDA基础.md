---
tags:
  - 并行编程
  - CUDA
---

> [!info] 参考内容来自：[CUDA Training Series – Oak Ridge Leadership Computing Facility](https://www.olcf.ornl.gov/cuda-training-series/)

# CUDA C++ 基础

CUDA 的设计目标主要包括：
1. **充分利用 GPU 的并行性**，将其用于通用计算任务。
2. **提高性能**，通过异构计算架构提升计算效率。

CUDA C++ 是基于工业基础的 C++ 语言扩展，支持异构编程，提供了直观的 API 来管理 NVIDIA GPU 设备及其内存。它允许开发者编写高效的并行代码，同时利用 C++ 强大的面向对象特性来组织和优化代码结构。

> [!info] CUDA C++ 是目前最广泛应用的 CUDA 接口。此外，其他编程语言如 Fortran、Python 和 Matlab 也支持 CUDA 编程，但在本节中主要介绍 C++ 接口。

## 基本概念

CUDA 的异构编程模型将计算任务分配到两种设备上：
- **Device**：指的是 GPU 和 GPU 上的显存。在 CUDA 中，GPU 执行并行计算任务，处理计算密集型工作。GPU 内部的并行架构使其非常适合执行大量相似的计算任务。
- **Host**：指的是 CPU 和 CPU 上的内存。在 CUDA 程序中，CPU 执行剩余的序列性任务，并负责协调与 GPU 之间的数据传输和指令调度。

## 一般的处理流程

一个典型的 GPU 处理流程如下：
1. 将数据从主机内存复制到显存中。数据传输通过 PCIe 或 NVLink 总线进行，显存通常是 GPU 上的 DRAM，通常称为全局显存 (Global memory)。
2. 加载 GPU 程序并在 GPU 上执行。GPU 会将数据缓存并进行并行计算，计算结果会存储在显存中。
3. 将计算结果从显存传回主机内存。

> [!note] GPU 计算通过大规模的并行线程进行，显著提升了计算效率。然而，数据传输时间可能成为瓶颈，因此尽量减少数据传输量是提高性能的关键之一。

## C++ 接口

### `__global__` 函数

在 CUDA C++ 中，`__global__` 修饰符标记的函数在 GPU 上运行，并且可以从 Host 端调用。这些函数被称为内核（Kernels），它们构成了 CUDA 程序的核心部分，执行具体的并行计算任务。

> [!note] **nvcc 编译器**
> nvcc 是 CUDA 专用的编译器。它能够将 CUDA 代码中的 Host 部分与 Device 部分分开处理，分别由 C++ 编译器和 NVIDIA 编译器编译。通常，`__global__` 函数会被编译为 GPU 代码，而 `main` 函数等 Host 代码则由标准的 C++ 编译器编译。

在 CUDA 中，使用 `<<<>>>` 语法启动内核函数，配置内核的执行参数（如线程块大小、线程网格大小等）。

### CUDA Memory API

CUDA 提供了一些基本的内存管理函数，用于显存的分配、复制和释放：
- `cudaError_t cudaMalloc(void **devPtr, size_t size)`：在 GPU 上分配内存。
- `cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)`：在 Host 和 Device 之间传输数据。
- `cudaError_t cudaFree(void *devPtr)`：释放 GPU 显存。

> [!tip] 这些函数与 C 语言中的 `malloc()`, `memcpy()`, `free()` 函数类似，因此很容易理解。正确管理内存是避免错误和提高程序性能的重要步骤。

### 内置变量

每个执行 `__global__` 函数的线程会自动生成一些内置变量，如 `blockIdx` 和 `threadIdx`，用来标识线程在块和网格中的位置。

> [!note] **CUDA 层次划分**
> 在 CUDA 中，计算线程被划分为多个层次，以便更好地组织并行计算任务：
> 1. **线程（Thread）**：执行程序的最小单位。GPU 通过并行执行大量线程加速计算。
> 2. **线程块（Block）**：由多个线程组成，线程块内的线程可以共享内存，这有助于提高某些类型应用的性能。
> 3. **线程网格（Grid）**：由多个线程块组成，代表了整个并行计算任务的规模。

每个核函数在执行时会自动创建以下变量，这些变量提供了线程的多维索引：
- `blockIdx`、`blockDim`、`threadIdx` 和 `gridDim`，每个变量包含三个维度（x、y、z）。

> [!example] 线程编号示例
> 下面是获取全局唯一线程编号的常见写法，这对于确保每个线程处理不同的数据点至关重要：
> ```cpp
int idx = threadIdx.x + blockIdx.x * blockDim.x;
int idy = threadIdx.y + blockIdx.y * blockDim.y;
int idz = threadIdx.z + blockIdx.z * blockDim.z;
> ```

### 例程

下面是一个简单的向量加法的 CUDA 示例，演示了如何定义内核函数以及如何在主函数中调用该内核：

```cpp
__global__ void vectorAdd(const float* A, const float* B, float* C, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < size)
		C[idx] = A[idx] + B[idx];
}

int main()
{
	float *h_A = new float[DSIZE];
	float *h_B = new float[DSIZE];
	float *h_C = new float[DSIZE];
	assignValue(h_A, h_B, h_C);

	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, DSIZE * sizeof(float));
	cudaMalloc(&d_B, DSIZE * sizeof(float));
	cudaMalloc(&d_C, DSIZE * sizeof(float));

	cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

	vectorAdd<<<DSIZE/1024, 1024>>>(d_A, d_B, d_C, DSIZE);

	cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
```

## 额外内容

### 声明多维

为了更高效地利用 GPU 资源，有时需要声明多维的网格和块配置，以匹配具体问题的空间分布特征。下面是一个二维块和网格配置的示例：

```cpp
dim3 block(block_x_dim, block_y_dim);
dim3 grid(floor(DSIZE/block.x), floor(DSIZE/block.y));
mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);`
```

在内核函数中，线程索引计算方式如下：

```cpp
int idx = threadIdx.x + blockIdx.x * blockDim.x;
int idy = threadIdx.y + blockIdx.y * blockDim.y;
d_A[idx + idy * DSIZE] = ...;
```

### 错误处理

CUDA 使用返回值而非终止进程来报告错误。以下是一个错误检查的宏定义，可以帮助开发者更快地定位问题所在：

```cpp
#define cudaCheckErrors(msg) \
	do { \
		cudaError_t __err = cudaGetLastError(); \
		if (__err != cudaSuccess) { \
			fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, \
			cudaGetErrorString(__err), __FILE__, __LINE__); \
			fprintf(stderr, "*** FAILED - ABORTING\n"); \
			exit(1); \
		} \
	} while (0)
```

---
| [[00-笔记/并行编程原理/CUDA/CUDA共享显存|共享显存]] >
