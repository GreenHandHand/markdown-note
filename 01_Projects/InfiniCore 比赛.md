# TODO

- [x] 尝试编译运行项目
- [x] 搞清楚如何提交、如何测试、如何开发 （试试写一个 ADD 的 demo）
- [x] 看看 CUDA 实现版本的接口
- [x] 看看 python 接口 (看看 ntops 库)
- [ ] 12/02 之前通过测试

| 题目          | 算子                  | 算子            | 算子          | 算子          | 算子        |
| ----------- | ------------------- | ------------- | ----------- | ----------- | --------- |
| T1-1-7      | *logsumexp*         | *lp_pool1d*   | *lp_pool2d* | *lp_pool3d* | max       |
| T1-1-8      | adaptive_avg_pool3d | argwhere      | addr        | fmin        | asin      |
| T1-1-25     | *log10*             | avg_pool3d    | histc       | dot         | log1p     |
| T1-1-28     | bitwise_left_shift  | index_select  | fold        | log2        | mish      |
| T1-1-29<br> | msort               | instance_norm | threshold   | celu        | transpose |

## 关于实现

1. *logsumexp*：这个算子的测试有一个是错误的。
2. *lp_pool1d*：torch 的默认行为是 replicate padding，但是我不知道九齿的框架下面能否实现？
	- 后面来回答：我使用 where + arrange 替代了 index，算是使用了一个麻烦的方法实现了。tile 函数不支持设置
	- 补充：写了 lp_pool2d 算子，发现实际上 torch 的处理方式是 $\sqrt[p]{ \text{avg}(W^{p}) * K }$，所以和补充 0 对不上。

> [!warn] CPU 实现
> 实现 CPU 版本比较麻烦，好在不需要考虑速度。实现一个 CPU 版本需要完成下面的内容：
> 1. 实现 c 这边的接口。
> 	1. 在 `include/infinicore/ops/` 下添加定义。定义可以直接复制其他算子然后稍作修改。
> 	2. 在 `src/infinicore/ops/*NAME*/*NAME*.cc` 中添加一系列函数。这个直接复制现成算子然后稍加修改。
> 	3. 在 `src/infinicore/ops/*NAME*/*NAME*_infinicore.cc` 中实现 CPU 版本。先复制其他算子，修改注册函数中中的 `Devie::Type::CPU`，然后就可以在 `calculate` 中写 CPU 实现了。
> 	4. 这里的 CPU 写起来非常麻烦，要考虑 stride、shape、多维度索引等等。
> 2. 实现 pybind 的接口。这里直接在 `src/infinicore/pybind11/ops/` 下定义即可，复制其他算子修改就行。
> 3. 实现分发函数。在 `python/infinicore` 中对应的位置（看测试）实现分发函数。

> [!note] 尽量使用 Python，因为 CUDA 在昇腾上面不知道如何适配。

> [!example] 文档地址
> https://gxtctab8no8.feishu.cn/wiki/KRIywGbrJiMdjtkj76cctaPvn4g#share-PWlDdHCXmoVbnRxTj6wcBwfPnT6

## 九源统一算子库

叫做这个：[GitHub - InfiniTensor/InfiniCore](https://github.com/InfiniTensor/InfiniCore)

文档说后面会补充有关比赛的说明文档和教程，先等等。

## 提交方式

给项目 [InfiniTensor/InfiniCore](https://github.com/InfiniTensor/InfiniCore) 提交 PR。

算子的测试在 `test\infinicore\ops\*.py`，测试方式为：

```
python test\infinicore\ops\*.py --verbose --bench --[平台]
```

## 参考

开发指南： [InfiniCore/src/infinicore/ops/README.md at main · GreenHandHand/InfiniCore](https://github.com/GreenHandHand/InfiniCore/blob/main/src/infinicore/ops/README.md)