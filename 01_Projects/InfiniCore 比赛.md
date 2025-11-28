# TODO

- [x] 尝试编译运行项目
- [x] 搞清楚如何提交、如何测试、如何开发 （试试写一个 ADD 的 demo）
- [x] 看看 CUDA 实现版本的接口
- [x] 看看 python 接口 (看看 ntops 库)
- [ ] 12/02 之前通过测试

| 题目          | 算子                 | 算子            | 算子        | 算子   | 算子        |
| ----------- | ------------------ | ------------- | --------- | ---- | --------- |
| T1-1-25     | *log10*            | avg_pool3d    | histc     | dot  | log1p     |
| T1-1-28     | bitwise_left_shift | index_select  | fold      | log2 | mish      |
| T1-1-29<br> | msort              | instance_norm | threshold | celu | transpose |

> [!note] 尽量使用 Python，因为 CUDA 在昇腾上面不知道如何适配。

> [!example] 文档地址
> https://gxtctab8no8.feishu.cn/wiki/KRIywGbrJiMdjtkj76cctaPvn4g#share-PWlDdHCXmoVbnRxTj6wcBwfPnT6

# 九源统一算子库

叫做这个：[GitHub - InfiniTensor/InfiniCore](https://github.com/InfiniTensor/InfiniCore)

文档说后面会补充有关比赛的说明文档和教程，先等等。

# 提交方式

给项目 [InfiniTensor/InfiniCore](https://github.com/InfiniTensor/InfiniCore) 提交 PR。

算子的测试在 `test\infinicore\ops\*.py`，测试方式为：

```
python test\infinicore\ops\*.py --verbose --bench --[平台]
```

# 参考

开发指南： [InfiniCore/src/infinicore/ops/README.md at main · GreenHandHand/InfiniCore](https://github.com/GreenHandHand/InfiniCore/blob/main/src/infinicore/ops/README.md)