# lingo 软件

LINGO是Linear Interactive and General Optimizer的缩写，即“交互式的线性和通用优化求解器”，由美国[LINDO](https://baike.baidu.com/item/LINDO/1923011?fromModule=lemma_inlink)系统公司（Lindo System [Inc.](https://baike.baidu.com/item/Inc./10390142?fromModule=lemma_inlink)）推出的，可以用于求解[非线性规划](https://baike.baidu.com/item/%E9%9D%9E%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92/5790466?fromModule=lemma_inlink)，也可以用于一些线性和[非线性方程](https://baike.baidu.com/item/%E9%9D%9E%E7%BA%BF%E6%80%A7%E6%96%B9%E7%A8%8B/4778289?fromModule=lemma_inlink)组的求解等，功能十分强大，是求解优化模型的最佳选择。

## 使用方式

lingo有两种计算方式，分别为：

- 直接运算
- 使用集合运算

在数据量较小时，我们直接将规划模型输入得到结果，当数据量较大时，就需要使用集合运算的方式了。

> 注：当求解非线性目标函数的最值时，请使用全局求解器，否则得到的将会是局部极值。

## 基本语法

### 集合

```lingo
sets:
集合名称1/成员列表1/:属性1_1,属性1_2,...,属性1_n1;
集合名称2/成员列表2/:属性2_1,属性2_2,...,属性2_n2;
派生集合名称（集合名称1，集合名称2）：属性3_1,...,属性3_n3;
endsets
```

### 数据

```lingo
data:
属性1 = 数据列表；
属性2 = 数据列表；
enddata
```

### 数据计算段

数据计算段部分不能含有变量，必须是已知数据的计算。

```lingo
calc:
b=0;
a=a+1;
endcalc
```

### 变量的初始化

变量初始化主要用于非线性问题赋值初始值。

```lingo
init:
X, Y = 5,1,2,7;
endinit
```

### 模型的目标函数与约束条件

顺序不重要。

```text
x1+x2<100;
max=98*x1+277*x2-x1^2-0.3*x1*x2-2*x2^2;
x1<=2*x2;
@gin(x1);@gin(x2);
```

## 集合运算

### 模板

lingo语言注意事项：

- 语句以分号(;)结尾
- 没有<与>符号，使用<=, >=, =, +, -,* ,/ ,^
- 函数写法为@+函数名，如@SUM
- 目标函数写法为：max/min = 目标函数;
- Lingo中是不区分大小写字符的。
- Lingo中数据部分不能使用分式，例如数据部分不能使用1/3.
- Lingo中的注释是使用"!"引导的。
- Lingo中默认所有的变量都是非负的。
- Lingo中矩阵数据是逐行存储的，而Matlab中数据是逐列存储的。这就解释了2.4中的那个小问题。

这里给出使用集合运算时的模板：

```lingo
!创建集合;
SETS:
SI/1..2/:a; !这样做将会创建一个含两个元素的集合a，实际使用中根据下标来规定集合;
SJ/1..3/:b; !这样做同理;
SIJ(SI, SJ):c; !多维的集合可以使用这种方法创建;
ENDSETS

!数据;
DATA:
a = 1, 2, 3; !可以使用逗号或者空格隔开;
b = 1, 2, 3;
c = 1, 2, 3
    4, 5, 6
    5, 6, 7; !这样可以创建二维的数据;
ENDDATA

min = @SUM(SI(i):a(i)) + @SUM(SJ(j):b(j)) + @SUM(SI(i):@SUM(SJ(j):a(i)*b(i)*C(i ,j))); !目标函数;
!约束条件;
```

上面是一个通常的规划问题编程模板。

## Lingo内置常用函数

### 内置算术运算符

加(+)、减(-)、乘(*)、除(/)、幂(^)

### 逻辑运算符

- 逻辑否#not#
- 相等#eq#
- 不相等#ne#
- 运算符大于#gt#
- 运算符大于等于#ge#
- 运算符小于#lt#
- 运算符小于等于#le#
- 逻辑与#and#
- 逻辑或#or#

### 关系运算符

可以使用大于等于（> 或 >=）、小于等于（< 或 <=）、等于（=）

### 数学函数

- @abs(x)：绝对值
- @sin(x)：返回x的正弦值，x采用弧度制
- @cos(x)：返回x的余弦值
- @tan(x)：返回x的正切值
- @exp(x)：返回e的x次方
- @log(x)：返回x的自然对数
- @lgm(x)：返回x的gamma函数的自然对数
- @mod(x, y)：返回x除以y的余数
- @sign(x)：如果x<0返回-1，否则返回0
- @floor(x)：返回x的整数部分
- @smax(x1, x2, ... , xn)：返回最大值
- @smin(x1, x2, ... , xn)：返回最小值

### 变量界定函数

- @bin(x)：限制x为0或者1
- @bnd(L, x, U)：限制L <= x <= U
- @free(x)：限制x为任意实数
- @gin(x)：限制x为整数
- 注：默认情况下x为大于等于0的数

### 集循环函数

- @for：遍历
- @sum：求和
- @min 与 @max：求最值

### 概率函数

- @pbn(p,n,x)：二项分布的累积分布函数。当n和（或）x不是整数时，用线性插值法计算。
- @pcx(n,x)：自由度为n的 χ2 分布的累积分布函数。
- @peb(a,x)：当到达负荷为a，服务系统有x个服务器且允许无穷排队时的Erlang繁忙概率。
- @pel(a,x)：当到达负荷为a，服务系统有x个服务器且不允许排队时的Erlang繁忙概率。
- @pfd(n,d,x)：自由度为n和d的F分布的累积分布函数。
- @pfs(a,x,c)：当负荷上限为a，顾客数为c，平行服务器数量为x时，有限源的Poisson服务系统的等待或返修顾客数的期望值。a是顾客数乘以平均服务时间，再除以平均返修时间。当c和（或）x不是整数时，采用线性插值进行计算。
- @phg(pop,g,n,x)：超几何(Hypergeometric)分布的累积分布函数。pop表示产品总数，g是正品数。从所有产品中任意取出n(n ≤ pop)件。pop、g、n和x都可以是非整数，这时采用线性插值进行计算。
- @ppl(a,x)：Poisson分布的线性损失函数，即返回max(0,z-x)的期望值，其中随机变量z服从均值为a的Poisson分布。
- @pps(a,x)：均值为a的Poisson分布的累积分布函数。当x不是整数时，采用线性插值进行计算。
- @psl(x)：单位正态损失函数，即返回max(0,z-x)的期望值，其中随机变量z服从标准正态分布。
- @psn(x)：标准正态分布的累积分布函数。
- @ptd(n,x)：自由度为n的t分布的累积分布函数。
- @qrand(seed)：产生服从(0,1)区间的伪随机数。

### 集操作函数

- @in(set_name, primitive_index1[, primitive_index_2, ...])
如果元素在指定集中，则返回1
- @index([set_name,] primitive *_* set_element)
该函数返回在集 set_name 中原始集成员 primitive *_* set *_* element 的索引。如果 set_name被忽略，那么Lingo将返回与 primitive *_* set *_* element 匹配的第一个原始集成员的索引。如果找不到，则产生一个错误。
- @wrap(index,limit)
该函数返回j=index-k*limit，其中k是一个整数，取适当值保证j落在区间[1，limit]内。该函数在循环、多阶段计划编制中特别有用。
- @size(set_name)
该函数返回集 set_name 的成员个数。在模型中明确给出集大小时最好使用该函数。它的使用使模型更加数据独立，集大小改变时也更易维护。
