# pandas

pandas 提供了高级数据结构和函数，这些数据结构和函数的设计使得利用结构化、表格化数据的工作快速、简单、有表现力。该部分内容是后续 [[00-笔记/实践知识/pandas/pandas数据载入与读取]]、[[00-笔记/实践知识/pandas/pandas数据类型与修改]]、[[00-笔记/实践知识/pandas/pandas数据清洗与准备]]、[[00-笔记/实践知识/pandas/pandas数据规整：连接、联合与重塑]]、[[00-笔记/实践知识/pandas/pandas数据聚合与分组操作]] 的前置知识。

## 1.pandas数据结构简介

### 1.Series

Series 是一种一维的数组对象，包含一个值序列与数据标签（引索，index）。对应表格中的一列元素。

#### 创建Series

使用pd.Series()创建，参数为可迭代对象。可以指定标签序列为引索值。若不指定引索值，默认的引索值为0到N-1。

例：

```python
obj1 = pd.Series([53, 70, 87, 64, 80])
obj2 = pd.Series([69, 47, 4, 55, 33],['v', 'n', 'y', 'e', 'w'])
```

```output
obj1:                   obj2:
0    53                 v    69
1    70                 n    47
2    87                 y     4
3    64                 e    55
4    80                 w    33
dtype: int64            dtype: int64
```

#### value与index属性，Series的引索

可以分别通过value与index两个属性获得表格的值与引索。同时可以使用引索值来进行引索。可以进行与numpy相同的操作,只是索引值不仅仅是整数。与普通的python切片不同，Series的切片是包含尾部的。
Series的索引还可以按位置方式改变，修改用的索引列表必须与原来的索引列表一致

```python
obj[['v','y']]
obj.index=[2,3,3,'a','b']
```

```output
v    69
y     4
dtype: int64

2    69
3    47
3     4
a    55
b    33
dtype: int64
```

#### 字典与Series

一个Series对象可以看作一个有序的字典。如果已有一个字典，那么也可以通过字典生成一个Series对象。也可以通过一个字典与一个列表生成以个Series对象，这么做将会从字典中选出对于的列表值，如果没有列表值，将会赋值为NaN。

```python
c = [chr(random.randint(97,123)) for _ in range(10)]
d = dict(zip(b,a))
pd.Series(d, c)
```

```output
z     NaN
m     NaN
c     NaN
f     NaN
l     NaN
m     NaN
e    55.0
n    47.0
w    33.0
z     NaN
dtype: float64
```

#### 检查NaN

使用 isnull 与 notnull 来检查NaN。他们既可以是方法也可以是函数。

```python
c = [chr(random.randint(97,123)) for _ in range(10)]
d = dict(zip(b,a))
obj = pd.Series(d, c)
pd.isnull(obj)
```

```output
i     NaN               i     True
l     NaN               l     True
i     NaN               i     True
j     NaN               j     True
v    69.0               v    False
n    47.0               n    False
n    47.0               n    False
e    55.0               e    False
h     NaN               h     True
d     NaN               d     True
dtype: float64          dtype: bool
```

#### Series与数据库

Series可以进行与数据库相同的运算。

|Series操作|数据库操作|
|--|--|
|+|join|

#### Series方法与属性总结

|名称|功能|
|:--:|:--:|
|name|Series名称|
|value|值|
|index|引索|
|dtype|类型|
|pd.Series()|创建对象|
|s > 0|布尔运算，返回布尔Series|
|s[s>0]|布尔引索|
|isnull,notnull|判断是否是NaN，返回布尔Series|
|a+b|a join b，对应引索相加，没有则合并|

### 2.DataFrame

DataFrame是表示矩阵的数据表，包含已排列的列集合，每一列都可以是不同的数据类型。

#### DataFrame的创建

创建DataFrame通过pd.DataFrame()，有多种方式可以创建，这里列举几个。

- 利用包含等长度列表的Numpy数字的字典来形成DataFrame。默认按照排序顺序排列。

```python
data = {'first':[1,2,3,4,5],
    'second':[2,4,6,8,10],
    'third':['a','b','c','d','e']}
frame = pd.DataFrame(data)
```

```output
    first    second    third
0    1    2    a
1    2    4    b
2    3    6    c
3    4    8    d
4    5    10    e
```

- 也可以使用columns参数使列按指定顺序排列。

```python
data = {'first':[1,2,3,4,5],
        'second':[2,4,6,8,10],
        'third':['a','b','c','d','e']}
frame = pd.DataFrame(data,columns=['third','second','first'])
```

```output
    third    second    first
0    a    2    1
1    b    4    2
2    c    6    3
3    d    8    4
4    e    10    5
```

- 可以通过index参数指定行引索

```python
data = {'first':[1,2,3,4,5],
        'second':[2,4,6,8,10],
        'third':['a','b','c','d','e']}
frame = pd.DataFrame(data,index=['aa','bb','cc','dd','ee'])
frame
```

```output
    first    second    third
aa    1    2    a
bb    2    4    b
cc    3    6    c
dd    4    8    d
ee    5    10    e
```

- 使用嵌套的字典创建DataFrame时，会将字典的键作为列，内部字典的键作为行引索。

```python
data = {'first':{'aa':1,'bb':2,'cc':3},'second':{'bb':8,'cc':9,'dd':4}}
pd.DataFrame(data)
```

```output
    first    second
aa    1.0    NaN
bb    2.0    8.0
cc    3.0    9.0
dd    NaN    4.0
```

- 使用值为Series的列表也同样可以构造DataFrame

#### DataFrame引索与切片

从DataFrame中的引索得到的结果为数据的视图，也就是修改得到的Series将该改变原数据，如果需要得到新的数据，应当使用Series的copy方法

##### 列引索

DataFrame中的一列，可以通过列引索，以字典型形式或者属性形式检索为Series。

```python
frame.second
frame['second']
```

```output
aa     2
bb     4
cc     6
dd     8
ee    10
Name: second, dtype: int64
```

列的引用是可以修改的。可以修改为列表或者一个值。当你将列表赋值给一列时，列表必须与列一样长。当你将Series赋值给一列时，Series引索将会按照DataFrame引索重新排列，并在空缺出填充缺失值。若赋值个不存在的列，将该创建一个新的列。
需要注意的是，像属性一样引用无法创建新的列。

```python
frame['new']=12
frame['new']=[_ for _ in range(10) if _ % 2 == 0]
```

```output
    first    second    third    new
aa    1    2    a    12
bb    2    4    b    12
cc    3    6    c    12
dd    4    8    d    12
ee    5    10    e    12


first    second    third    new
aa    1    2    a    0
bb    2    4    b    2
cc    3    6    c    4
dd    4    8    d    6
ee    5    10    e    8
```

del 关键词可以像字典一样删除一列。

```python
del frame['series']
```

##### 行引索

DataFrame的行索引通过loc与iloc属性实现。
loc支持使用行标签进行索引，iloc支持使用整数进行索引。他们两个也支撑切片。
使用这两个特殊属性可以使用同numpy的方法对DataFrame进行索引。
索引选项：

|类型|描述|
|:--:|:--:|
|df[val]|从DataFrame中选择单列或列序列|
|df.loc[val]|根据标签选择DataFrame的单行或多行|
|df.loc[:, val]|根据标签选择单列或多列|
|df.loc[val1, val2]|同时选中行列的一部分|
|df.iloc[where]|根据整数位置选择单行或多行|
|df.iloc[:,where]|根据整数位置选择单列或者多列|
|df.iloc[where_i,where_j]|根据整数位置选择行和列|
|df.at[label_i,label_j]|根据行列标签选择单个标量值|
|df.iat[i,j]|根据行列整数位置选择单个标量值|
|reindex 方法|通过标签寻找行和列|
|get_value, set_value|根据行和列的标签设置单个值|

```python
frame.loc['aa']
frame
```

```output
first     1
second    2
third     a
Name: aa, dtype: object
```

#### DataFrame索引对象

索引对象用于储存轴标签和其他元数据。相当于一个不可变的集合。索引对象的不可变性使得在不同的数据结构中分享索引对象是安全的。
与python中的集合不同，索引对象中的元素是可以重复的。
下面的是索引对象的一些方法：

|方法|描述|
|--|--|
|append|将额外的索引对象添加到原索引之后，产生一个新的索引|
|difference|计算两个索引的差集|
|intersection|计算两个索引的交集|
|union|计算两个索引的并集|
|isin|计算每个值是否在传值容器中的布尔数组|
|delete|将位置i的元素删除，并产生新的索引|
|drop|根据传参删除指定的索引值，并产生新的引索|
|insert|在位置i上插入元素，并产生新的引索|
|is_monotonic|如果索引序列递增则返回True|
|is_unique|如果索引序列唯一则返回True|
|unique|计算索引的唯一值序列|

#### 对DataFrame进行操作

- 转置，DataFrame.T

#### DataFrame方法与属性总结

##### 可以用于创建DataFrame的对象

|类型|注释|
|--|--|
|2D ndarray|数据的矩阵，行列标签为可选参数|
|数组、列表与元组构成的字典|每个序列称为DataFrame的一行，所有的序列必须长度相等|
|Series构成的字典|每个值称为一列，每个Series的引索联合起来称为行引索，也可以显示地传递行引索|
|字典构成的字典|每个值称为一列，键联合起来形成的结果构成行引索|
|列表后元组构成的列表|与2D array相同|
|其他DataFrame|如果不显示地传递引索，则引索为原来的|
|Numpy MaskedArray|与2D array相同，但隐蔽值会在结果中成为NaN|

##### DataFrame的一些属性与操作

|名称|功能|
|:--:|:--:|
|pd.DataFrame()|创建DataFrame|
|columns|返回列引索|
|index|返回行引索|
|frame['first']|返回Series对象|

### 3.基本功能

#### 重建引索 reindex

该方法用于创建一个符合新引索的新对象。
当Series调用该方法时，会将数据按照新的引索进行排列，不存在的引索赋值为NaN。
对于顺序数据，如时间序列，在重建引索时可能需要插值或填值。method可选参数运行我们使用ffill等方法插值。
在DataFrame中，reindex可以改变行索引，类索引，也可以同时改变二者。当仅传入一个序列时，优先重建行序列。使用columns参数可以重建列序列。
reindex参数表：

|参数|描述|
|:--:|:--:|
|index|新建作为引索的序列|
|method|插值方法，ffill为向前填充,bfill为向后填充|
|fill_value|通过重新索引引入缺失数据时使用的替代值|
|limit|向前或向后填充时，所需填充的最大尺寸间隔（单位为元素数量）|
|tolerance|向前或向后填充时，所需填充的不精确匹配下的最大尺寸间隔（以绝对数字距离为单位）|
|level|匹配MultiIndex级别的简单索引；否则选择子集|
|copy|如果为True，创建新的数据|

#### 轴向上删除条目

通过drop方法，可以从轴向上删除一个条目，即直接删去列或者行条目。
若不指定axis，则从行上删除。如果要删除列，则可以使用axis=1或者axis='columns'参数。
drop方法会直接修改Series或DataFrame的尺寸和形状，直接操作原对象。

#### 算术

当将不同的数据相加时，没有交叠的数据上将会产生缺失值。

```python
frame
frame1
frame + frame1
```

```output
frame:                                  frame1:
    first    second    third                   first    second
aa    1    2    a               aa    5.0    NaN
bb    2    4    b               bb    2.0    8.0
cc    3    6    c               cc    3.0    9.0
dd    4    8    d               dd    NaN    4.0
ee    5    10    e

frame + frame1:
    first    second    third
aa    6.0    NaN    NaN
bb    4.0    12.0    NaN
cc    6.0    15.0    NaN
dd    NaN    12.0    NaN
ee    NaN    NaN    NaN
```

灵活算术方法：可以通过fill_value参数指定不重叠位置赋值。
灵活算术方法：

|方法|描述|
|:--:|:--:|
|add,radd|加法(+)|
|subr,sub|减法(-)|
|div,rdiv|除法(/)|
|floordiv,rfloordiv|整除(//)|
|mul,rmul|乘法(*)|
|pow,rpow|幂次方(**)|

当算术对象为Series与DataFrame时，默认情况下，是将Series与DataFrame的列进行匹配，再广播到每一行。若想要广播到每一列，则需要使用上面是算术方法，并指定参数axis='index'。如果一个引索既不在DataFrame的列中也不在Series中，则会重建一个引索并形成联合。

#### 函数和通用映射

Numpy中的通用函数对DataFrame也同样适用。
将函数应用到一行或者一列一维数组上，可以通过apply方法，apply(function, *axis)，在不传入axis='columns'时，默认将函数应用到每一列上。
将函数应用到每一个数据上，可以使用applymap方法。
Series可以使用map方法。

pandas对象装配了常用数学、统计学方法的集合，其中大部分属于归约或汇总统计的类别。与NumPy数组中的方法相比，它们内建了处理缺失值的功能。

可以通过参数include='all'指定所有列，include后加指定的类型。

归约方法：除非整个切片上都是NaN，否则NaN都会被忽略。
归约方法可选参数：

|方法|描述|
|:--:|:--:|
|axis|归约轴，0为行向，1为纵向|
|skipna|排除缺失值，默认为True|
|level|如果轴是多层索引的，该参数可以缩减分组层级|

|方法|描述|
|:--:|:--:|
|count|非NA值的个数|
|describe|计算Series和DaraFrame各列的汇总统计集合|
|min,max|计算最小值、最大值|
|argmin,argmax|分别计算最小值、最大值所在是索引处（整数）|
|idxmin,idxmax|分别计算最小值或最大值所在的索引标签|
|quantile|计算样本从0到1间的分位数|
|sum|加和|
|mean|均值|
|median|中位数|
|mad|平均值的平均绝对偏差|
|prod|所有值的积|
|var|值的样本方差|
|std|值的样本标准差|
|skew|样本偏度值|
|kurt|样本峰度值|
|cumsum|累计值|
|cummin,cummax|累计值的最大、最小值|
|cumprod|值的累计积|
|diff|计算第一个算术差值|
|pct_change|计算百分比|

#### 排名与排序

按行或者按列索引进行排序：sort_index方法，该方法返回一个新的，排好序的对象。
根据值排序：sort_value方法，默认情况下，所有缺失的值都会被排在Series的尾部。当对DataFrame进行排序时，可以传递一个或者多个列给可选参数by作为排序键。ascending为True时为升序。
参数列表：

|参数|描述|
|:--:|:--:|
|axis|0时按行排序，1时按列排序|
|ascending|默认为True，升序，False时为降序|
|by|指定排序键|
|method|指定排序方法，可以打破平级关系|

对数据进行排名，即对数组中的有效数据从1开始分配名次的操作：rank方法。参数同上。

rank方法的参数method的可选参数：

|参数|描述|
|:--:|:--:|
|'average'|默认：在每个组中平均分配排名|
|'min'|对整个组使用最小排名|
|'max'|对整个组使用最大排名|
|'first'|按照值在数据中出现的次序分配排名|
|'dense'|类似于'min'，但组间排名总是增加1|

#### 判断唯一索引、计数与成员属性

索引重复的标签将会输出所有满足的标签。
可以通过标签的is_unique来判断该标签是否唯一
unique方法可以给出一个Series中的唯一值
value_counts方法计算Series中包含的每个值的个数，按照降序排列
isin方法用于判断每一个数据是否在参数列表中

#### 相关性与协方差

corr：计算两个Series中重叠的、非NA的、按索引对齐的值的相关性。
cov：计算协方差
