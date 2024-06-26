## 3.pandas数据清洗与准备

pandas 提供了一个高级、灵活和快速的数据处理工具集。

### 1.处理缺失值

|函数名|描述|
|:--:|:--:|
|dropna|根据每个标签的值是否缺失数据来筛选标签|
|fillna|用某些值填充缺失的数据或使用插值方法('ffill'或'bfill')|
|isnull|返回表明那些值是缺失值的布尔值|
|notnull|isnull的反函数|

#### dropna

传入一个DaraFrame或者Series对象，默认情况下删去所有包含NA值的行。
可以通过传入how='all'参数来删除全部数据都是NA的行。
可以通过axis=1来对列进行操作。
可以通过thresh来保留含有一定数量观察值的行。

#### fillna

可以通过传入一个常数来代替NA值。
可以通过传入一个字典来为不同列设定不同的填充值。
fillna返回一个新的对象，但是也可以通过传入inplace=True来修改已经存在的对象。
重建索引中的插值方法也可以通过method参数传入，如果没有设置则默认为ffill(向前填充)，可以通过limit参数来设置向前或向后填充的最大范围。

### 2.数据转换

#### 删除重复值

通过方法**duplicated**与**drop_duplicates**可以对DaraFrame中的重复值进行操作。

duplicated将会返回DataFrame中每一行是否重复的布尔值。
drop_duplicates可以删除DataFrame中的重复行。

可以通过切片的方式对子结构进行操作来达到基于子结构进行操作的目的。

通过参数keep='last'可以保留最后一个观测值，默认情况下通常保留第一个观测值。

#### 通过函数进行数据转换

通过map方法可以对Series数据中的每一个值进行操作。也可以通过对DataFrame的列索引进行操作以达到对DataFrame进行操作的目的。

可以通过传入一个函数来对对象中的每一行进行操作。
可以通过传入一个包含映射关系的字典对象来将对应的数据进行转换。

该方法产生一个新对象，通过赋值的方法对原对香进行修改。

#### 替代值

使用fillna可以对NA值进行替代。但是对于普通值，可以使用map与replace方法进行替代。

replace可以传入两个参数，前者为被替代值，后者为替代值。
被替代值可以是一个常数，这么做将会将对象中的对应值替代。被替代值也可以是一个列表，这么做将会将列表中的所有值进行替代。
替代值可以是常数，这么做将会将被替代值中的所有值替代为替代值。替代值可以是与被替代值相同大小的列表，这么做将会将对应值替代。
参数也可以通过字典的方式传入。

#### 重命名轴索引

轴索引也同样有方法map。

通过DataFrame.index.map来对列进行修改。
通过DataFrame.columns.map来对行进行修改。

map返回一个新的对象，通过赋值的方式进行修改。

通过rename的方法结合字典型对象为轴标签提供新的值。

```python
frame.rename(columns={'new':'another'}, index={'ee':'ff'})
```

该方法生成一个新的对象，通过传入inplace=True来对原对象进行修改。

#### 离散化与分箱

通过cut方法将数据进行分箱，通过传入一个数组来对数据进行切割，该方法返回一个特殊的Categorical对象，该对象含有属性categories（类别），codes（每个数据属于的箱子）。

通过向pd.value_counts()中传入该对象可以得到每个类别与该类中的数量。

**cut函数**：
基于区间对数据进行划分。
可以通过right=False来确定右边是否封闭，即左边是否是开区间。
可以通过labels参数来传入每一类的名字。
可以通过传入一个整数而不是列表来指定将数据划分为几类，cut将会更加最大值与最小值进行划分。
可以通过precision来确定划分边界的精度。

**qcut函数**：
基于样本分位数进行分类。由于分位数的特殊性，传入一个常数进行划分时，得到的将会是等数量的划分。
也可以通过传入分位数列表进行划分，其余参数使用与cut函数相同。

#### 检测与过滤异常值

通过数组操作进行过滤。

#### 置换与随机抽样

置换即将数据进行重排序。可以通过**numpy.random.permutation**来对DataFrame中的Series或行进行置换。

numpy.random.permutation()传入一个整数作为参数，产生一个对应长度的打乱的随机数组。可以通过该数组对DataFrame进行随机排序。

通过iloc对行进行赋值可以实现随机排序。通过take函数传入该数组也可以实现相同的功能。

要从DataFrame中选出一个随机子集，可以通过sample方法。该方法传入一个n=m参数抽取m个样本，默认不可以重复选取。当replace=True传入允许重复选择。

#### 计算指标、虚拟变量

暂时没看懂

#### 字符串操作

##### python内建字符操作函数

python内建的字符串操作函数对于简单的应用是足够的：

|方法|描述|
|:--:|:--:|
|count|返回子字符串在字符串中的非重叠次数|
|endwith|如果字符串以后缀结尾，返回True|
|startswith|如果字符串以前缀开始，返回True|
|join|使用字符串作为间隔符，用于粘连其他字符串序列|
|index|如果在字符串中找到，则返回子字符串中第一个字符的位置，如果找不到则抛出ValueError|
|find|返回字符串中的第一个出现子字符串的位置；如果没找到，则返回-1|
|rfind|返回字符串中的最后一次出现子字符串的位置；如果没找到，则返回-1|
|replace|使用一个字符串替代另一个字符串|
|strip,rstrip,lstrip|修建空白，包括换行符|
|split|使用分隔符将字符串拆分为字符串列表|
|lower|将大写字符转换为小写字符|
|upper|将小写字符转换为大写字符|
|casefold|将字符转换为小写，并将任何特定区域的变量字符组合转换为常见的可比较形式|
|ljust,rjust|左对齐、右对齐；用空格或一些其他字符填充|

##### 正则表达式

python的正则表达式可以方便的处理文本。

通过python的re模块进行正则表达式操作。

正则表达式方法：

|方法|描述|
|:--:|:--:|
|findall|将字符串中所有的非重叠匹配模式以列表的形式返回|
|finditer|与findall类似，但返回迭代器|
|match|在字符串的起始位置匹配模式，也可以将模式组建分配到分组中；如果模式匹配上了，返回匹配对象，否则返回None|
|search|扫描字符串的匹配模式，如果扫描到了匹配对象返回。|
|split|根据模式，将字符串拆分|
|sub,subn|用替换表达式替换字符串中所有匹配（sub）或第n个出现的匹配串（subn），使用符号\1、\2来引用替换字符串中的匹配组元素。|

##### pandas中的向量化字符串函数

|方法|描述|
|:--:|:--:|
|cat|根据可选的分隔符按元素黏合字符串|
|contains|返回是否含有某个模式、正则表达式的布尔数组|
|count|模式出现的次数|
|extract|使用正则表达式从字符串Series中分组抽取一个或多个字符串；返回的结果是每个分组形成一列的DataFrame|
|endwith|等价于对每个元素使用x.endwith|
|startwith|等价与对每个元素使用x.startwith|
|findall|找出字符串中的所有的模式、正则表达式的匹配项，以列表返回|
|get|对每个元素进行索引（获得第i个元素）|
|isalnum|等价于内建的str.alnum|
|isalpha|等价于内建的str.isalpha|
|isdecimal|等价于内建的str.isdecimal|
|isdigit|等价于内建的str.isdigit|
|islower|等价于内建的str.islower|
|isnumeric|等价于内建的str.isnumeric|
|isupper|等价于内建的str.isupper|
|join|根据传递的分隔符，将Series中的字符串联合|
|len|计算每个字符串的长度|
|lower,upper|转换大小写|
|match|使用re.match将正则表达式应用到每个元素上，将匹配分组以列表的形式返回|
|pad|将空白加到字符串的左边、右边或者两边|
|center|等价与pad(side='both')|
|repeat|重复值|
|replace|以其他字符串替代模式、正则表达式的匹配项|
|slice|对Series中的字符串进行切片|
|split|以分隔符或正则表达式对字符串进行拆分|
|strip|对字符串两侧的空白进行消除，包括换行符|
|rstrip|消除字符串右边的空白|
|lstrip|消除字符串左边的空白|
