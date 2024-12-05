## 3.数据规整：连接、联合与重塑

在很多应用中，数据可能分布在多个文件或者数据库中，抑或以某种不易于分析的格式进行排列。

### 1.分层索引

分层索引允许在一个轴向上拥有多个索引层级。通过分层索引，你可以简洁地选择出数据的子集。

#### 分层索引的创建

创建一个拥有分层索引的DataFrame可以通过使用元素为列表的列表实现，要求元素一一对应。下面是一个分层索引的例子：

```python
frame = pd.DataFrame(np.arange(12).reshape((4,3))
                    ,index=[['a','a','b','b'],[1,2,3,4]]
                    ,columns=[['first','first','second'],['one','two','three']])
```

```output
        first         second
          one    two   three
a    1      0      1      2
     2      3      4      5
b    3      6      7      8
     4      9     10     11
```

还可以使用index.names或者columns.names为层级命名，如果层级有名字，则会在输出中显示。

#### 获得层级索引内容

多层索引的查看可以使用get_level_values方法，该方法属于索引对象。

可以使用元组来一次完成多层索引的目的。

```python
#获得索引第二层的内容
df.index.get_level_values(1)
df.columns.get_level_values(1)
```

#### 重排序与层级排序

有时需要重新排列轴上的层级次序，或者按特定层级的值对数据进行排序，可以通过**swaplevel**方法。

**swaplevel**方法接受两个层级序号或层级名称，返回一个进行了层级变更的新对象（数据不变）。

sort_index方法只能在单一层级上实现，通过接受参数level可以指定排序的层级，最里层序号为0，所以在多层级排序时需要结合使用swaplevel与sort方法。

#### 按层级进行汇总统计

DataFrame的很多描述性、汇总性统计中都含有参数level可以指定层级。

#### 使用DataFrame的列进行索引

可以通过set_index方法，将DataFrame中的一个或者多个列作为行索引创建一个新的DataFrame。

默认情况下这些列将会被移除，可以通过参数drop=False将他们保留在数据中。

reset_index是set_index的反操作，分层索引的索引层级将=将会被移动到列中。

可以使用reset_index(drop = True)来将DataFrame的索引重新从零开始排列。

### 2.联合、合并数据集

包含在pandas对象中的数据可以通过多种方式进行联合：

|函数|描述|
|:--:|:--:|
|pandas.merge|根据一个或多个键将行连接。|
|pandas.concat|使对象在轴向上进行黏合或堆叠。|
|combine_first|实例方法允许将重叠的数据拼接在一起，以一个对象中的值填充另一个对象的缺失值。|

#### merge  数据库风格的DataFrame连接

通过一个或多个键连接行来联合数据集。这种方法同数据库中的join操作。

该方法需要传递两个DataFrame对象，输出将他们连接的结果。若不显式的指定连接键，默认会选择重叠键进行连接。

若要指定连接键，通过参数on='key'可以实现。
如果连接键的列名不同，可以通过left_on、right_on来指定左右。

how参数可以指定连接方式。默认情况下merge使用的是内连接（'inner' join，取交集）的方式，其他的可选选项有'left','right','outer'（外连接）。总结如下表：

|选项|行为|
|:--:|:--:|
|inner|对两张表的所有键的交集进行联合|
|left|对左表的所有键进行联合|
|right|对右表的所有键进行联合|
|outer|对两张表的所有键的并集进行联合|

merge参数表：

|参数|描述|
|:--:|:--:|
|left|合并时操作中左边的DataFrame|
|right|合并时操作中右边的DataFrame|
|how|'inner'、'left'、'right'、'outer'之一，默认为'inner'|
|on|需要连接的列名，必须为两边DataFrame都有的列名，并以left和right中列名的交集作为连接键。|
|left_on|left DataFrame中用作连接键的列|
|right_on|right DataFrame中用作连接键的列|
|left_index|使用left的行索引作为它的连接键|
|right_index|使用right的行索引作为它的连接键|
|sort|通过连接键按字母顺序对合并的数据进行排序，默认为True|
|suffixes|在重叠的情况下，添加到列名后的字符串元组；默认是('data_x','data_y')|
|copy|如果为False，则在某些情况下避免将数据复制到结果数据结构中，默认情况下总是复制|
|indicator|添加一个特殊的列_merge指示每一行的来源；值将根据每行中连接数据的来源分别为'left_only'、'right_only'或'both'|

#### join函数

join是一种实例方法，可以用于按照索引合并。也可以用于多个索引相同或相似但是没有重叠列的DataFrame对象。
该方法传入一个用于合并DataFrame对象，同样可以使用merge函数的部分参数。
该方法也可以传入一个DataFrame列表，这样将会合并列表中的所有DataFrame对象。

#### 沿轴向连接

concat方法可以实现DataFrame与Series对象的拼接。
默认情况下该方法是沿着axis=0方向上生效的。可以通过传递axis=1来改变方向。
通过join='inner'选项可以改变连接的方式。
连接多层索引可以使用keys参数，在轴向上创建一个多层索引。当沿着axis=1方向连接Series时，keys则成为DataFrame的列头。
可以传入字典对象而不是列表对象，这样做将会把字典的键作为keys参数。
concat参数列表：

|参数|描述|
|:--:|:--:|
|objs|需要连接的pandas对象列表或字典|
|axis|连接的轴向，默认为0|
|join|可以是'inner'或者'outer'(默认为'outer')|
|join_axes|用于指定其他n-1轴的特定索引，可以替代内\外连接的逻辑|
|keys|与要连接的对象相关联的值，沿着连接轴形成分层索引。可以是任意值的列表或数组，也可以是元组的数组，也可以是数组的列表|
|levels|在键值传递时，该参数用于指定多层索引的层级|
|names|如果传入了keys和/或levels参数，该参数用于多层索引的层级名称|
|verify_integeity|检查连接对象中的新轴是否重负，如果是，则引发异常；默认为False|
|ignore_index|不沿着连接轴保留索引，而产生一段新的索引|

#### 联合重叠数据conbine_first

conbine_first方法。该方法可以根据传入的对象来填充缺失值。

### 3.重塑和透视

重排列表格型数据有多种基础操作，这些操作被称为重塑或透视。

#### 使用多层索引进行重塑

##### stack

该操作会'旋转'或将列中的数据透视到行，即建堆操作，每一列作为一个堆。
默认情况下，建堆将会过滤出缺失值，所以拆堆是可逆的。通过参数dropna=False可以不过滤缺失值。

##### unstack

该操作会将行中的数据透视到列，即拆堆操作，将每一个堆拆成一列，若数据不匹配，将会引入缺失值。
在DataFrame中，拆堆将会将其作为列的最低层级。

在调用上述方法时，我们可以传入一个字符串指定要堆叠的轴的名称。

#### 时间序列

pivot方法。该方法可以将堆叠格式或者长格式处理为按data与item为索引轴的DataFrame。
该方法传入三个参数，分别为行索引、列索引与数据。
第三个为可选参数，用于指定数据列，若不指定，则会生成一个含多层列的DataFrame。
该方法等价与使用set_index创建分层索引然后调用unstack。

pivot方法的反操作为pandas.melt。该方法将多列合并成一列，生成一个长格式表格。
