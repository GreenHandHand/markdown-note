# python可视化简介

制作提供信息的可视化（有时称为绘图）是数据分析中最重要的任务之一。可视化可能是探索过程中的一部分，例如，帮助识别异常值或所需的数据转换，或者为建模提供一些想法。python中内置了很多的绘图库，这里最主要的绘图库是 [[00-笔记/实践知识/python绘制图表/matplotlib]] 与以它为基础的库。

## matplotlib初步

matplotlib是一个用于生成出版级质量图表（通常是二维的）的桌面绘图包。目的在于在python环境下进行MATLAB风格的绘图。

要在Jupyter notebook中使用交互式绘图，需要执行下面的魔术函数：

```python
%matplotlib notebook
```

在VS Code Jupyter环境下使用交互式绘图，需要执行下面的魔术函数：

```python
%matplotlib widget
```

在使用matplotlib时，我们使用如下导入惯例：

```python
import matplotlib.pyplot as plt
```

### 1.图片与子图

matplotlib所绘制的图片位于Figure对象中，通过plt.figure生成一个Figure对象。这样在jupyter notebook中不会生成任何图片，但在IPython中将会生成一个空白图窗。
figure中可以使用一些参数进行调整：

|参数|描述|
|:--:|:--:|
|name|图表的名字|
|figsize|传入(float,float)，分别为图表的宽与高|
|facecolor|背景色，默认为白色|
|clear|默认为False，是否清除已存在的figure|

你不能使用空白的图片进行绘图。你需要使用add_subplot创建一个或多个子图(subplot)。

注：因为每次绘图都会刷新所有的子图，所以必须将所有的绘图命令放在同一个单元格中。

使用子图网格创建图片是非常常见的任务，使用plt.subplots方法可以直接创建一个子图网格，该方法返回了已生成子图对象的NumPy数组。你可以使用数组索引的方式方便的访问每一张图片。
subplots参数：

|参数|描述|
|:--:|:--:|
|nrows|子图的行数|
|ncols|子图的列数|
|sharex|所有子图使用相同的x轴刻度(调整xlim将会影响所有子图)|
|sharey|所有子图使用相同的y轴刻度(调整ylim将会影响所有子图)|
|subplot_kw|传入add_subplot的相关字参数字典，用于生成子图|
|**fig_kw|在生成图片时使用的额外关键字参数，例如plt.subplots(2,2,figsize=(8,6))|

得到子图后可以使用绘图函数在子图上进行绘图。

#### 调整子图周围的间距

使用subplots_adjust方法更改间距，该方法也可以用作顶层函数。

```python
subplots_adjust(left=None,bottom=None,right=None,
                top=None,wspace=None,hspace=None)
```

*top、bottom、left、right*分别对应左右位置，范围在[0,1]之间，0指左下，1指右上。当值大于1.0的时候子图会超出figure的边界从而显示不全；值不大于1.0的时候，子图会自动分布在一个矩形区域（下图灰色部分）。
要保证left < right, bottom < top，否则会报错。
*wspace、hspace*对应宽度和高度占图表的百分比。

#### 颜色、标记和线类型

绘图函数plot接收带有x和y轴的数组以及一些可选的字符串缩写参数来指明颜色和线类型，线型参数与标记参数必须在颜色参数之后。例如用绿色破折号绘制x对y的线，标记点：

```python
ax.plot(x,y,'go--')
```

也可以显式地指定使用什么：

```python
ax.plot(x,y,linestyle='--',color='g',marker='o')
```

很多颜色缩写被用于常用颜色，也可以通过指定十六进制颜色代码的方式指定任何颜色(例如'#CECECE')。

plot参数：

|参数|描述|
|:--:|:--:|
|x,y|数据|
|[pm]|格式化字符串|
|marker|标记格式|
|linestyle|线型格式|
|color|颜色|
|label|图例名称|

|marker参数|描述|
|:--:|:--:|
|``'.'``|point marker点|
|``','``|pixel marker像素|
|`'o'`|circle marker圆形|
|``'v'``|triangle_down marker下三角形|
|``'^'``|triangle_up marker上三角形|
|``'<'``|triangle_left marker左三角形|
|``'>'``|triangle_right marker右三角形|
|``'1'``|tri_down marker下三角形|
|``'2'``|tri_up marker上三角形|
|``'3'``|tri_left marker左三角形|
|``'4'``|tri_right marker右三角形|
|``'8'``|octagon marker八边形|
|``'s'``|square marker方形|
|``'p'``|pentagon marker五边形|
|``'P'``|plus (filled) marker加号|
|``'*'``|star marker星号|
|``'h'``|hexagon1 marker六边形1|
|``'H'``|hexagon2 marker六边形2|
|``'+'``|plus marker加号|
|``'x'``|x marker|
|``'X'``|x (filled) marker|
|``'D'``|diamond marker菱形|
|``'d'``|thin_diamond marker薄菱形|
|`'|'`|
|``'_'``|hline marker高亮线F|

|linestyle参数|描述|
|:--:|:--:|
|``'-'``|solid line style实线|
|``'--'``|dashed line style虚线|
|``'-.'``|dash-dot line style点划线|
|``':'``|dotted line style点|

|color字母参数|描述|
|:--:|:--:|
|``'b'``|blue蓝色|
|``'g'``|green绿色|
|``'r'``|red 红色|
|``'c'``|cyan 青色|
|``'m'``|magenta洋红|
|``'y'``|yellow黄色|
|``'k'``|black 黑色|
|``'w'``|white 白色|

#### 刻度、标签和图例

pyplot接口函数有两种调用方法：

- 在没有参数的情况下调用，将返回当前的参数值
- 出入参数的情况下调用，并设置参数值

所有的这些方法都会在当前活动的或最近创建的AxesSubplot上生效。

大部分方法都有x、y两个方法，这里仅举例x轴上的方法。

|方法|描述|
|:--:|:--:|
|xlim|设置x绘图范围|
|xticks|设置x刻度位置|
|xticklabels|设置x刻度标签|
|set_title|设置图表名称|
|set_xlabel|设置x轴名称|
|set|批量设置绘图属性，传入以属性为键，参数为值的字典作为参数|
|legend|生成图例，有多个参数指定位置，loc='best'自动选择位置|
|text|添加注释文本|
|arrow|添加注释箭头|
|annotate|同时添加注释和箭头，即在指定坐标绘制箭头注释|

更多详细的方法参考[matplotlib官方网站](https://matplotlib.org/)

## 使用Pandas绘图

pandas自身有很多的内建方法用于可视化。

### 折线图

Series 与 DataFrame 都有一个plot属性用于绘制基本的图形。默认情况下，plot绘制的是折线图。

Series.plot选项列表：

|参数|描述|
|:--:|:--:|
|label|图例标签|
|ax|绘图所用的matplotlib子图对象；如果没有传值，则使用当前活动的matplotlib子图|
|style|传给matplotlib的样式字符串，比如'ko--'|
|alpha|图片不透明度(从0到1)|
|kind|可以是'area','bar','barh','density','hist','kde','line','pie'|
|logy|在y轴上使用对数缩放|
|use_index|使用对象索引刻度标签|
|rot|刻度标签的旋转(0到360)|
|xticks|用于x轴刻度的值|
|yticks|用于y轴刻度的值|
|xlim|x轴范围|
|ylim|y轴范围|
|grid|展示轴网格|

DataFrame.plot选项：

|参数|描述|
|:--:|:--:|
|subplots|将DataFrame的每一列绘制在独立的子图|
|sharex|如果subplots=True，则共享相同的x轴、刻度和范围|
|sharey|如果subplots=True，则共享相同的y轴|
|figsize|用于生成图片的尺寸元组|
|title|标题字符串|
|legend|添加子图图例（默认为True）|
|sort_columns|按字母顺序绘制各列，默认情况下使用已有的列顺序|

### 柱状图

plot.bar与plot.barh()分别可以绘制垂直和水平的柱状图。
默认情况下DataFrame中将每一行的值分组到并排的柱子中的一组。可以通过stacked=True绘制堆积柱状图。

### 直方图和密度图

plot.hist()用于绘制直方图。使用plot.density()绘制密度图。

## 使用seaborn绘图

```python
import seaborn as sns
```

seaborn包是一个建立在matplotlib上的根据美观的绘图库。它简化列很多常用的可视化类型生成。
导入seaborn会修改默认的matplotlib配色方案和绘图样式。即使不使用seaborn包进行绘图，也可以通过导入seaborn来为通用的matplotlib图表提供更好的视觉美观度。

- 柱状图：barplot
- 直方图和密度图：distplot
- 回归/散点图：regplot
- 成对图/散点图矩阵：pairplot
- 分面网格/分组绘图：factorplot
