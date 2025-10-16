# matplotlib

```python
import numpy as np
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
```

Matplotlib是一个Python 2D绘图库，能够以多种硬拷贝格式和跨平台的交互式环境生成出版物质量的图形，用来绘制各种静态，动态，交互式的图表。

Matplotlib可用于Python脚本，Python和IPython Shell、Jupyter notebook，Web应用程序服务器和各种图形用户界面工具包等。

Matplotlib是Python数据可视化库中的泰斗，它已经成为python中公认的数据可视化工具，我们所熟知的pandas和seaborn的绘图接口其实也是基于matplotlib所作的高级封装。

本节中介绍了 Matplotlib 中的基本概念，要绘制更加美观的图表，参照 [[00_Inbox/实践知识/python绘制图表/matplotlib色彩]]。

> **[matplotlib官方文档](https://matplotlib.org/)**


## Figure对象

Matplotlib的图像是画在figure（如windows，jupyter窗体）上的，每一个figure又包含了一个或多个axes（一个可以指定坐标系的子区域）。最简单的创建figure以及axes的方式是通过 `pyplot.subplots`命令，创建axes以后，可以使用 `Axes.plot`绘制最简易的折线图。

与Matlab类似，如果没有指定axes，matplotlib会为用户自动创建一个axes。

> **figure的组成**

一个完整的matplotlib图像通常会包括以下四个层级，这些层级也被称为容器（container）。在matplotlib的世界中，我们将通过各种命令方法来操纵图像中的每一个部分，从而达到数据可视化的最终效果，一副完整的图像实际上是各类子元素的集合。

![[Assets/1675166629375.png]]

- Figure：顶层级，用来容纳所有绘图元素
- Axes：matplotlib宇宙的核心，容纳了大量元素用来构造一幅幅子图，一个figure可以由一个或多个子图组成
- Axis：axes的下属层级，用于处理所有和坐标轴，网格有关的元素
- Tick：axis的下属层级，用来处理所有和刻度有关的元素

> matplotlib的绘图接口

matplotlib提供了两种绘图的接口：

- 显式创建figure和axes，在上面调用绘图方法，也被称为OO模式（object-oriented style)
- 依赖pyplot自动创建figure和axes，并绘图

## 通用绘图模板

由于matplotlib的知识点非常繁杂，在实际使用过程中也不可能将全部API都记住，很多时候都是边用边查。因此这里提供一个通用的绘图基础模板，任何复杂的图表几乎都可以基于这个模板骨架填充内容而成。

```python
# step1 准备数据
x = np.linspace(0, 2, 100)
y = x**2

# step2 设置绘图样式，这一步不是必须的，样式也可以在绘制图像是进行设置
mpl.rc('lines', linewidth=4, linestyle='-.')

# step3 定义布局
fig, ax = plt.subplots()  

# step4 绘制图像
ax.plot(x, y, label='linear')  

# step5 添加标签，文字和图例
ax.set_xlabel('x label') 
ax.set_ylabel('y label') 
ax.set_title("Simple Plot")  
ax.legend() ;
```

## matplotlib图像设置

matplotlib的原理或者说基础逻辑是，用Artist对象在画布(canvas)上绘制(Render)图形。

所以matplotlib有三个层次的API：

- `matplotlib.backend_bases.FigureCanvas` 代表了绘图区，所有的图像都是在绘图区完成的
- `matplotlib.backend_bases.Renderer` 代表了渲染器，可以近似理解为画笔，控制如何在 FigureCanvas 上画图。
- `matplotlib.artist.Artist` 代表了具体的图表组件，即调用了Renderer的接口在Canvas上作图。

前两者处理程序和计算机的底层交互的事项，第三项Artist就是具体的调用接口来做出我们想要的图，比如图形、文本、线条的设定。所以通常来说，我们95%的时间，都是用来和matplotlib.artist.Artist类打交道的。

### Artist对象

Artist有两种类型：primitives与containers。

- primitive：基本要素，包含在绘图区作图要用到标准图形对象，如**曲线Line2D，文字text，矩形Rectangle，图像image**等。
- container是容器，即用来装基本要素的地方，包括**图形figure、坐标系Axes和坐标轴Axis**。

![[Assets/1675174939837.png]]

可视化常见的artist类见下表：

|Axes helper method|Artist|container|
|--|--|--|
|bar - 柱状图|Rectangle|ax.pathches|
|errorbar - 误差线|Line2D and Rectangle|qx.lines and ax.patches|
|fill - shared area|Polygon|ax.patches|
|hist - 直方图|Rectangle|ax.patches|
|imshow - image data|AxesImage|ax.image|
|plot - xy plot（直线）|Line2D|ax.lines|
|scatter - 散点图|PolyCollection|ax.collections|

### 基本元素 - primitives

各容器中可能会包含多种**基本要素-primitives**

#### Line2D

在matplotlib中曲线的绘制，主要就是通过类matplotlib.lines.Line2D来完成。

在matplotlib中线-line的含义：它表示可以连接所有顶点的实线样式，也可以是每个顶点的标记。此外，这条线也会受到绘画风格的影响。

构造函数：

```python
class matplotlib.lines.Line2D(
    xdata, ydata, linewidth=None, linestyle=None, color=None,
    marker=None, markersize=None, markeredgewidth=None,
    markeredgecolor=None, markerfacecolor=None,
    markerfacecoloralt='none', fillstyle=None,
    antialiased=None, dash_capstyle=None,
    solid_capstyle=None, dash_joinstyle=None,
    solid_joinstyle=None, pickradius=5, drawstyle=None,
    markevery=None, \*\*kwargs
)
```

常用参数列表：

|参数|描述|
|--|--|
|xdata|需要绘制的line中点在x轴上的取值，若忽略，<br/>则默认为range(1, len(ydata) + 1)|
|ydata|需要绘制的line中点在y轴上的取值|
|linewidth|线条的宽度|
|linestyle|线型|
|color|线条的颜色|
|marker|点的标记，详细参考[markers API](https://matplotlib.org/api/markers_api.html#module-matplotlib.markers "https://matplotlib.org/api/markers_api.html#module-matplotlib.markers")|
|markersize|标记的size|

其他参数参考[Line2D官方文档](https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html "https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html")。

> 设置Line2D的属性

可以通过下面三种方法设置Line2D的属性：

1. 直接在plot函数中设置
2. 通过获得线对象，对线对象进行设置
3. 获得线属性，使用setp函数设置

```python
x = range(0, 5)
y = [2, 5, 7, 8, 10]

# 1) 直接在plot()函数中设置
plt.plot(x, y, linewidth=10)

# 2) 通过获得线对象，对线对象进行设置
line, = plt.plot(x, y, '-') #返回一个Line2D的列表
line.set_antialiased(False) #关闭抗锯齿功能

# 3) 获得线属性，使用setp()函数设置
lines = plt.plot(x, y)
plt.setp(lines, color='r', linewidth=10)
```

> 绘制lines

- plot方法绘制
  ```python
  # 1. plot方法绘制
  x = range(0,5)
  y1 = [2,5,7,8,10]
  y2= [3,6,8,9,11]
  
  fig,ax= plt.subplots()
  ax.plot(x,y1)
  ax.plot(x,y2)
  print(ax.lines); # 通过直接使用辅助方法画线，打印ax.lines后可以看到在matplotlib在底层创建了两个Line2D对象
  ```
- Line2D对象方法绘制
  ```python
  # 2. Line2D对象绘制
  
  x = range(0,5)
  y1 = [2,5,7,8,10]
  y2= [3,6,8,9,11]
  fig,ax= plt.subplots()
  lines = [Line2D(x, y1), Line2D(x, y2,color='orange')]  # 显式创建Line2D对象
  for line in lines:
      ax.add_line(line) # 使用add_line方法将创建的Line2D添加到子图中
  ax.set_xlim(0,4)
  ax.set_ylim(2, 11);
  ```

> 绘制误差线

pyplot中可以通过errorbar类来绘制误差线，它的构造函数为：

```python
matplotlib.pyplot.errorbar(
    x, y, yerr=None, xerr=None, fmt='',
    ecolor=None, elinewidth=None, capsize=None, barsabove=False,                         lolims=False, uplims=False, xlolims=False, xuplims=False, 
    errorevery=1, capthick=None, *, data=None, \*\*kwargs
)
```

常用参数：

|参数|描述|
|--|--|
|x|需要绘制的line中点在x轴上的取值|
|y|需要绘制的line中点在y轴上的取值|
|xerr|指点x轴水平的误差|
|yerr|指定y轴水平的误差|
|fmt|指定折线图中某点的颜色、形状、线条风格，例如'co--'|
|ecolor|指定error bar的颜色|
|elinewidth|指定error bar的线条宽度|

绘制效果如下：

```python
fig = plt.figure()
x = np.arange(10)
y = 2.5 * np.sin(x / 20 * np.pi)
yerr = np.linspace(0.05, 0.2, 10)
plt.errorbar(x,y+3,yerr=yerr,fmt='o-',ecolor='r',elinewidth=2);
```

![[Assets/1675177873598.png]]

#### patches

matplotlib.patches.Patch类是二维图形类，并且它是众多二维图形的父类，它的所有子类见[matplotlib.patches API](https://matplotlib.org/stable/api/patches_api.html "https://matplotlib.org/stable/api/patches_api.html") ，
Patch类的构造函数：

```python
Patch(
    edgecolor=None, facecolor=None, color=None, linewidth=None, 
    linestyle=None, antialiased=None, hatch=None, fill=True, 
    capstyle=None, joinstyle=None, \*\*kwargs
)
```

##### Rectangle - 矩形

**Rectangel**矩形类在官网中的定义是： 通过锚点xy及其宽度和高度生成。 Rectangle本身的主要比较简单，即xy控制锚点，width和height分别控制宽和高。它的构造函数：

```python
class matplotlib.patches.Rectangle(xy, width, height, angle=0.0, \*\*kwargs)
```

在实际中我们常见的矩形图有直方图（hist）与条形图（bar）。

> 直方图 - hist

```python
matplotlib.pyplot.hist(
    x,bins=None,range=None, density=None, bottom=None, histtype='bar', 
    align='mid', log=False, color=None, label=None, stacked=False, normed=None
)
```

常用参数：

|参数|描绘|
|--|--|
|x|数据集，最终直方图将对数据集进行统计|
|bins|统计的区间分布|
|range|tuple，显示的区间，range在没有给出bins时生效|
|density|bool，默认为false，显示的是频数统计结果，为True则<br/>显示频率统计结果，这里需要注意，频率统计结果=<br/>区间数目/(总数 * 区间宽度)，和normed效果一致，<br/>官方推荐使用density|
|histtype|可选{'bar'，'barstacked'，'step'，'stepfilled'}之一，<br/>默认为bar，推荐使用默认配置，step使用的是梯状，<br/>stepfill则会对梯状内部进行填充，效果与bar类似|
|align|可选{'left'，'mid'，'right'}之一，默认为'mid'，控制<br/>柱状图的水平分布，left或者right，会有部分空白区域，<br/>推荐使用默认|
|log|bool，默认False，即y轴坐标是否选择指数刻度|
|stacked|bool，默认为False，是否为堆积状图|

> bar - 柱状图

```python
matplotlib.pyplot.bar(left, height, alpha=1, width=0.8, color=, edgecolor=, label=, lw=3)
```

常用参数：

|参数|描述|
|--|--|
|left|x轴的位置序列，一般采用range函数产生一个序列，<br/>但是有时候也可以是字符串|
|height|y轴的数组序列，也就是柱形图的高度，一般就是<br/>我们需要展示的数据|
|alpha|透明度，值越小越透明|
|width|为柱形图的宽度，一般这是为0.8即可|
|color/<br/>facecolor|柱形图填充的颜色|
|edgecolor|图形边缘的颜色|
|label|解释每个图像代表的含义，这个参数是为legend函数<br/>做铺垫的，表示该次bar的标签|

可以使用bar或者Rectangle矩形类绘制柱状图。

> Polygon - 多边形

matplotlib.patches.Polygon类是多边形类。它的构造函数：

```python
class matplotlib.patches.Polygon(xy, closed=True, \*\*kwargs)
```

参数含义：

|参数|描述|
|--|--|
|xy|一个N×2的numpy array，为多边形的顶点|
|closed|True则指定多边形将起点与终点重合从而显示<br/>关闭多边形|

matplotlib.patches.Polygon类中常用的是fill类，它是基于xy绘制一个填充的多边形，它的定义：

```python
matplotlib.pyplot.fill(\*args, data=None, \*\*kwargs)
```

参数说明：关于x、y和color的序列，其中color是可选的参数，每个多边形都是由其节点的x和y位置列表定义的，后面可以选择一个颜色说明符。您可以通过提供多个x、y、[颜色]组来绘制多个多边形。

```python
x = np.linspace(0,5*np.pi, 1000)
y1 = np.sin(x)
y2 = - 0.5 * np.sin(x)
fig = plt.figure()
plt.fill(x, y1, color = 'g', alpha = 0.3)
plt.fill(x, y2, color = 'g', alpha = 0.3)
```

![[Assets/1675234694027.png]]

> Wedge - 楔形

matplotlib.patches.Wedge类是楔型类。其基类是matplotlib.patches.Patch，它的构造函数：

```python
class matplotlib.patches.Wedge(center, r, theta1, theta2, width=None, \*\*kwargs)
```

一个Wedge-楔型 是以坐标x,y为中心，半径为r，从θ1扫到θ2(单位是度)。
如果宽度给定，则从内半径r -宽度到外半径r画出部分楔形。wedge中比较常见的是绘制饼状图。

matplotlib.pyplot.pie的用法：

```python
matplotlib.pyplot.pie(
    x, explode=None, labels=None, colors=None, 
    autopct=None, pctdistance=0.6, shadow=False, 
    labeldistance=1.1, startangle=0, radius=1, 
    counterclock=True, wedgeprops=None, textprops=None, 
    center=0, 0, frame=False, rotatelabels=False, *, 
    normalize=None, data=None
)
```

#### collections

collections类是用来绘制一组对象的集合，collections有许多不同的子类，如RegularPolyCollection, CircleCollection, Pathcollection, 分别对应不同的集合子类型。其中比较常用的就是散点图，它是属于PathCollection子类，scatter方法提供了该类的封装，根据x与y绘制不同大小或颜色标记的散点图。 它的构造方法：

```python
Axes.scatter(
    self, x, y, s=None, c=None, marker=None, cmap=None, 
    norm=None, vmin=None, vmax=None, alpha=None, 
    linewidths=None, verts=, edgecolors=None, *, 
    plotnonfinite=False, data=None, \*\*kwargs
)
```

```
主要参数：
```

|参数|描述|
|--|--|
|x|数据点x轴的位置|
|y|数据点y轴的位置|
|s|尺寸大小|
|c|可以是单个颜色格式的字符串，也可以是一系列颜色|
|marker|标记的类型|

#### images

images是matplotlib中绘制image图像的类，其中最常用的imshow可以根据数组绘制成图像，它的构造函数：

```python
class matplotlib.image.AxesImage(
    ax, cmap=None, norm=None, interpolation=None, 
    origin=None, extent=None, filternorm=True, 
    filterrad=4.0, resample=False, \*\*kwargs
)
```

imshow根据数组绘制图像

```python
matplotlib.pyplot.imshow(
    X, cmap=None, norm=None, aspect=None, 
    interpolation=None, alpha=None, vmin=None, 
    vmax=None, origin=None, extent=None, shape=, 
    filternorm=1, filterrad=4.0, imlim=, resample=None, 
    url=None, *, data=None, \*\*kwargs
）
```

使用imshow画图需要先传入一个数组，数组对应的是空间内的像素位置和像素点的位置，interpolation参数可以设置不同的插值方法。具体效果如下：

![[Assets/1675236803394.png]]

### 对象容器

容器会包含一些primitive，并且容器还有它自身的属性。
比如 Axes Artist，它是一种容器，它包含了很多primitives，比如 Line2D，Text；同时，它也有自身的属性，比如xscal，用来控制X轴是 linear还是log的。

#### Figure

matplotlib.figure.Figure是Artist最顶层的container对象容器，它包含了图表中的所有元素。一张图表的背景就是在Figure.patch的一个矩形Rectangle。

当我们向图表添加Figure.add_subplot()或者Figure.add_axes()元素时，这些都会被添加到Figure.axes列表中。

由于Figure维持了current axes，因此不能通过Figure.axes列表来添加删除元素，而是通过Figure.add_subplot()、Figure.add_axes()来添加元素，通过Figure.delaxes()来删除元素。但是可以通过迭代或者访问Figure.axes中的Axes，然后修改这个Axes的属性。

Figure也有自己的text，line，patch，image。可以直接通过add primitive语句直接添加。但是Figure默认的坐标系是以像素为单位，可能需要转换为figure坐标系：(0, 0)表示左下，(1, 1)表示右上。

> Figure常用属性

|属性|描述|
|--|--|
|Figure.patch|Figure的背景矩形|
|Figure.axes|一个Axes实例的列表（包括subplot）|
|Figure.images|一个FigureImages.patch列表|
|Figure.lines|一个line2D实例的列表|
|Figure.legends|一个Figure Legend实例的列表<br/>（不同于Axes.legends）|
|Figure.texts|一个Figure Text实例列表|

#### Axes容器

matplotlib.axes.Axes是matplotlib的核心。大量的用于绘图的Artist存放在它内部，并且它有许多辅助方法来创建和添加Atrist给它自己，而且它也有许多赋值方法来访问和修改这些Artist。

Subplot就是一个特殊的Axes，其实例是位于网格中某个区域的Subplot实例。其实你也可以在任意区域创建Axes，通过**Figure.add_axes([left,bottom,width,height])**来创建一个任意区域的Axes，其中left,bottom,width,height都是[0—1]之间的浮点数，他们代表了相对于Figure的坐标。

你也可以使用Axes的辅助方法.add_line()和.add_patch()方法来直接添加。

Axes容器常见属性：

|属性|描述|
|--|--|
|artists|Aritst实例列表|
|patch|Axes所在矩形实例|
|collections|Collection实例|
|images|Axes图像|
|legends|Legend实例|
|lines|Line2D实例|
|patches|Patch实例|
|texts|Text实例|
|xaxis|matplotlib.axis.XAxis实例|
|yaxis|matplotlib.axis.YAxis实例|

#### Axis容器

matplotlib.axis.Axis实例处理tick line、grid line、tick label以及axis label的绘制，它包括坐标轴上的刻度线、刻度label、坐标网格、坐标轴标题。通常你可以独立的配置y轴的左边刻度以及右边的刻度，也可以独立地配置x轴的上边刻度以及下边的刻度。

刻度包括主刻度和次刻度，它们都是Tick刻度对象。

Axis也存储了用于自适应，平移以及缩放的data_interval和view_interval。它还有Locator实例和Formatter实例用于控制刻度线的位置以及刻度label。

刻度是动态创建的，只有在需要创建的时候才创建（比如缩放的时候）。

使用Axes.tick_params()可以修改坐标轴的样式，下面列出了一些参数：

|参数|描绘|
|--|--|
|axis|{'x', 'y', 'both'}，样式应用的坐标轴|
|which|{'major', 'minor', 'both'}，样式应用的主次坐标轴|
|reset|bool，是否重设坐标|
|direction|{'in', 'out', 'inout'}，坐标轴的方向|
|length|float，长度|
|width|float，宽度|
|color|color，颜色|
|pad|float，标签与坐标轴的距离|
|labelsize|float or str，坐标轴字体大小|
|labelcolor|color，坐标轴文字颜色|
|colors|color，坐标轴与文字颜色|
|zorder|float，坐标轴的zorder，即显示图层|
|bottom<br/>top<br/>left<br/>right|bool，图的位置是否显示坐标轴|
|labelbottom<br/>labeltop<br/>labelleft<br/>labelright|bool，是否绘制坐标轴标签|
|labelrotation|float，坐标轴标签旋转角度|
|grid_color|color，网格线的颜色|
|grid_alpha|float，网格线的透明度|
|grid_linewidth|float，网格线的宽度|
|grid_linestyle|str，网格线的样式|

Axis也提供了一些辅助方法来获取刻度文本、刻度线位置等。

```python
axis = ax.xaxis # axis为X轴对象
axis.get_ticklocs()     # 获取刻度线位置
axis.get_ticklabels()   # 获取刻度label列表(一个Text实例的列表）。 可以通过minor=True|False关键字参数控制输出minor还是major的tick label。
axis.get_ticklines()    # 获取刻度线列表(一个Line2D实例的列表）。 可以通过minor=True|False关键字参数控制输出minor还是major的tick line。
axis.get_data_interval()# 获取轴刻度间隔
axis.get_view_interval()# 获取轴视角（位置）的间隔
```

使用Axes.spines可以获得Axes的上下左右四个坐标轴，可以以字典的形式访问它们：

```python
xaxis = Axes.spines['top']
```

#### Tick容器

matplotlib.axis.Tick是从Figure到Axes到Axis到Tick中最末端的容器对象。

Tick包含了tick、grid line实例以及对应的label。

所有的这些都可以通过Tick的属性获得，常见的tick属性有：

|属性|描述|
|--|--|
|Tick.tick1line|Line2D实例，左侧的y轴，下侧的x轴|
|Tick.tick2line|Line2D实例，右侧的y轴，上侧的x轴|
|Tick.gridline|Line2D实例|
|Tick.label1|Text实例，同line|
|Tick.label2|Text实例，同line|

## matplotlib布局调整

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   #用来正常显示负号
```

### 子图

#### 使用plt.subplots绘制均匀状态下的子图

> 直角坐标系

返回元素分别是画布和子图构成的列表，不传入时默认值都为1

```python
fig, axes = plt.subplots() #这样获得的将会获得figure对象与axes对象列表
```

figsize参数可以指定整个画布的大小

sharex和sharey分别表示是否共享横轴和纵轴刻度

tight_layout函数可以调整子图的相对大小是字符不会重叠

> 当子图只有一行时，获得的axes列表是一维的，其余情况是二维的，第一个数字是行，第二个是列。

subplots是基于OO模式的写法，显示地创建一个或多个axes对象，然后再对应的子图对象上进行绘图操作。

还可以通过subplot函数，每次指定位置新建一个子图，并且在之后的绘图都指向该图。该方法本质上与add_subplot相同。

在调用这两个方法时一般需要传入三个数字，分别代表总行数、总列数和当前使用的子图的index。

> 极坐标系

可以通过projection方法创建极坐标系下的图表。

```python
N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2
colors = theta

plt.subplot(projection='polar')
plt.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75);
```

![[Assets/1675259490324.png]]

#### 使用GridSpec绘制非均匀子图

所谓非均匀包含两层含义，第一是指图的比例大小不同但是没有跨行或者跨列，第二是指图为跨行或者跨列状态。

利用add_gridspec可以指定相对宽度比例width_ratios和相对高度比例参数height_ratios

```python
fig = plt.figure(figsize=(10, 4))
spec = fig.add_gridspec(nrows=2, ncols=5, width_ratios=[1,2,3,4,5], height_ratios=[1,3])
fig.suptitle('样例2', size=20)
for i in range(2):
    for j in range(5):
        ax = fig.add_subplot(spec[i, j])
        ax.scatter(np.random.randn(10), np.random.randn(10))
        ax.set_title('第%d行，第%d列'%(i+1,j+1))
        if i==1: ax.set_xlabel('横坐标')
        if j==0: ax.set_ylabel('纵坐标')
fig.tight_layout()
```

![[Assets/1675342874871.png]]

上面的方法使用了spec[i, j]的用法，实际上通过切片就可以实现子图合并的效果从而达到跨图的功能。即通过spec切片的写法可以实现将不同的子图合并的效果：

```python
fig = plt.figure(figsize=(10, 4))
spec = fig.add_gridspec(nrows=2, ncols=6, width_ratios=[2,2.5,3,1,1.5,2], height_ratios=[1,2])
fig.suptitle('样例3', size=20)
# sub1
ax = fig.add_subplot(spec[0, :3])
ax.scatter(np.random.randn(10), np.random.randn(10))
# sub2
ax = fig.add_subplot(spec[0, 3:5])
ax.scatter(np.random.randn(10), np.random.randn(10))
# sub3
ax = fig.add_subplot(spec[:, 5])
ax.scatter(np.random.randn(10), np.random.randn(10))
# sub4
ax = fig.add_subplot(spec[1, 0])
ax.scatter(np.random.randn(10), np.random.randn(10))
# sub5
ax = fig.add_subplot(spec[1, 1:5])
ax.scatter(np.random.randn(10), np.random.randn(10))
fig.tight_layout()
```

![[Assets/1675343050845.png]]

### 子图上的方法

常用的子图方法：

> 绘制直线

常用的绘制直线的方法有：**axhline, axvline, axline(水平，垂直，任意方向)**

- axhline传入y坐标、开始x坐标、结束x坐标
- axvline传入x坐标、开始y坐标、结束y坐标
- axline传入起点、终点的坐标元组

```python
fig, ax = plt.subplots(figsize=(4,3))
ax.axhline(0.5,0.2,0.8)
ax.axvline(0.5,0.2,0.8)
ax.axline([0.3,0.3],[0.7,0.7]);
```

> 添加网格

使用grid可以添加灰色网格、

> 设置坐标轴的规度

使用set_xscale可以设置坐标轴的规度（指对数坐标等）

```python
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for j in range(2):
    axs[j].plot(list('abcd'), [10**i for i in range(4)])
    if j == 0:
        axs[j].set_yscale('log')
    else:
        pass
fig.tight_layout()
```

![[Assets/1675343751605.png]]

### matplotlib图例与文字

#### Figure与Axes上的文本

Matplotlib具有广泛的文本支持，包括对数学表达式的支持、对栅格和矢量输出的TrueType支持、具有任意旋转的换行分隔文本以及Unicode支持。

##### 文本API简介

|pyplot API|OO API|描述|
|--|--|--|
|text|text|在子图axes的任意位置添加文本|
|annotate|annotate|在子图axes的任意位置添加注释，包<br/>含指向性的箭头|
|xlabel|set_xlabel|为子图axes添加x轴标签|
|ylabel|set_ylabel|为子图axes添加y轴标签|
|title|set_title|为子图添加标题|
|figtext|text|在画布figure的任意位置添加文本|
|suptitle|suptitle|为画布figure添加标题|

下面一个综合例子：

```python
fig = plt.figure()
ax = fig.add_subplot()


# 分别为figure和ax设置标题，注意两者的位置是不同的
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
ax.set_title('axes title')

# 设置x和y轴标签
ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

# 设置x和y轴显示范围均为0到10
ax.axis([0, 10, 0, 10])

# 在子图上添加文本
ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

# 在画布上添加文本，一般在子图上添加文本是更常见的操作，这种方法很少用
fig.text(0.4,0.8,'This is text for figure')

ax.plot([2], [1], 'o')
# 添加注解
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),arrowprops=dict(facecolor='black', shrink=0.05));
```

![[Assets/1675351032404.png]]

#### text - 子图上的文本

text的调用方式为 `Axes.text(x, y, s, fontdicr=None, \*\*kwargs)`其中x，y为文本出现的位置，默认情况下即当前坐标系下的坐标值，s为文本内容。

- fontdict是可选参数，用于覆盖默认的文本属性
- \*\*kwargs为关键词参数，也可以用于传入文本样式参数

> fontdict与\*\*kwargs参数都可以用于调整呈现的文本样式，最终效果是一样的，不仅text方法，其他文本方法如set_slabel，set_title等同样使用这种方法修改样式。下面是一个示例：
> 
> ```python
> fig = plt.figure(figsize=(10,3))
> axes = fig.subplots(1,2)
> 
> # 使用关键字参数修改文本样式
> axes[0].text(0.3, 0.8, 'modify by \*\*kwargs', style='italic',
>         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10});
> 
> # 使用fontdict参数修改文本样式
> font = {'bbox':{'facecolor': 'red', 'alpha': 0.5, 'pad': 10}, 'style':'italic'}
> axes[1].text(0.3, 0.8, 'modify by fontdict', fontdict=font);
> ```

matplotlib所有样式参考[官方模板](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html#matplotlib.axes.Axes.text "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html#matplotlib.axes.Axes.text")。

下面列举常用的参数方便查询：

|参数|描述|
|--|--|
|alpha|float / None： 透明度，越接近0越透明|
|backgroundcolor|color：文本的背景颜色|
|bbox|dict with properties for<br/>patches.FancyBboxPatch<br/>用来设置text的box外框|
|color / C|color：字体的颜色|
|fontfamily / family|{FONTNAME, 'serif', 'sans-serif', 'cursive'<br/>,'fantasy', 'monospace'}字体的类型|
|fontsize / size|float / {'xx-small', 'x-small', 'small',<br/>'medium', 'large', 'x-large', 'xx-large'}<br/>字体大小|
|fontstyle / style|{'normal', 'italic', 'oblique'}字体样式，<br/>如倾斜|
|fontweight / weight|{a numeric value in range 0-1000,<br/>'ultralight', 'light', 'normal', 'regular', <br/>'book', 'medium', 'roman', 'semibold',<br/> 'demibold', 'demi', 'bold', 'heavy', <br/>'extra bold', 'black'} 文本粗细|
|horizontalalignment / ha|{'center', 'right', 'left'}<br/>选择文本左对齐右对齐还是居中对齐|
|linespacing|float文本间距|
|rotation|float / {'vertical', 'horizontal'}<br/>指text逆时针旋转的角度，'horizontal'<br/>等于0，'vertical'等于90度|
|verticalalignment / va|{'center', 'top', 'bottom', 'baseline',<br/>'center_baseline'}<br/>文本在垂直角度的对齐方式|

#### xlabel 和 ylabel - 子图的x，y轴标签

xlabel的调用方式为Axes.set_xlabel(xlabel, fontdict=None, labelpad =None, \*, loc=None, \*\*kwargs)

ylabel同xlabel

- xlabel：标签内容
- fontdict与\*\*kwargs：用于修改样式
- labelpad：标签与坐标轴的距离，默认为4
- loc：标签的位置，可选值为'left', 'center', 'right'，默认居中

#### title和suptitle - 子图与画布的标题

title的调用方式为Axes.set_title(label, fontdict=None, loc=None, pad=None, \*, y=None, \*\*kwargs)

- label：标签内容
- fontdict与\*\*kwargs：用于修改样式
- pad：标签偏离图表顶部的距离，默认为6
- y：title所在子图垂向的位置，默认为1，即title位于子图的顶部。

suptitle的调用方式为figure.suptitle(t, \*\*kwargs)，其中t为画布的标题内容。

#### annotate - 子图的注解

annotate的调用方式为Axes.annotate(text, xy, \*args, \*\*kwargs)

- text为注解内容
- xytext：注解文字的坐标
- xycoords：用来定义xy参数的坐标
- textcoords：用来定义xytext参数的坐标系
- arrowprops：用来定义指向箭头的样式

由于annotate的参数过于复杂，详细可以参考[官方文档](https://matplotlib.org/stable/tutorials/text/annotations.html#plotting-guide-annotation "https://matplotlib.org/stable/tutorials/text/annotations.html#plotting-guide-annotation")。

#### 字体的属性设置

字体设置一般有全局字体设置和自定义局部字体设置两种方法。

[常用中文字体的英文名称](https://www.cnblogs.com/chendc/p/9298832.html "https://www.cnblogs.com/chendc/p/9298832.html")

```python
#该block讲述如何在matplotlib里面，修改字体默认属性，完成全局字体的更改。
plt.rcParams['font.sans-serif'] = ['SimSun']    # 指定默认字体为新宋体。
plt.rcParams['axes.unicode_minus'] = False      # 解决保存图像时 负号'-' 显示为方块和报错的问题。
```

```python
#局部字体的修改方法1
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(x, label='小示例图标签')

# 直接用字体的名字
plt.xlabel('x 轴名称参数', fontproperties='Microsoft YaHei', fontsize=16)         # 设置x轴名称，采用微软雅黑字体
plt.ylabel('y 轴名称参数', fontproperties='Microsoft YaHei', fontsize=14)         # 设置Y轴名称
plt.title('坐标系的标题',  fontproperties='Microsoft YaHei', fontsize=20)         # 设置坐标系标题的字体
plt.legend(loc='lower right', prop={"family": 'Microsoft YaHei'}, fontsize=10) ;   # 小示例图的字体设置
```

### Tick上的文本

设置tick（刻度）和ticklabel（刻度标签）也是可视化中经常需要操作的步骤，matplotlib既提供了自动生成刻度和刻度标签的模式（默认状态），同时也提供了许多让使用者灵活设置的方式。

#### 简单模式

可以使用axis的set_ticks方法手动设置标签位置，使用axis的set_ticklabels方法手动设置标签格式：

```python
x = np.linspace(0, 5, 100)
y = np.cos(2 * np.pi * x) * np.exp(-x)

# 使用axis的set_ticks方法手动设置标签位置的例子，该案例中由于tick设置过大，所以会影响绘图美观，不建议用此方式进行设置tick
fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)
axs[0].plot(x, y)
axs[1].plot(x, y)
axs[1].xaxis.set_ticks(np.arange(0., 10.1, 2.));
```

![[Assets/1675355968938.png]]

```python
x = np.linspace(0, 5, 100)
y = np.cos(2 * np.pi * x) * np.exp(-x)

# 使用axis的set_ticklabels方法手动设置标签格式的例子
fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)
axs[0].plot(x, y)
axs[1].plot(x, y)
ticks = np.arange(0., 8.1, 2.)
tickla = [f'{tick:1.2f}' for tick in ticks]
axs[1].xaxis.set_ticks(ticks)
axs[1].xaxis.set_ticklabels(tickla);
```

![[Assets/1675356013615.png]]

```python
fig, axs = plt.subplots(2, 1, figsize=(6, 4), tight_layout=True)
x1 = np.linspace(0.0, 6.0, 100)
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
axs[0].plot(x1, y1)
axs[0].set_xticks([0,1,2,3,4,5,6])

axs[1].plot(x1, y1)
axs[1].set_xticks([0,1,2,3,4,5,6])#要将x轴的刻度放在数据范围中的哪些位置
axs[1].set_xticklabels(['zero','one', 'two', 'three', 'four', 'five','six'],#设置刻度对应的标签
                   rotation=30, fontsize='small')#rotation选项设定x刻度标签倾斜30度。
axs[1].xaxis.set_ticks_position('bottom')#set_ticks_position()方法是用来设置刻度所在的位置，常用的参数有bottom、top、both、none
print(axs[1].xaxis.get_ticklines());
```

![[Assets/1675356111594.png]]

#### Tick Locators and Formatters

除了上述的简单方式，还可以使用Tick Locators and Formatters完成对于刻度位置和刻度标签的设置。其中Axis.set_major_locator和Axis.set_minor_locator方法用来设置标签的位置，Axis.set_major_formatter和Axis.set_minor_formatter方法用来设置标签的格式。这种方法的好处是不用显式地列举出刻度值列表。

set_major_formatter和set_minor_formatter这两个formatter格式命令可以接受字符串格式(matplotlib.ticker.StrMethodFormatter)或函数参数(matplotlib.ticker.FuncFormatter)来设置刻度值的格式。

##### Tick Formatters

```python
# 接收字符串格式的例子
fig, axs = plt.subplots(2, 2, figsize=(8, 5), tight_layout=True)
for n, ax in enumerate(axs.flat):
    ax.plot(x1*10., y1)

formatter = matplotlib.ticker.FormatStrFormatter('%1.1f')
axs[0, 1].xaxis.set_major_formatter(formatter)

formatter = matplotlib.ticker.FormatStrFormatter('-%1.1f')
axs[1, 0].xaxis.set_major_formatter(formatter)

formatter = matplotlib.ticker.FormatStrFormatter('%1.5f')
axs[1, 1].xaxis.set_major_formatter(formatter);
```

![[Assets/1675356528990.png]]

```python
# 接收函数的例子
def formatoddticks(x, pos):
    """Format odd tick positions."""
    if x % 2:
        return f'{x:1.2f}'
    else:
        return ''

fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
ax.plot(x1, y1)
ax.xaxis.set_major_formatter(formatoddticks);
```

![[Assets/1675356583920.png]]

##### Tick Locators

在普通的绘图中，我们可以直接通过上图的set_ticks进行设置刻度的位置，缺点是需要自己指定或者接受matplotlib默认给定的刻度。当需要更改刻度的位置时，matplotlib给了常用的几种locator的类型。如果要绘制更复杂的图，可以先设置locator的类型，然后通过axs.xaxis.set_major_locator(locator)绘制即可

```python
locator=plt.MaxNLocator(nbins=7)#自动选择合适的位置，并且刻度之间最多不超过7（nbins）个间隔 locator=plt.FixedLocator(locs=[0,0.5,1.5,2.5,3.5,4.5,5.5,6])#直接指定刻度所在的位置
locator=plt.AutoLocator()#自动分配刻度值的位置
locator=plt.IndexLocator(offset=0.5, base=1)#面元间距是1，从0.5开始
locator=plt.MultipleLocator(1.5)#将刻度的标签设置为1.5的倍数
locator=plt.LinearLocator(numticks=5)#线性划分5等分，4个刻度
```

```python
# 接收各种locator的例子
fig, axs = plt.subplots(2, 2, figsize=(8, 5), tight_layout=True)
for n, ax in enumerate(axs.flat):
    ax.plot(x1*10., y1)

locator = matplotlib.ticker.AutoLocator()
axs[0, 0].xaxis.set_major_locator(locator)

locator = matplotlib.ticker.MaxNLocator(nbins=3)
axs[0, 1].xaxis.set_major_locator(locator)


locator = matplotlib.ticker.MultipleLocator(5)
axs[1, 0].xaxis.set_major_locator(locator)


locator = matplotlib.ticker.FixedLocator([0,7,14,21,28])
axs[1, 1].xaxis.set_major_locator(locator);
```

![[Assets/1675356701886.png]]

此外matplotlib.dates模块还提供了特殊的设置日期型刻度格式和位置的方式：

```python
# 特殊的日期型locator和formatter
locator = mdates.DayLocator(bymonthday=[1,15,25])
formatter = mdates.DateFormatter('%b %d')

fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
base = datetime.datetime(2017, 1, 1, 0, 0, 1)
time = [base + datetime.timedelta(days=x) for x in range(len(x1))]
ax.plot(time, y1)
ax.tick_params(axis='x', rotation=70);
```

### legend(图例)

下面介绍几个图例术语：

- legend entry（图例条目）：每个图例由一个或多个legend entries组成。一个entry包含一个key和其对应的label。
- legend key（图例键）：每个legend label左边的colored/patterned marker（彩色 / 图案标记）
- legend label（图例标签）：描述由key来表示的handle的文本
- legend handle（图例句柄）：用于在图例中生成适当的图例条目的原始对象

图例的绘制同样有OO模式与pyplot两种模式，写法相同，使用legend即可调用。

我们可以不传入任何参数，此时matplotlib将会自动寻找合适的图例条目。

我们也可以手动传入变量、句柄和标签，用于指定特定绘图对象和显示的标签值。

```python
fig, ax = plt.subplots()
line_up, = ax.plot([1, 2, 3], label='Line 2')
line_down, = ax.plot([3, 2, 1], label='Line 1')
ax.legend(handles = [line_up, line_down], labels = ['Line Up', 'Line Down']);
```

![[Assets/1675357827195.png]]

legend常用参数如下：

> 设置图例的位置

loc 参数接收一个字符串或数字表示图例出现的位置，ax.legend(loc='upper center')等同于ax.legend(loc=9)

|Location String|Location Code|
|--|--|
|'best'|0|
|'upper right'|1|
|'upper left'|2|
|'lower left'|3|
|'lower right'|4|
|'right'|5|
|'center left'|6|
|'center right'|7|
|'lower center'|8|
|'upper center'|9|
|'center'|10|

![[Assets/1675357663503.png]]

> 设置图例边框即背景

```python
fig = plt.figure(figsize=(10,3))
axes = fig.subplots(1,3)
for i, ax in enumerate(axes):
    ax.plot([1,2,3],label=f'ax {i}')
axes[0].legend(frameon=False) #去掉图例边框
axes[1].legend(edgecolor='blue') #设置图例边框颜色
axes[2].legend(facecolor='gray'); #设置图例背景颜色,若无边框,参数无效
```

![[Assets/1675357710317.png]]

> 设置图例标题

```python
fig,ax =plt.subplots()
ax.plot([1,2,3],label='label')
ax.legend(title='legend title');
```

![[Assets/1675357736013.png]]

### matplotlib中样式与颜色的使用

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
```

颜色是丰富可视化图表的重要手段。关于绘图样式，常见有三种方法：

- 修改预定义样式
- 自定义样式
- rcparams

颜色的使用有5中常见的表示单色的基本方法，以及用colormap表示多色的方法。

#### matplotlib绘图样式(style)

在matplotlib中，要想设置绘制样式，最简单的方法是在绘制元素时单独设置样式。 但是有时候，当用户在做专题报告时，往往会希望保持整体风格的统一而不用对每张图一张张修改，因此matplotlib库还提供了四种批量修改全局样式的方式

##### matplotlib预先定义样式

matplotlib贴心地提供了许多内置的样式供用户使用，使用方法很简单，只需在python脚本的最开始输入想使用style的名称即可调用。

```python
plt.style.use('default')
```

总共以下26种丰富的样式可供选择。

```python
[
    'Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 
    'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 
    'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 
    'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 
    'seaborn-deep', 'seaborn-muted', 'seaborn-notebook',
     'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 
    'seaborn-talk', 'seaborn-ticks', 'seaborn-white',
     'seaborn-whitegrid', 'tableau-colorblind10'
]
```

##### 用户自定义stylesheet

matplotlib支持使用后缀名为mplstyle的样使文件，通过plt.style.use(path)即可使用。

> 值得特别注意的是，matplotlib支持混合样式的引用，只需要在引用时输入一个样式列表，若是几个样式中涉及到同一个参数，右边的样式表会覆盖左边的样式表。

##### 设置rcparams

我们还可以通过修改默认rc设置的方式改变样式，所有rc设置都保存在一个叫做 matplotlib.rcParams的变量中。

```python
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.linestyle'] = '--'
plt.plot([1,2,3,4],[2,3,4,5]);
```

![[Assets/1675358521688.png]]

#### matplotlib色彩设置

在可视化中，如何选择合适的颜色和搭配组合也是需要仔细考虑的，色彩选择要能够反映出可视化图像的主旨。
从可视化编码的角度对颜色进行分析，可以将颜色分为色相、亮度、饱和度三个视觉通道。通常来说：

- 色相： 没有明显的顺序性、一般不用来表达数据量的高低，而是用来表达数据列的类别。
- 明度和饱和度： 在视觉上很容易区分出优先级的高低、被用作表达顺序或者表达数据量视觉通道。

在matplotlib中，设置颜色有以下几种方式：

##### RGB或RGBA

```python
# 颜色用[0,1]之间的浮点数表示，四个分量按顺序分别为(red, green, blue, alpha)，其中alpha透明度可省略
plt.plot([1,2,3],[4,5,6],color=(0.1, 0.2, 0.5))
plt.plot([4,5,6],[1,2,3],color=(0.1, 0.2, 0.5, 0.5));
```

![[Assets/1675358785648.png]]

##### HEX RGB 或 RGBA

```python
# 用十六进制颜色码表示，同样最后两位表示透明度，可省略
plt.plot([1,2,3],[4,5,6],color='#0f0f0f')
plt.plot([4,5,6],[1,2,3],color='#0f0f0f80');
```

![[Assets/1675358832476.png]]

> RGB颜色与HEX颜色之间是可以一一对应的，下面的网站提供了两种色彩表示方法的转换工具：
> 
> [Color Hex - ColorHexa.com](https://www.colorhexa.com/)

##### 灰度色阶

```python
# 当只有一个位于[0,1]的值时，表示灰度色阶
plt.plot([1,2,3],[4,5,6],color='0.5');
```

![[Assets/1675358963441.png]]

##### 单字符基本颜色

```python
# matplotlib有八个基本颜色，可以用单字符串来表示，分别是'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'，对应的是blue, green, red, cyan, magenta, yellow, black, and white的英文缩写
plt.plot([1,2,3],[4,5,6],color='m');
```

##### 颜色名称

```python
# matplotlib提供了颜色对照表，可供查询颜色对应的名称
plt.plot([1,2,3],[4,5,6],color='tan');
```

![[Assets/1675359040456.png]]

![[Assets/1675359048542.png]]

##### 使用colormap设置一组颜色

有些图表支持使用colormap的方式配置一组颜色，从而在可视化中通过色彩的变化表达更多信息。

在matplotlib中，colormap共有五种类型:

- 顺序（Sequential）。通常使用单一色调，逐渐改变亮度和颜色渐渐增加，用于表示有顺序的信息
- 发散（Diverging）。改变两种不同颜色的亮度和饱和度，这些颜色在中间以不饱和的颜色相遇;当绘制的信息具有关键中间值（例如地形）或数据偏离零时，应使用此值。
- 循环（Cyclic）。改变两种不同颜色的亮度，在中间和开始/结束时以不饱和的颜色相遇。用于在端点处环绕的值，例如相角，风向或一天中的时间。
- 定性（Qualitative）。常是杂色，用来表示没有排序或关系的信息。
- 杂色（Miscellaneous）。一些在特定场景使用的杂色组合，如彩虹，海洋，地形等。

[colormap官网解释](https://matplotlib.org/stable/tutorials/colors/colormaps.html "https://matplotlib.org/stable/tutorials/colors/colormaps.html")

更多有关colormap的内容参见colormap部分。
