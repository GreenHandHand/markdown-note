# Colormap专题，matplotlib中的色彩

在matplotlib中，色彩的表示有一下方式：

|格式|例子|
|--|--|
|RGB 或RGBA元组，区间为[0,1]浮点数<br/>A为alpha值|RGB (0.1, 0.2, 0.5)<br/>RGBA (0.1, 0.2, 0.5, 0.3)|
|RGB 或 RGBA 的十六进制表示法|RGB, "#0f0f0f"<br/>RGBA, "#0f0f0f80"|
|RGB 或 RGBA 的十六进制表示法，<br/>重叠字母简写|"#aabbcc" as "#abc"|
|灰度值，区间为[0, 1]的浮点型字符串|'0' as black, '1' as while|
|基础色彩|单字母简写|
|不区分大小写的X11/CSS4颜色名称<br/>不带空格|'aquamarine'|
|xkcd：前缀不区分大小写的颜色名称|'xkcd:sky bule'|
|T10分类调色板不区分大小写的Tableau|'tab:bule'|
|"CN"颜色规范，C为默认属性循环索引<br/> 的数字之前|rcParams["axes.prop_cycle"] (default: cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])|

## 颜色叠加

matplotlib根据在order参数确定Artists图层的关系，zorder较大者在上方图层。默认情况下，后添加的Artist在上方图层。

matplotlib使用下面的公式来计算混合结果：

$$
RGB_{mix} = RGB_{below}\times(1-\alpha_{cover}) + RGB_{cover} * \alpha_{cover}
$$

## Norm的概念

**Norm （范式）是在 matplotlib.colors() 模块中定义的类，它规定了上述的映射是何种数学变换。**

### Normalize

默认的 Norm 是 matplotlib.colors.Normalize()，通过最简单的线性映射来绘制数据。

通过参数 vmin 和 vmax 就能快速构造一个 matplotlib.colors.Normalize() 实例：

```python
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
```

如下所述， 调用 norm(data) 将查找 data 在 colormap 中的位置（[0, 1]）。

```python-repl
In [3]: norm(0)
Out [3]: 0.5
```

### LogNorm

matplotlib.colors.**LogNorm**() 是通过取数据的对数（以 10 为底）来绘制数据。

即：$value_{map} = norm(data) = k \cdot \lg(data) + b$

![[Assets/1675434970136.png]]

### CenteredNorm

matplotlib.colors.CenteredNorm() 也是线性的，但它通过 vcenter 参数指定了的 data 的哪个值将放在 colormap 的 0.5 位置。

![[Assets/1675435048659.png]]

### SymLogNorm

我们知道，对数函数的定义域为 (0,+∞) ，而 matplotlib.colors.**SymLogNorm**() 则允许 data 在区间 (-∞, +∞) 使用对数映射的一种 Norm 。

由于 data 的区间跨域 0 ，而 0 的对数为负无穷，则必须设置一个特定的界限值——linthresh ，使得 data 在区间 (-∞,-linthresh) ∪(linthresh,+∞) 使用对数映射，而在区间 [-linthresh, linthresh] 使用线性映射。

**matplotlib.colors.**SymLogNorm**( *linthresh* ,  *linscale=1.0* ,  *vmin=None* ,  *vmax=None* , *clip=False* , *** ,  *base=10* )**

- linthresh：设定线性映射区间[-linthresh，linthresh]
- linscale：设定线性区间占用colormap的相对宽度
- base：对数底数

![[Assets/1675435253062.png]]

### PowerNorm

matplotlib.colors.**PowerNorm**() 是通过取数据的幂（以参数 gamma 为幂）来绘制数据。

即：$value_{map} = norm(data) = k \cdot data^\gamma + b$

![[Assets/1675435403709.png]]

### BoundaryNorm

matplotlib.colors.**BoundaryNorm**() 是一种离散边界的 Norm ，它在线性映射的基础上，通过参数 boundaries确定这些边界

```python-repl
N = 150
X, Y = np.mgrid[0:100:complex(0,N), 0:1:complex(0,N)]

Z = X

fig, (ax0,ax1,ax2) = plt.subplots(3, 1, figsize=(9,6), constrained_layout=True)

x = np.linspace(0,1000,1000)
y = x
ax0.plot(x,y)
plt.setp(ax0, title='$f(x)=x$', xlim=[-3,1003])

pcm = ax1.pcolor(X, Y, Z,
                 norm=mpl.colors.Normalize(vmin=0.0, vmax=Z.max()),
                 cmap='PuBu', shading='auto')
fig.colorbar(pcm, ax=ax1, extend=None)
plt.setp(ax1, yticks=[], title='Normalize()')

pcm = ax2.pcolor(X, Y, Z,
                 norm=mpl.colors.BoundaryNorm(boundaries=[0,10,40,70,80,100],ncolors=256),
                 cmap='PuBu', shading='auto')   
fig.colorbar(pcm, ax=ax2, extend=None)
plt.setp(ax2, yticks=[], title='BoundaryNorm(boundaries=[0,10,40,70,80,100],ncolors=256)')

plt.show()
```

![[Assets/1675435515315.png]]

### TwoSlopeNorm

有时我们希望在某个 data 作为中心点，两侧都有不同的 colormap ，并且两个 colormap 具有不同的线性比例。例如，地形图，我们把海拔为零的地方设为中心，陆地和海洋拥有不同比例的 colormap 。

此时，我们可以运用 colors.TwoSlopeNorm() 完成这一操作。

```python-repl
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
all_colors = np.concatenate((colors_undersea, colors_land),0)
new_cmap = colors.LinearSegmentedColormap.from_list('new_cmap', all_colors)

pcm = ax.pcolormesh(经度, 纬度, 地形, rasterized=True, 
                    norm=colors.TwoSlopeNorm(vmin=-500,vcenter=0,vmax=4000),
                    cmap=new_map, shading='auto')
```

### FuncNorm

如果上述的 Norm 你还不满意，你可以使用 colors.FuncNorm() 来自定义 Norm 。

**FuncNorm**( *functions* ,  *vmin=None* ,  *vmax=None* )

- functions：是一个函数名的二元组，包含：
  - _forward：原函数，用于计算 norm(data)
  - _inverse(x)：逆函数，用于计算 colorbar 的刻度
  - 要注意的是， _forward 必须是单调函数，因为其逆函数是 _inverse(x)

## Colormap的使用

**colormap 是将特定的色谱，映射到 [0,1] 之间的值，因而称之为颜色映射（数学概念）。**

通过使用 matplotlib.cm.get_cmap() 函数，我们可以访问 Matplotlib 的内置 colormap （颜色映射）。除了内置的 colormap ，还有一些外部库，如 palettable，同样拥有丰富的 colormap 。

### 获取colormap

首先，先看看如何从内置的 colormap 中获取新的 colormap 及其颜色值

```python-repl
matplotlib.cm.get_cmap(name=None, lut=None)
```

- name：内置的colormap的名称
- lut：重置colormap的采样区间，默认为256

### ListedColormap

ListedColormaps 是列表形式的 colomap ，其颜色的值存储在 .colors 中，可以使用 colors 属性直接访问。

也可以通过使用 np.linspace 和 np.arange 来访问，但应该要保持采样间隔的一致。如果采用不一致的采样间隔，则系统会采取最临近的值代替。

> ListedColormap的创建

要创建 ListedColormap ，只需要提供颜色值的列表或数组作为参数。

**matplotlib.colors.ListedColormap(colors, name='from_list', N=None)**

- colors：颜色列表或数组
  - 颜色列表：如['r', 'g', 'b']
  - [0, 1]区间的浮点数表示的RGB或RGBA的数组
- name：给自定义的Colormap命名，将这个Colormap注册到matplotlib，后面可以反复使用。
- N：颜色分为多少段
  - N < len(colors)，则只取colors中的前N个颜色
  - N > len(colors)，则通过重复colors，以拓展colors直至colors的个数为N

### LinearSegmentedColormap

LinearSegmentedColormaps 是线性分段形式的 colormap。色谱中有多个特定的间断点（colorvalue），这些间断点之间又以线性插值的形式自动填充一些点，使其看起来连续。

LinearSegmentedColormaps 的没有 .colors 属性。但仍然可使用 np.linspace 和 np.arange 来访问颜色值。

> LinearSegmentedColormap的创建

LinearSegmentedColormap 是基于线性分段查找表，在指定的间断点（colorvalue）之间进行RGB(A)的线性插值，而生成的 colormap 。

- 每个间断点是以[x[i], yleft[i], yright[i]] 为形式的三元组列表。
- 其中x[i]是锚点，表示间断点在colormap中的位置，必须在区间[0, 1]
- yleft[i]和yright[i]是间断点两侧的颜色值
  - 当yleft[i] == yright[i]，则颜色在该点是连续的
- 插值发生在yright[i]和yleft[i+1]之间

**matplotlib.colors.LinearSegmentedColormap(name, segmentdata, N=256, gamma=1.0)**

- name：给自定义的Colormap命名，将这个Colormap注册到matplotlib，后面可以反复使用。
- segmentdata：间断点查找矩阵。segmentdata是一个字典，有三个keys：'red'、'green'、'blue'。每个key的value就是以[x[i]，yleft[i]，yright[i]]三元组为行的矩阵。
- N：采样区间
- gamma：gamma值

还有一种简单的方法，即使用：

**LinearSegmentedColormap.from_list(*name* , *colors* , *N=256* , *gamma=1.0* )**

其中colors为颜色列表，或形如(x, value)的元组列表，x是锚定点，value就是颜色值。这样得到的颜色始终是连续的。

## 创建colorbar

通过colormap与norm两个对象我们可以创建出colorbar。

通常情况下，image（或其他"可映射"的对象）就能默认设置colormap和norm。

使用fig.colorbar()能够创建一个独立的colorbar，其构造如下：

```python
fig.colorbar(mappable, cax=None, ax=None, use_gridspec=True, **kw)
```

- mappable：通常是mpl.cm.ScalarMappable，用于指示colorbar的colormap。
- ax/cax：对应的axes，用ax则色条在axes外，用cax则色条为axes本身
- orientation：色条方向，vertical/horizontal

### 连续化色条

使用连续的Norm对象可以创建连续化的色条。这里使用Normalize进行绘制。

```python-repl
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)   # 设置子图到下边界的距离

cmap = mpl.cm.spring
norm = mpl.colors.Normalize(vmin=5, vmax=10)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='horizontal', label='spring');
```

![[Assets/1675433123468.png]]

### 色阶色条

使用边界化范式可以创建色阶色条，即norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend=extend)

```python-repl
fig, ax = plt.subplots(4, 1, figsize=(6, 4), constrained_layout=True)

cmap = mpl.cm.viridis
bounds = [-1, 2, 5, 7, 12, 15]

for i,extend in (0,'neither'),(1,'min'),(2,'max'),(3,'both'):
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend=extend)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax[i], orientation='horizontal',
             label=f"extend={extend}")
```

![[Assets/1675433237604.png]]

### 离散色条

构造ListedColormap对象，在使用该对象即可创建离散色条。

```python-repl
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = (mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])
        .with_extremes(under='0.75', over='0.25')) 
  
# under 和 over 用于显示 norm 范围之外的数据。

bounds = [1, 2, 4, 7, 8]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=ax,
    boundaries=[0] + bounds + [13],  # 增加一头一尾三角的位置
    extend='both',                   # 这次 extend 放在这里
    ticks=bounds,
    spacing='proportional',          # 刻度成比例
    orientation='horizontal',
    label='Discrete intervals, some other units',
)
```

![[Assets/1675433436507.png]]

> extendfrac参数可以调节两边三角的长度，可以使用'auto'作为参数
