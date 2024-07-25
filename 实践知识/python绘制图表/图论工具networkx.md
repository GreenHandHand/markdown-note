# # 图论工具 networkx库

networkx是Python的一个包，用于构建和操作复杂的图结构，提供分析图的算法。图是由顶点、边和可选的属性构成的数据结构，顶点表示数据，边是由两个顶点唯一确定的，表示两个顶点之间的关系。顶点和边也可以拥有更多的属性，以存储更多的信息。
对于networkx创建的无向图，允许一条边的两个顶点是相同的，即允许出现自循环，但是不允许两个顶点之间存在多条边，即出现平行边。边和顶点都可以有自定义的属性，属性称作边和顶点的数据，每一个属性都是一个Key:Value对。

## 导入networkx库

如果没有networkx库，可以在控制台通过`pip install networkx`进行安装
networkx库有下面的惯例导入方式：

```python
import networkx as nx
```

networkx还可以使用 [[实践知识/python绘制图表/matplotlib]] 进行绘制，使用该功能需要导入matplotlib库

## 创建图

networkx中可以创建四种图分别是：
无向图  Graph()
有向图  DIGraph()
多重无向图 MultiGraph()  --两个结点之间的边数多于一条，又允许顶点通过同一条边和自己关联。
多重有向图 MultiGraph()

在创建后不会有图像出现，这样只是创建一个空对象

## 图的操作

下面列举了一些可以对图进行操作的方法：

### 节点Node

|方法|说明|
|:--:|:--:|
|add_node|添加节点|
|add_nodes_form|添加一系列节点，传入可迭代对象|
|nodes|获得所有节点，返回Node_view对象，使用参数data=True来显示值|
|remove_node|删除一个节点，同时会删除相连的边|
|remove_nodes_form|删除一系列节点，传入可迭代对象|
|update|更新值，有两个参数，可以更改边的值和节点的值，传入字典|
|has_node|检查是否含有某个节点|

### 边Edge

|方法|说明|
|:--:|:--:|
|add_edge|添加边|
|add_edges_form|添加一系列边，传入可迭代对象（由元组构成的列表），元组第三个值可选，可以用字典来一次设置多个值|
|edges|获得所有边，返回edge_view对象，传入data=True来获得值|
|add_edges_weighted_form|添加边，元组的第三个元素将被识别为weight|
|get_edge_data|获得边的值|
|update|更新值，有两个参数，可以更改边的值和节点的值，传入字典|
|has_edge|检查是否含有某条边|
|clear|清除所有的边和节点|

### 图Graph

|方法|说明|
|:--:|:--:|
|adj|ajd返回的是一个AdjacencyView视图，该视图是结点的相邻的顶点和顶点的属性，用于显示用于存储与顶点相邻的顶点的数据，这是一个只读的字典结构，Key是结点，Value是结点的属性数据。|
|edges|图的边是由边的两个顶点唯一确定的，边还有一定的属性，因此，边是由两个顶点和边的属性构成的|
|nodes|图的顶点是顶点和顶点的属性构成的|
|degree|图的节点的度|

### 一些迭代操作

|方法|说明|
|:--:|:--:|
|adjacency|获得所有节点元组(节点, 节点相关的边)的迭代器|
|adj|图的邻接对象，所有节点的邻接边字典|

## 图的索引

|方法|说明|
|:--:|:--:|
|G[node]|获得与节点相连的节点的字典(键为节点，值为node到节点的边的信息)|
|G[node1][node2]或者G.edge[node1, node2]|获得边(node1, node2)的值|
|G[node].update|更新节点的数据|
|G[node1][node2]|更新边的数据|
|G[node1][node2]['name']或者G.edges[node1, node2] = data|更新边的数据|
|del G[node1][node2]['name']|删除边的标签|

## 图的绘制

networkx库中带有基于matplotlib的绘图函数。使用这些函数需要导入matplotlib.pyplot模块

### nx.draw

使用matplotlib绘制图，draw_network的简化。默认不绘制节点名称。

|参数|描述|
|:--:|:--:|
|G|绘制的图|
|pos|图的位置|
|ax|绘制使用的Axes图表|
|**kwds|其他参数见draw_networkx等|

### **nx.draw_networkx**

使用matplotlib绘制图，比draw函数更加美观，支持更多的细节操作。默认绘制节点名称，带边框。

|参数|描述|
|:--:|:--:|
|G|绘制的图|
|pos|图的位置|
|ax|绘制使用的Axes图表|
|with_labels|True绘制节点标签|
|arrows|True绘制箭头，False绘制线，默认None|
|arrowstyle|箭头样式（有向图）|
|arrowsize|箭头大小，默认为10|
|nodelist|绘制特定的几个节点|
|edgelist|绘制特定的边|
|node_size|节点大小，默认300|
|node_color|节点颜色|
|node_shape|节点形状，默认圆|
|alpha|不透明度，0到1之间|
|width|边的宽度，默认1.0|
|line_width|节点边框的宽度，默认1.0|
|edge_color|边的颜色|
|style|线的样式|
|labels|传入标签字典|
|font_size|文字大小|
|font_color|文字颜色|
|font_weight|文字粗细|
|font_family|文字字体|
|label|图例|

### **nx.draw_networkx_\***

|\*处可选|描述|
|:--:|:--:|
|nodes|绘制节点|
|labels|绘制节点的值|
|edges|绘制边|
|edge_labels|绘制边的值|

## 图的其他操作

这些操作都是建立在networkx中的顶层函数

### 连通性

|函数|描述|
|:--:|:--:|
|is_connected|判断是否是连通图（无向图）|
|is_weakly_connected|判断是否是弱连通图（有向图）|
|is_strongly_connected|判断是否是强连通图（无向图）|
|connected_components|获得图的所有连通分支，返回图的连通分支的迭代器|
|number_connected_components|连通分支的数量|
