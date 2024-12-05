---
tags:
  - 最优化
  - 搜索算法
---

# A Start 算法

A\* 算法是启发式搜索算法，由于 A\* 算法在寻路问题中有着广泛的应用，这里以寻路问题为例介绍 A\* 算法，一个 A\* 算法使用了下面的几种要素：
1. open list：下一步要搜索的解，即下一步要移动的路径
2. close list：已经探索过的解，即已经探索过的路径
3. G 值：从起点开始的累计损失，例如移动的消耗、已经移动的路径的长度等
4. H 值：启发函数，例如从当前路径到目标坐标的距离
5. parent：记录每个点的父节点，即记录每个结点的上一个结点，用于回溯最佳路径

A\* 算法的步骤可以描述如下：
1. 计算起点邻域各点的 F 值，记录父节点, 并将他们加入 open list
2. 从 open list 中找到 F 值最小的点，将其移入 close list，计算其邻域的点的 F 值，并将他们加入 open list
3. 重复，直到到达目标点

下面给出伪代码:
```python
def A_star(G, start, end):
	open_list = [start]
	close_list = []
	F = {}
	parent = {}
	while Ture:
		cur_node = min_F_node(open_list)
		close_list.append(cur_node)
		compute_neighborhood_F(cur_node)
		parent[neighbor[cur_node]] = cur_node 

		if cur_node == end:
			break

	# 回溯，找到路径
	node = end
	while node != start:
		path.append(node)
		node = parent[node]
	path.append[start]
	path = reverse(path)
	return path
```