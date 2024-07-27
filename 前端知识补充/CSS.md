---
tags:
  - 前端
---
# CSS

本文档仅简单介绍 CSS 的基本知识。

**CSS** (Cascading Style Sheets，层叠样式表），是一种用来为结构化文档（如 HTML 文档或 XML 应用）添加样式（字体、间距和颜色等）的计算机语言，**CSS** 文件扩展名为 `.css`。

## 语法

CSS 规则由两个主要部分构成，分别是选择器和一条或多条声明。其中
- 选择器为需要改变样式的 HTML 元素
- 每条声明由一个或者多个属性-值对组成，属性为想要修改的样式名称，值为样式的值，属性和值使用冒号分开。

### 选择器

#### id 选择器

id 选择器可以为标有特定 id 的 HTML 元素指定特定的样式。HTML 元素以 id 属性来设置 id 选择器,CSS 中 id 选择器以 "#" 来定义。下面是一个例子：
```CSS
#para1
{
    text-align:center;
    color:red;
}
```

上述 id 选择器选择了 `id="para1"` 的 HTML 元素。ID属性不要以数字开头，数字开头的ID在 Mozilla/Firefox 浏览器中不起作用。

#### class 选择器

class 选择器用于描述一组元素的样式，class 选择器有别于 id 选择器，class 可以在多个元素中使用。class 选择器在 HTML 中以 class 属性表示, 在 CSS 中，类选择器以一个点 `.` 号显示, 下面是一个例子：
```CSS
p.center{
	text-align:center;
}
```
在该例子中，所有的 `<p>` 元素中 `class="center"` 的都被设为了居中。

## 样式表的创建

### 外部样式表

外部样式表使用 `.css` 后缀名储存样式，并在 HTML 中在 `<head>` 中使用 ` <link rel="stylesheet" type="text/css" href="mystyle_loc.css"> ` 来引用。

### 内部样式表

内部样式表在 `<head>` 中使用 `<style>` 直接创建。

### 多重样式

当内部样式与外部样式都使用并且冲突时，按照下面的优先级：
>（内联样式）Inline style > （内部样式）Internal style sheet >（外部样式）External style sheet > 浏览器默认样式