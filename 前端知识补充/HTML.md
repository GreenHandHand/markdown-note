# HTML

HTML 是一种网页描述语言，是超文本标记语言 (Hyper Text Markup Language) 的缩写。确切的说，HTML 不是一种编程语言，而是一种标记语言，使用标记标签来描述网页。

## HTML 标签

HTML 标记标签通常被称为 HTML 标签 (HTML tag)。是由尖括号包围的关键词，通常成对出现，如 `<b> text </b>`，其中第一个是开始标签，第二个是结束标签。

Web 浏览器的作用就是读取 HTML 文档，并以网页的形式显示出来。使用浏览器的查看源代码功能可以查看 HTML 文档的源代码。

### HTML 要素

#### HTML 元素

HTML 中的元素指从开始标签到结束标签中间的所有代码。例如 `<p>元素内容</p>` 是一个完整的 html 元素。空元素在开始标签中进行关闭，如图片，大多数的 HTML 元素可以拥有 [[#属性]]。

- `<p>` ：该元素定义了 HTML 文档中的一个段落。
- `<body>` ：该元素定义了 HTML 文档的主体。
- `<html>` : 该元素定义了整个 HTML 文档。

虽然在大多数的浏览器中，缺少结束标签仍然可以正确的显示，但是这样做会产生歧义, 因此不推荐。对于空标签，也推荐书写结束标签。例如换行 `<br>` 空标签，在最后添加斜杠来关闭空标签，因此正确的写法是 `<br/>`。

HTML 语言对大小写不敏感，也就说 `<p>` 与 `<P>` 的作用是相同的。

#### 属性

HTML 标签可以拥有属性，属性提供了更多的有关 HTML 元素的信息。这些属性总是以名称/值对的形式出现，如 `name="value"`，属性总是在 HTML 元素的开始标签中定义，如网页 `<a href="www.com"></a>`，并且并列的标签使用空格隔开。

同样的，HTML 对属性的大小写不敏感，但是在 (X) HTML 中，要求使用小写属性，因此推荐使用小写的属性。

属性始终被包含在引号中，其中双引号是最常用的，但是也可以使用单引号。

下面列举了使用于大多数 HTML 元素的属性：

| 属性  | 值               | 描述                       |
| ----- | ---------------- | -------------------------- |
| class | classname        | 规定元素的类名 (classname) |
| id    | id               | 规定元素的唯一 id          |
| style | style_definition | 规定元素的行内样式         |
| title | text             | 规定元素的额外信息         |

##### style 属性

style 属性用于改变 HTML 元素的样式。style 属性通过 [[CSS]] 样式来调整文字与段落的属性。由于引入了 CSS 样式，因此一些曾经使用的标签与属性应当避免使用，如下表所示：

| 标签或者属性              | 描述               |
| ------------------------- | ------------------ |
| `<center>`                | 定义居中的内容     |
| `<front>` 和 `<basefont>` | 定义 HTML 的字体   |
| `<s>` 和 `<strike>`       | 定义删除线文本     |
| `<u>`                     | 定义下划线文本     |
| `align`                   | 定义文本的对齐方式 |
| `bgcolor`                 | 定义背景颜色       |
| `color`                   | 定义文本颜色                   |

对于上面的这些标签和属性，应当使用样式替代。

#### 注释

HTML 中使用 `<! 注释 >` 来对代码进行注释。

### 常见 HTML 标签

这里列举一些常见的 html 标签，这些标签应当知道其作用。

#### 标题

HTML 的标题通过 `<h1> </h1>` 到 `<h6> </h6>` 实现的六级标题。标题对于一个网页来说非常重要，搜索引擎通常使用标题来为网页的结构和内容编制索引。
![[Pasted image 20230804113058.png]]

#### 段落

段落通过 `<p></p>` 来进行定义。需要注意的是，HTML 会将连续的空格视为一个空格，因此无法通过添加额外的空格来改变 HTML 的输出效果。

#### 链接

链接通过 `<a href="www.google.com">This is Google.</a>` 来指定，其中 href [[#属性]] 指定网页的地址。

#### 图像

图像通过 `<img src="test.jpg" width="104" height="142" />`，其中图像的名字是 `src` 属性提供，宽度与高度分别由 `width` 和 `height` 提供。图像的文本由属性 `alt` 提供。

#### 文本格式化标签

| 标签                                                         | 描述                                                                  |
| ------------------------------------------------------------ | --------------------------------------------------------------------- |
| `<b>`                                                        | <b>粗体字</b>                                                         |
| `<big>`                                                      | <big>大号字体</big>                                                   |
| `<em>`                                                       | <em>着重文字</em>                                                     |
| `<i>`                                                        | <i>斜体字</i>                                                         |
| `<small>`                                                    | <small>小号字</small>                                                 |
| `<strong>`                                                   | <strong>加重语气</strong>                                             |
| `<sub>`                                                      | 下<sub>标字</sub>                                                     |
| `<sup>`                                                      | 上<sup>标字</sup>                                                     |
| `<ins>`                                                      | <ins>插入字</ins>                                                     |
| `<del>`                                                      | <del>删除字</del>                                                     |
| `<code>`                                                     | <code>text, 计算机代码</code>                                         |
| `<kbd>`                                                      | <kbd>text, 键盘码</kbd>                                               |
| `<samp>`                                                     | <samp>text,计算机代码样本</samp>                                      |
| `<tt>`                                                       | <tt>text,打字机代码</tt>                                              |
| `<var>`                                                      | <var>text,变量</var>                                                  |
| `<pre>`                                                      | <pre>text,    预格式文本，保留空格与换行</pre>                        |
| `<abbr title="Hyper Text Markup Language">HTML</abbr>`       | <abbr title="Hyper Text Markup Language">HTML</abbr>,缩写             |
| `<acromym title="Hyper Text Markup Language">HTML</acronym>` | <acronym title="Hyper Text Markup Language">HTML</acronym> 首字母缩写 |
| `<address>`                                                  | <address>text,地址</address>                                          |
| `<bdo dir="rtl/ltr">`                                        | <bdo dir="rtl">文字方向</bdo>                                         |
| `<blockquote>`                                               | <blockquote>长引用</blockquote>                                       |
| `<p>`                                                        | <p>短引用</p>                                                         |
| `<cite>`                                                     | <cite>引用，引证</cite>                                               |                                                                      |
#### 其他

这里提供其他的标签，这些标签的内容可以简单的使用一行概括。

| 标签     | 作用                     |
| -------- | ------------------------ |
| `<hr />` | 在 HTML 页面中创建水平线 |
| `<br />` | 换行                     |
|          |                          |

