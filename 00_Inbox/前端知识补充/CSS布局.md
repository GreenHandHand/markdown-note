---
tags:
  - 前端
---
# CSS 布局

如果我们只是想把内容都塞进一行里，那么不用设置任何布局都是可以的。然而，这样意味着内容很多时，我们就需要将浏览器窗口调整得很大，这会使得阅读网页非常难受。

在解决这个问题之前，我们需要了解 [[00_Inbox/前端知识补充/CSS|CSS]] 中一个很重要的属性：`display`。

## display 属性

`display` 是 CSS 中最重要的用于控制布局的属性。每个元素都有一个默认的 `display` 值，这和元素的种类有关。大多数元素的默认 `display` 值为 `block` 或者 `inline`。一个 `block` 通常被称为块级元素，一个 `inline` 元素通常被称为行内元素。

<div style="display: block; border: 4px solid #8DBCEF;">
<code>div</code>是一个标准的块级元素。一个块级元素会新开始一行，并且尽可能撑满容器。其他常用的块级元素包括<code>p</code>，<code>form</code>和 HTML5 中的<code>header</code>、<code>footer</code>、<code>section</code>等。
</div>

而<span style="display: inline; border: 4px solid #8DBCEF;">span</span>就是一个标准的行内元素。行内元素可以包裹一些文字而不打乱段落的布局。<code>a</code>元素是最常用的行内元素，它可以用作链接。

另一个常用的 `display` 值为 "<span style="display: none; border: 4px solid #004fAA;">看不见这段文本</span>"，即 `<span style="display: none; border: 4px solid #004fAA;">看不见这段文本</span>`。它和 `visibility` 属性不一样，把 `display` 设置为 `none` 元素不会占据它本来应该显示的空间，但是设置为 `visibility: hidden` 还会占据空间。

其他的 `display` 还包括 `list-item` 和 `table` 等，常用的还有 `inline-block` 与 `flex`。

## `block` 布局与 `span` 布局

### `width`

> [!note] `width`
> 设置块级元素的 `width` 属性可以防止它从左到右撑满整个容器，然后你就可以设置左右外边距为 `auto` 来使其水平居中。元素会占据你指定的宽度，然后剩余宽度会一分为二成为左右外边距。
> 
> 唯一的问题是当浏览器的宽度比元素的宽度还要窄时，浏览器会显示一个水平滚动条来容纳页面。

> [!example] `max-width`
> 在这种情况下，使用 `max-width` 替代 `width` 可以使浏览器更好的处理小窗口的情况。这点在移动设备上尤其重要。

当你设置了元素的宽度，实际展现的元素却超出了你的设置。这是因为元素的边框和内边距会撑开元素。在下面的例子中，两个相同宽度的元素显示的实际宽度却不一样：
```css
.simple { 
	width: 500px;
	margin: 20px auto;
}

.fancy { 
	width: 500px;
	margin: 20px auto;
	padding: 50px;
	border-width: 10px;
}
```
<div style="width: 500px; margin: 20px auto; border: 1px solid black; box-sizing: content-box;">我小一些...</div>
<div style="width: 500px; margin: 20px auto; padding: 50px; border-width: 10px; border: 1px solid black; box-sizing: content-box;">我比它大！</div>

一个解决方案是通过数学计算，使用比实际想要的宽度小一点的宽度减去内边距和边框的宽度。但是人们慢慢意识到了传统的盒子模型不直接，因此新增了一个叫做 `box-sizing` 的 CSS 属性。当你设置一个元素为 `box-sizing: border-box` 时，此元素的内边距和边框不会再增加它的宽度。这里有一个与之前相同的例子，但是区别是两个元素都设置了 `box-sizing: border-box`：
```css
.simple {
	width: 500px;
	margin: 20px auto;
	-webit-box-sizing: border-box;
	  -moz-box-sizing: border-box;
	       box-sizing: border-box;
}

.fancy {
	width: 500px;
	margin: 20px auto;
	padding: 50px;
	border: solid blue 10px;
   	box-sizing: border-box;
}
```
<div style="width: 500px; margin: 20px auto; box-sizing: border-box; border: 1px solid black;">现在我们一样大了</div>
<div style="width: 500px; margin: 20px auto; padding: 50px; box-sizing: border-box; border: 1px solid black;">现在我们一样大了</div>
这是目前最好的方法，一些 CSS 开发者想要页面上的所有元素都有如此表现，所以把如下的 CSS 代码放在他们的页面上(例如 obsidian 中就是默认启用的)：
```css
* {
	box-sizing: border-box;
}
```
> [!note] 由于 `box-sizing` 是一个较新的属性，因此还应该像上面那样使用 `-webkit-` 和 `-moz-` 前缀，这可以启用特定浏览器实验中的特性。它们是支持 IE8+ 的。

### `position`

为了制作更多复杂的布局，我们现在讨论一下 `position` 属性。

|     值      |                                                                         行为                                                                          |
| :--------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: |
|  `static`  |                             默认值，任意一个 `static` 元素不会被特殊的定位。一个 `static` 元素表示它不会被 `positioned`。一个元素被设置其它值表示它会被 `positioned`                             |
| `relative` |            `relative` 表现得和 `static` 一样，除非你添加了一些额外的属性。在一个相对定位的元素上设置 `top`、`right`、`bottom` 和 `left` 属性会使其偏离其正常位置。其他元素的位置不会受到该元素的影响发生位置改变。            |
|  `fixed`   |                        一个 `fixed` 元素会相对于视窗来定位，这意味着即便页面滚动，它还是会停留在相同的位置。与 `relative` 一样，`top`、`right`、`bottom` 和 `left` 属性都可用。                        |
| `absolute` | `absolute` 是最棘手的值。`absolute` 与 `fixed` 的表现类似，但是它不是相对于视窗而是相对于最近的 `positioned` 祖先元素。如果绝对定位的元素没有 `positioned` 祖先元素，那么它是相对于文档的 `body` 元素，并且他会随着页面滚动而移动。 |
### `float`

<div>
 <div style="float: right; width: 50%; height: 8em; margin: 0 0 1em 1em; border: 2px solid black; text-align: center;">环绕我!</div>
 
 <section> 
 <code>float</code> 属性在布局中也是常用的属性，该属性可以实现文字环绕图片的功能。</section>

<section style="clear: right;">另外还有一个 <code>clear</code> 属性用于控制浮动。对上面的段落使用 <code>clear: right</code>，就可以把这个段落移动到浮动的下方。</section>
</div>

在使用浮动的时候经常会遇到一个古怪的事情，即当图片比包含它的元素还高、而且它是浮动的时候，这个图片就会溢出到容器外面。对包含它的容器使用样式就可以修正。
```css
.container{
	overflow: auto;
	zoom: 1; /* needed by IE6 */
}
```

完全使用浮动来布局也是常见的，它可以实现与 `position` 布局相同的效果。

### 百分比宽度

百分比是一种包含于块的计量单位。它对于图片很有用。使用百分比布局可以方便的调整宽度比例。但是使用百分比宽度会导致使用 `min-width` 布局方式失效，因为其他的元素部分遵从该属性。

### 媒体查询

响应式设计是一种让网站针对不同浏览器和设备呈现不同显示效果的策略，这样可以让网站在任何情况下显示的很棒。

媒体查询是做此事所需的最强大的工具。我们可以使用百分比布局，然后在浏览器变窄到无法容纳侧边栏中的菜单时，把布局显示成一列。
```css
@media(min-width: 600px) {
	nav {
		float: left;
		width: 25%;
	}
	section {
		margin-left: 25%;
	}
}
@media(max-width: 599px){
	nav li {
		display: inline;
	}
}
```


## `inline-block` 布局

使用 `display` 属性的值为 `inline-block` 可以方便的创建行内块，使用他们来组合出一些效果。

使用 `inline-block` 进行布局时，要知道
- `vertical-align` 属性会影响到 `inline-block` 元素，我们可以把它的值设置为 `top`
- 我们需要设置每一列的宽度
- 如果 HTML 源码中元素之间有空格，那么列与列之间就会产生空隙。


## 多列布局

这里提供了两个的属性用于实现文字的多列布局。
```css
.therr-column{
	padding: 1em;
	column-count: 3;
	column-gap: 1em;
}
```
还有很多的其他与 `column` 相关的属性，更多的需要查阅文档。

## `flexbox`

`flexbox` 布局模式被用来重新定义 CSS 中的布局方式，也是一种目前常用的方式。这里不作介绍。