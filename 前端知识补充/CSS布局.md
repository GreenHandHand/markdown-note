---
tags:
  - 前端
---
# CSS 布局

如果我们只是想把内容都塞进一行里，那么不用设置任何布局都是可以的。然而，这样意味着内容很多时，我们就需要将浏览器窗口调整得很大，这会使得阅读网页非常难受。

在解决这个问题之前，我们需要了解 [[前端知识补充/CSS|CSS]] 中一个很重要的属性：`display`。

## display 属性

`display` 是 CSS 中最重要的用于控制布局的属性。每个元素都有一个默认的 `display` 值，这和元素的种类有关。大多数元素的默认 `display` 值为 `block` 或者 `inline`。一个 `block` 通常被称为块级元素，一个 `inline` 元素通常被称为行内元素。

<div style="display: block; border: 4px solid #8DBCEF;">
<code>div</code>是一个标准的块级元素。一个块级元素会新开始一行，并且尽可能撑满容器。其他常用的块级元素包括<code>p</code>，<code>form</code>和 HTML5 中的<code>header</code>、<code>footer</code>、<code>section</code>等。
</div>

而<span style="display: inlin; border: 4px solid #8DBCEF;">span</span>就是一个标准的行内元素。行内元素可以包裹一些文字而不打乱段落的布局。<code>a</code>元素是最常用的行内元素，它可以用作链接。

另一个常用的 `display` 值为 "<span style="display: none; border: 4px solid #004fAA;">看不见这段文本</span>"，即 `<span style="display: none; border: 4px solid #004fAA;">看不见这段文本</span>`。它和 `visibility` 属性不一样，把 `display` 设置为 `none` 元素不会占据它本来应该显示的空间，但是设置为 `visibility: hidden` 还会占据空间。

其他的 `display` 还包括 `list-item` 和 `table` 等，常用的还有 `inline-block` 与 `flex`。

## 块级元素属性

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

|    值     |             行为             |
| :------: | :------------------------: |
| `static` | 默认值，任意一个 `static` 元素不会被特殊的 |
