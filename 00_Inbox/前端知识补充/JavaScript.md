---
tags:
  - 前端
---

# JavaScript

本文档仅介绍 JavaScript 中最基本的知识。

JavaScript 是 [[00_Inbox/前端知识补充/HTML]] 中默认的脚本语言，可以动态的改变网页，实现计算等。在 HTML 中，使用 `<script>` 来插入脚本。

JavaScript 每条命令以分号结尾，按顺序执行。下面是 JavaScript 中的一些保留字符，与大多数编程语言用法相同。

| 保留字       | 作用                                                             |
| ------------ | ---------------------------------------------------------------- |
| break        | 用于跳出循环。                                                   |
| catch        | 语句块，在 try 语句块执行出错时执行 catch 语句块。               |
| continue     | 跳过循环中的一个迭代。                                           |
| do ... while | 执行一个语句块，在条件语句为 true 时继续执行该语句块。           |
| for          | 在条件语句为 true 时，可以将代码块执行指定的次数。               |
| for ... in   | 用于遍历数组或者对象的属性（对数组或者对象的属性进行循环操作）。 |
| function     | 定义一个函数                                                     |
| if ... else  | 用于基于不同的条件来执行不同的动作。                             |
| return       | 退出函数                                                         |
| switch       | 用于基于不同的条件来执行不同的动作。                             |
| throw        | 抛出（生成）错误 。                                              |
| try          | 实现错误处理，与 catch 一同使用。                                |
| var          | 声明一个变量。                                                   |
| while        | 当条件语句为 true 时，执行语句块。                               |

## 变量类型

JavaScript 中，可以使用 `var` 定义变量的类型，这样定义的变量类型是动态类型的。也可以使用 `const` 定义一个常量类型，或者使用 `let` 关键词定义限定范围的类型。

实际上 JavaScript 中的变量类型为以下几种：
- **值类型 (基本类型)**：字符串（String）、数字 (Number)、布尔 (Boolean)、空（Null）、未定义（Undefined）、Symbol。
- **引用数据类型（对象类型）**：对象 (Object)、数组 (Array)、函数 (Function)，还有两个特殊的对象：正则（RegExp）和日期（Date）。

### 对象类型

对象类型可以使用 `var object = {dict:key}` 进行初始化，通过 `.` 运算符访问对象的属性和方法。JavaScript 中的对象使用与其他的语言基本相同。

#### 构造器

使用函数的形式创建一个对象构造器。使用 `this` 指代当前对象。下面是一个例子：
```JavaScript
function object_name(a, b){
	this.a = a;
	this.b = b;
}
```
在实例化对象时，可以使用 `var ob = new object_name(a, b);` 来实例化。通过 `for (.. in ..)` 可以遍历一个对象的所有属性。

## 函数

使用 `function function_name(var1, var2){return x}` 来定义一个函数。在函数中定义的变量为局部变量。

## 运算符

JavaScript 中支持与 C 语言相同的算术运算，包括 `+,-,*,/,++,--` 与组合运算符。对于比较运算符，除了大于小于运算符，JavaScript 在等于与不等于符号上与其他语言不同。

JavaScript 中，使用 `==` 比较两个值是否相等，使用 `===` 比较两个变量是否绝对相等 (变量的值和类型均相等)。使用 `!=` 比较两个值是否不等，使用 `!==` 比较两个变量是否不绝对相等 (变量的值和类型至少一个不等)。

使用 `&&`、`||` 与 `!` 作为逻辑运算符。

## 控制流

JavaScript 中的控制流使用与 C 语言完全相同。特别的，JavaScript 支持额外的 Foreach 循环，使用 `for(x in X){}`。

JavaScript 中可以使用一个元素标记一个代码块，然后使用 break 可以跳出任意代码块。下面是一个例子：
```JavaScript
list:{
	document.write(cars[2] + "<br>"); 
	break list; 
	document.write(cars[3] + "<br>");
} //使用break list可以直接跳出list代码块
```

## 文档对象 Document

当网页被加载时，浏览器会创建页面的文档对象模型（Document Object Model）。下面是该文档对象类型的对象树：
![[Assets/Pasted image 20230804205732.png]]

JavaScript 通过访问文档对象可以对文档进行编写与修改。

### 查找 HTML 元素

有两种方法查找 HTML 元素：
- 通过 id 找到 HTML 元素 : `document.getElementById("intro");`
- 通过类名找到 HTML 元素 ：`document.getElementsByClassName("intro");`

如果未找到，则会返回 NULL。

### 修改 HTML 元素

在找到 HTML 的元素后，可以通过修改其属性的方式来修改元素的内容、属性等。特别的，修改内容是通过属性 `innerHTML` 实现的。

HTML 的元素还有一个 `style` 属性，该属性指向 [[00_Inbox/前端知识补充/CSS|CSS]] 格式，可以通过修改 `style` 的 CSS 属性来修改元素的样式。

### 事件

一些事件可以触发 JavaScript 代码，这里以 `onclick` 为例，通过 `onclick="JacaScript"` 来实现事件的触发，这里推荐使用函数式编程的方式。

还有一种方式是在 JavaScript 模块中添加事件，只需在脚本中获取到触发实践的元素，修改其事件触发属性即可。

下面是几个常用的可以触发事件的属性：

| 属性          | 作用               |
| ------------- | ------------------ |
| `onload`      | 用户进入网页时触发 |
| `onunload`    | 用户离开网页时触发 |
| `onchange`    | 用户修改字段时触发 |
| `onmouseover` | 用户鼠标移入时触发 |
| `onmouseout`  | 用户鼠标移出时触发 |
| `onmousedown` | 用户鼠标按下时触发 |
| `onmouseup`   | 用户鼠标松开时触发 |
| `onclick`     | 用户完成点击时触发 |
