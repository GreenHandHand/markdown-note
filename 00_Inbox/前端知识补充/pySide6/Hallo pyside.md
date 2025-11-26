---
tags:
  - qt
  - 前端
---

# Pyside6

在本节中学习如何通过 pyside6 与 python 创建一个简单的应用程序。pyside6 通过下面的命令安装。
```terminal
pip install pyside6
```

## 创建一个 application

```python
import sys
from pyside6.widget import QApplication, QWidget

app = QApplication(sys.argv)
window = QWidget()
window.show()

app.exec()
```

上述代码可以创建可以 qt 程序，包含一个空白的窗口。pyside 中的核心模块包括 QtWidgets, QtGui, QtCore, 这里简单了解一些。

在最开始，我们创建了一个 QApplication 对象，并传入了 `sys.argv`，用于传入命令行参数。如果可以确定不使用命令行参数，直接传入 `[]` 空列表也可以直接运行。每个 Qt 程序都有且仅有一个 QApplication。

之后，我们创建了一个 QWidget 对象，该对象是一个顶层的窗口，该窗口没有任何父类。每个 qt 程序由至少一个窗口组成，当所有的窗口都被关闭后，qt 程序将会自动退出。在一个顶层窗口创建后，需要使用 show 方法来使其显示，因为没有父类的窗口在创建时默认是隐藏的。

最后，我们使用了 `app.exec()` 来开启事件循环。

## 事件循环

在继续介绍后面的内容之前，我们需要先了解 qt 应用程序的构成方式。

Qt 程序的核心是 `QApplication` 类，每个程序有且仅有一个。该对象控制 Qt 程序的事件循环，该事件控制所有用户交互。用户的每一次交互，例如点击鼠标、输入文本或者按下键盘，都会被作为一个事件 (event) 插入事件队列中。事件循环在每一次迭代都会查看事件队列，并把正在等待处理的事件发送到特定的事件处理对象手中。处理完毕后，事件处理对象 (event handler) 将会将处理的结果反馈给事件循环，并等待下一个事件。由此看来，一个程序只能有一个事件循环。

## QMainWindow

我们在之前尝试了将 QWidget 作为窗口，在 qt 中可以使用多种控件作为窗口。实际上，如果你将 QWidget 换成 QPushButton，同样可以创建一个只有按钮的窗口。

创建一个只有按钮的窗口虽然没有什么用，但是这告诉我们一个事实，即复杂的 UI 实际上都是通过嵌套简单的 QWidget 创建的。

在 QT 中，提供了 QMainWindow 作为主窗口。这是一个标准的窗口，提供了许多可能用得上的功能，包括工具栏 (toolbars), 目录 (menus), 状态栏 (statusbar), 可活动的 widget 等。在本节中，我们先创建一个简单的空白主窗口。

单独的一个主窗口过于单调，我们现在向其中添加一些内容。如果要想主窗口中添加其他的窗口，最好的方式是继承一个 QMainWindow 的子类，并在 `__init__` 方法中创建初始程序，这样可以保证每个窗口独立的。

```python
class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("My App")

		button = QPushButton("Press Me!")

		self.setCentralWidget(button)
```

在上面的代码中，我们创建了一个 QMainWindow 的子类，并在其中添加了一个按钮。与 QMainWindow 和 QWidget 相同，这些 qt 的核心控件都在命名空间 QtWidgets 中。之后，我们使用 `setCentralWidget` 来放置这个 widget，默认的，这个 widget 将会占据整个窗口。

> 对于多个 widget 的布局 (layout) 方式，在之后的 [[00_Inbox/前端知识补充/pySide6/Layout]] 中介绍。

## 改变窗口、控件的大小

我们创建的窗口目前都是可以随意改变大小的，只需要拖动任何角落，就可以任意的变化。但是有时候我们希望可以锁定控件的大小。

在 QT 中，大小被定义为了 `QSize` 对象，该对象接受高度与宽度参数。下面的代码将窗口的大小固定为了 400x300。
```python
self.setFixedSize(QSize(400, 300))
```
除了固定大小，还可以使用 `setMinimumSize` 与 `setMaximumSize` 来设置最大与最小的尺寸。你可以在任意的 widget 对象上使用这些控制大小的方法。

想要使用控件自定义窗口，我们需要了解 QT 中的 [[00_Inbox/前端知识补充/pySide6/Widgets]]。为了响应用户与窗口的操作、处理事件，QT 中引入了 [[00_Inbox/前端知识补充/pySide6/Signals、Slots&Events]] 的概念。
