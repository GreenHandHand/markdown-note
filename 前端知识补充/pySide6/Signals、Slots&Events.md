---
tags:
  - qt
  - 前端
---
# 信号、槽与事件

在之前，我们已经创建了一个窗口，并且加入了一个简单的按钮，但是这个按钮没有任何用处。在本节中，我们需要一个方法来将按下按钮动作与完成一些任务联系起来。QT 通过信号与槽、信号与事件两种方式。

## 信号与槽

信号 (signals) 在一些被标记的事情发生时被发出。这个事情可以是任何事情，例如按按钮、输入文字或者窗口改变。很多的信号都被初始化为用户的动作，但是这不是一定的。除了表示某事发生外，信号还可以发送关于发生事件的信息。

槽 (slots) 负责在 qt 中接受信号。在 python 中，任何应用中的函数都可以作为槽，只需要将信号连接到函数上。如果信号发送了数据，那么接受槽也应该接受这些数据。许多 qt 控件都有自己的内置槽，使你可以直接将多个控件连通。

下面让我们了解一下基本的 QT 信号与如果将其连接到 widget 上。

### QPushButton Signals

这里使用之前编写的简单应用，该应用只包含主窗口和一个中心的按钮。现在我们将该按钮连接到一个 python 方法上。在这里我们创建了一个简单的槽函数 `the_button_was_clicked()`，接受来自按钮的 `clicked` 信号。
```python
class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()

		self.setWindowTitle("My App")

		button = QPushButton("Press Me!")
		button.setCheckable(True)
		button.clicked.connect(self.the_button_was_clicked)

		self.setCentralWidget(button)


def the_button_was_clicked(self):
	print("Clicked!")
```

### 接受数据

我们现在可以使用信号与槽函数了，现在我们需要了解上面的代码是如何工作的。在我们上面的代码中，按钮设置了 `setCheckable(True)` 来让按钮可以提供一个 checked 状态信号，对于一般的按钮，该信号总是 False。在我们上面的代码中，没有使用这个状态信号。下面我们再创建一个新的槽函数，来看看 checked 信号的作用。
```python
def the_button_was_toggled(self, checked):
	print("Checked?", checked)
```
多次点击按钮后，我们可以发现输出为：
```terminal
Clicked!
Checked? True
Clicked!
Checked? False
Clicked!
Checked? True
```
当我们将 Checked 信号打开后，Checked 信号实际上表示的就是按钮按下与松开的状态, 每一次按下按钮，都会转换 Checked 信号。

在 QT 中，你可以将任意多的槽函数同时连接到一个信号上，也可以使用一个槽函数同时响应任意多的信号。

### 存储数据

保存该 Widget 的状态在很多情况下是十分有用的。我们可以 python 的类方法来存储变量，下面的例子中，我们将 checked 变量存储在 self 中：
```python
def the_button_was_toggled(self, checked):
	self.button_is_checked = checked
	print(self.button_is_checked)
```

我们可以将上面的方法用在任何 PySide widgets 中。如果一个 widget 没有提供发送当前状态数据的 signal，那么我们就需要直接从 widget 中得到值，例如：
```python
def __init__(self):
	..
	self.button.released.connect(self.the_button_was_released)
def the_button_was_released(self):
	self.button_is_checked = self.button.isChecked()
```
在上面的例子中，释放信号在释放按钮时触发，但是不发送状态，因此需要直接通过 `isChecked()` 来检查按钮状态。

### 修改应用

到目前为止，我们已经完成了接受信号与输出控制台的功能，但是如何在点击按钮后完成一些功能？现在，让我们修改我们的槽函数与按钮，使其被按下后就不能被再次按下。
```python
def the_button_was_clicked(self):
	self.button.setText("You already clicked me.")
	self.button.setEnable(False)

	self.setWindowTitle("My Oneshot App")
```

上面，我们已经尝试了信号与槽函数的基本用法。下面我们将了解一些 QT 提供的 Widget。

### Widget 之间的直接连接

到目前为止，我们已经尝试了 Widget 的信号与 python 方法的连接。当信号被 Widget 发出后，python 方法将被自动调用，并且接受该信号。但是在 QT 中，我们并不总是需要使用 python 函数来处理信号，QT 提供了直接连接到其他 Widget 的方式。

下面的例子中，我们添加了一个 `QLineEdit` 与一个 `QLable`, 并且在 `__init__` 方法中将 line edit 的 `textChanged` 信号与 `QLabel` 的 `setText` 方法连接。现在，每当文字被修改时，`QLabel` 的 `setText` 方法都会接受被修改的文字。
```python
Class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()

		self.label = QLabel()
		self.lineEdit = QLineEdit()

		layout = QVBoxLayout()
		layout.addWidget(self.label)
		layout.addWidget(self.lineEdit)

		container = QWidget()
		container.setLayout(layout)
		self.setCentralWidget(container)
```
大多数的 Qt Widget 都有可以使用的槽函数与其对应的信号。在 Widget 的文档中，列出了每个 Widget 对应的槽函数。

## 事件

用户的每次交互都被视为一个事件。QT 中有多种交互事件，QT 将这些交互事件打包成了事件对象。事件对象中存储了发生的事件的信息，并被传递给了相应的处理对象。

通过定义一个事件处理对象 (event handler)，我们可以更改 Widgets 响应时间的方式。事件对象与其他方法的定义方式相同，但是名字需要是特定的、与事件对应的。

`QMouseEvent` 是一个常用的事件，该事件在鼠标移动或者点击时创建，下面的几个事件处理对象可以处理该事件：

| Event handler             | Event type moved      |
| :------------------------ | :-------------------- |
| `mouseMoveEvent()`        | Mouse moved           |
| `mousePressEvent()`       | Mouse button pressed  |
| `mouseReleaseEvent()`     | Mouse button released |
| `mouseDoubleClickEvent()` | Double click detected |
例如，当点击这个 Widget 的时候，`QMouseEvent` 将会被发送到 `mousePressEvent()` 事件处理对象中。这个事件处理对象可以通过 `QMouseEvent()` 对象了解到发生的事件，例如什么触发了它、在哪里触发的。

我们可以通过继承事件对象并通过重写处理方法来修改事件的响应。例如下面的代码：
```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.label = QLabel("Click in this window")
        self.setCentralWidget(self.label)

    def mouseMoveEvent(self, e):
        self.label.setText("mouseMoveEvent")

    def mousePressEvent(self, e):
        self.label.setText("mousePressEvent")

    def mouseReleaseEvent(self, e):
        self.label.setText("mouseReleaseEvent")

    def mouseDoubleClickEvent(self, e):
        self.label.setText("mouseDoubleClickEvent")
```
上面的代码创建了一个 label，并实时的显示当前发生的事件。值得注意的是，鼠标移动事件仅当鼠标被按下并移动时触发。如果你想要不被按下时触发，可以通过 `setMouseTracking(True)` 方法来设置。此外，可以发现鼠标的按下事件与双击事件在按下鼠标的时候同时被触发了，而释放鼠标时只有释放鼠标事件被触发。一般来说，为了追踪用户的点击，我们常同时关注按下与释放事件。

在事件处理对象中，我们可以获得一个事件对象。这个对象保存了一些可能被处理对象用到的信息，下面我们通过鼠标事件来详细了解。

### 鼠标事件

QT 中所有的鼠标事件都可以在 `QMouseEvent` 对象中得到，其中包含的有：

| Method         | Returns                                        |
| :------------- | :--------------------------------------------- |
| `.button()`    | Specific button that triggered this event      |
| `.buttons()`   | State of all mouse buttons (OR'ed flags)       |
| `.globalPos()` | Application-global position as a `QPoint`      |
| `.globalX()`   | Application-global horizontal X position       |
| `.globalY()`   | Application-global vertical Y position         |
| `.pos()`       | Widget-relative position as a `QPoint` integer |
| `.posF()`      | Widget-relative position as a `QPointF` float  |
我们可以在不同的事件处理对象中使用这些方法，或者完全无视它们。这些位置方法分别提供了全局位置与相对于控件的位置，并以 `QPoint` 的形式。对于鼠标的按键，使用标识符进行区分，定义在 `QtCore.Qt` 命名空间中，如下：

| Identifier                    | Value (binary) | Represents                                                       |
| :---------------------------- | :------------- | :--------------------------------------------------------------- |
| `Qt.MouseButton.NoButton`     | 0 (000)        | No button pressed, or the event is not related to a button press |
| `Qt.MouseButton.LeftButton`   | 1 (001)        | The left button is pressed                                       |
| `Qt.MouseButton.RightButton`  | 2 (010)        | The right button is pressed                                      |
| `Qt.MouseButton.MiddleButton` | 4 (100)        | The middle button is pressed                                     |

## 上下文菜单

上下文菜单是右键窗口时出现的小菜单。Qt 支持生成这种菜单，并且 Widget 有一个特别的方式来触发它们。在下面的例子中，我们修改主窗口的 `contextMenuEvent()`。这个事件在上下文将要被展示的时候触发，并传递一个 `QContextMenuEvent`。

为了修改这个事件，我们简单的重写了事件方法：
```python
class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
	def contextMenuEvent(self, e):
		context = QMenu(self)
		context.addAction(QAction("test 1", self))
		context.addAction(QAction("test 2", self))
		context.addAction(QAction("test 3", self))
		context.exec(e.globalPos())
```
如果你右击主窗口，将会看到一个上下文菜单出现，并且可以通过设置触发槽函数来是使菜单栏执行功能。

> 在执行 `exec()` 方法时，可以传入一个位置参数。如果该组件有父组件，那么该位置参数就是相对于父组件的位置。这里使用的父组件是 self，因此我们可以使用全局位置。

作为对比，下面是使用信号和槽函数实现的方式：
```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.show()
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_context_menu)

    def on_context_menu(self, pos):
        context = QMenu(self)
        context.addAction(QAction("test 1", self))
        context.addAction(QAction("test 2", self))
        context.addAction(QAction("test 3", self))
        context.exec(self.mapToGlobal(pos))
```

## 事件层次

在 PySide6 中，每一个 widget 都是一个特定的层次：Python 对象层次与 Qt 布局层次。如何响应、无视事件决定了你的 UI 的行为。

### python 对象层次

当你正在重写一个事件响应函数的时候，需要使用原来的方法，可以使用 python 的 `super` 方法调用父对象的方法。

### Layout 层次

当你添加一个 widget 到你的应用中时，也同时获得了另一种父对象。这种父对象可以通过 `.parent()` 方法得到。但是有的时候，你需要具体说明这些父对象是什么，例如 `QMenu` 或者 `QDialog`，但是这通常是自动的。例如当你将一个 widget 添加到主窗口时，这个主窗口将会变成这个 widget 的父对象。

当事件由于用户的交互而被创建时，这些事件对象将被首先发送到最上层的小部件中，例如，你在窗口中创建了一个按钮，并点击了这个按钮，那么这个事件将会先发送到按钮上，再发送到窗口上。

如果第一个 widget 无法处理事件或者选择不处理事件，那么这个事件将会一层一层的传递给他们的父对象。这个事件将被持续传递直到它被处理或者抵达主窗口上。

在自定义的事件处理方法中，可以使用 `.accept()` 方法来标记事件已被处理。或者使用 `.ignore()` 来标记事件没有被处理，在这种情况下，该事件将会被继续向父对象传递。