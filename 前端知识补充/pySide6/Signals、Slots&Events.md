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


def the_button_was_clicked(self):
	print("Clicked!")
```