# Layout

到目前为止我们已经成功的创建了一个窗口，并且添加了 widget 到其中。如果如果我们想要继续向其中添加更多的 widget，我们就不得不考虑控件的布局。Qt 为我们提供了 4 中基本的 layout 来管理布局。

| Layout        | Behaviour |
| ------------- | --------- |
| `QHBoxLayout` | 水平布局  |
| `QVBoxLayout` | 垂直布局  |
| `QGridLayout` | 网格布局  |
| `QStackedLayout`               |    在另一个布局上重叠布局       |

> 我们可以使用 Qt designer 来图形化设置布局，在这里我们使用代码以便于理解底层系统。

正如你所见，QT 中有三种位置的 layout，分别是 `QVBoxLayout`、`QHBoxLayout` 和 `QGridLayout`，在这之外还有 `QStackedLayout`，允许我们将一个 widget 放置在其他控件的上面。

在开始介绍 Layout 之前，我们先实现一个可视化 layouts 的应用。首先，创建一个显示单独颜色的 widget。
```python
class Color(QWidget):
	def __init__(self, color):
		super(Color, self).__init__()
		self.setAutoFillBackground(True)

		palette = self.palette()
		palette.setColor(QPalette.Window, QColor(color))
		self.setPalette(palette)
```
 在这个代码中，我们继承了 QWidget，并接受一个颜色参数。首先，我们设置 `setAutoFillBackground` 为 True，使得 widget 使用背景颜色自动填充窗口。接下来我们使用 `palette` 调色板来修改 `QPalette.Window` 的颜色。最后我们将 Widget 的调色板更新。

之后我们就可以使用上面的 Color 类来显示 Layout 是怎样分布的。

## QVBoxLayout

QVBoxLayout 将 Widget 竖着排列，每一个 Widget 将会被添加到最下方。现在我们可以向 app 中填加 QVBoxLayout。需要注意，为了向 QMainWindow 中添加 Layout，我们需要先将 Layout 应用到一个空白的 QWidget 上，然后使用 `.setCentralWidget` 来应用 Widget 与 Layout。
![[Pasted image 20240311183828.png]]

## QHBoxLayout

QHBoxLayout 是相同的。每个被新添加的 Widget 将会在最右边。
![[Pasted image 20240311183949.png]]
## 嵌套 Layout

如果想要更加复杂的布局，我们可以嵌套使用 Layout，即在一个 Layout 中使用另一个 Layout。 Qt 会先应用作为父对象的 Layout，再使用作为子对象的 layout。

我们可以使用 `Layout` 对象的 `.setContentMargins(up,down, left, right)` 来设置 Layout 边缘的空白或者使用 `.setSpacing` 来设置 Layout 中元素的间隔。

## QGridLayout

如果你尝试使用水平布局和垂直布局来模拟网格布局，你会发现很难控制每个元素的大小一致。使用 QGridLayout 可以很方便的控制网格形式的布局。

在网格布局中，我们使用索引来放置 Widget，并且你不需要填满所有网格。

## QStackedLayout

最后一种布局是重叠布局，正如其描述的，这个布局允许你将一个元素直接放置在其他元素之上，并且你可以选择展示哪一个 Widget。你可以使用这个来实现图层功能，事实上 QT 还有一个相同功能的控件称为 `QStackedWidget`。

使用这个控件，我们可以实现类似标签页的功能。不过在 QT 页提供了一个标签页控件 `QTabWidget`。该控件可以通过设置 `.setMoveable` 来允许拖动。