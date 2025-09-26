---
tags:
  - qt
  - 前端
---

# Widget in PySide6

在 QT 中，Widget 是用户可以交互的 UI 组件。用户接口由多个 widget 组合而成，并显示在窗口上。

QT 拥有大量的可选择的 widget，此外，还可以自定义自己的 widget。

## 一个简单的 Demo

下面是一个简单的 Demo，包含了 QT 中大多数的组件。这些组件被代码创建，并添加到了 layout 中。我们可以通过下面的 Demo 对这些 QT 的组件有一个初步的认识。
```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Widgets App")

        layout = QVBoxLayout()
        widgets = [
            QCheckBox,
            QComboBox,
            QDateEdit,
            QDateTimeEdit,
            QDial,
            QDoubleSpinBox,
            QFontComboBox,
            QLCDNumber,
            QLabel,
            QLineEdit,
            QProgressBar,
            QPushButton,
            QRadioButton,
            QSlider,
            QSpinBox,
            QTimeEdit,
        ]

        for widget in widgets:
            layout.addWidget(widget())

        central_widget = QWidget()
        central_widget.setLayout(layout)

        self.setCentralWidget(central_widget)
```

下面列出了上述组件及其作用：

| Widget           | What id does                            |
| :--------------- | :-------------------------------------- |
| `QCheckBox`      | A checkbox                              |
| `QComboBox`      | A dropdown list box                     |
| `QDateEdit`      | For editing dates and datetimes         |
| `QDateTimeEdit`  | For editing dates and datetimes         |
| `QDial`          | Rotateable dial                         |
| `QDoubleSpinBox` | A number spinner for floats             |
| `QFontComboBox`  | A list of fonts                         |
| `QLCDNumber`     | A quite ugly LCD display                |
| `QLabel`         | Just a label, not interactive           |
| `QLineEdit`      | Enter a line of text                    |
| `QProgressBar`   | A progress bar                          |
| `QPushButton`    | A button                                |
| `RadioButton`    | A toggle set, with only one active item |
| `QSlider`        | A slider                                |
| `QSpinBox`       | An integer spinner                      |
| `QTimeEdit`      | For editing times                       |
上面仅仅是 QT 提供的 Widget 的一小部分，更多的可以查看 QT 的文档。接下来我们将会看到一些最常用的 widget 并了解其使用的细节。

## QLabel

QLabel 是一行简单的文本，你可以将其放置在应用的任何位置。在创建时，可以传入一行字符串作为显示的文本，或者使用 `.setText` 方法来修改其显示的文本。
```python
label = QLabel("Hello")
label.setText("Hello2")
```
我们可以调整文字的参数，例如修改字体大小、对其方式等。
```python
font = label.font()
font.setPointSize(30) # 字体大小
label.setFont(font) # 应用到label上
label.setAlignment(
	Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
) # 对齐方式
```
在上面的代码中，我们使用的方式是从 widget 中获得文本，修改对应属性后再更新。这样可以保证字体与原始的一致。对其方式使用了 `Qt.AlignmentFlag` 命名空间的变量，水平对齐方式如下：

| Flag                            | Behavior                                    |
| :------------------------------ | ------------------------------------------- |
| `Qt.AlignmentFlag.AlignLeft`    | Aligns with the left edge                   |
| `Qt.AlignmentFlag.AlignRight`   | Aligns with the right edge                  |
| `Qt.AlignmentFlag.AlignHCenter` | Centers horizontally in the available space |
| `Qt.ALignmentFlag.AlignJustify` | Justifies the text in the available space   |
|                                 |                                             |

竖直对齐方式如下：

| Flag                           | Behavior               |
|:------------------------------ | ---------------------- |
| `Qt.AlignmentFlag.AlignTop`    | Aligns with the top    |
| `Qt.AlignmentFlag.AlignBottom` | Aligns with the bottom |
| `Qt.AlignmentFlag.AlignVCenter` |Centers vertically in the available space|                                |                        |     |     |

你可以使用 `|` 符号来连接两种对其方式，但是同一时刻只能使用一种水平对齐方式和一种竖直对齐方式。
> 除了上面的两种对其方式外，还可以使用 `Qt.AlignmentFlag.AlignCenter` 来将文本水平与竖直居中。

QLabel 还有一种用法，即使用 `.setPixmap()` 方法来显示图片。该方法接受一个 `pixmap` 对象，我们可以通过 `QPixmap(image_file_name)` 来创建。在默认情况下，图像将会保持其纵横比进行缩放。如果你想要延展图像，使其适应窗口，可以使用 `.setScaledContents(True)` 来使其适应窗口。

## QCheckBox

QCheckBox 提供了一个选项框。与其他的 widget 相同，有很多的方式可以改变这个 widget 的行为。CheckBox 如下所示：
![[Assets/Pasted image 20240311172858.png]]
如果想要使用程序修改 CheckBox 的状态，可以使用 `.setCheckState()` 并使用 `Qt.CheckState` 中的变量来修改，或者直接使用 `.setChecked()` 并传入 True 或者 False 来修改。在修改 QCheckBox 时，会发出信号 `stateChanged`。

| Flag                             | Behaivor                  |
| -------------------------------- | ------------------------- |
| `Qt.CheckState.Unchecked`        | Item is un checked        |
| `Qt.CheckState.PartiallyChecked` | Item is partially checked |
| `Qt.CheckState.Checked`                                 |     Item is checked                      |

Checkbox 有一个额外的状态 partially-checked，表示该 Checkbox 既没有打开也没有关闭，此时 Checkbox 一般为灰色且不可勾选的，一般用于多级选项中父级选项没有开启时。

## QComboBox

QComboBox 是一个下拉列表，默认关闭并且有一个箭头用于打开。QComboBox 适合用于选择不同的选项。
![[Assets/Pasted image 20240311174224.png]]
在改变其内容的时候，可以通过 `currentIndexChanged` 与 `currentTextChanged` 得到改变的条目与改变的文本。

此外，QComboBox 也可以是可编辑的，使用 `.setEditable(True)` 使其变为可以编辑的。用户可以在其中输入他们没有插入过的内容。我们可以通过一个 Flag 来控制 QComboBox 的行为：

| Flag                            | Bahavior                        |
| ------------------------------- | ------------------------------- |
| `QComboBox.NoInsert`            | No insert                       |
| `QComboBox.InsertAtTop`         | Insert as first item            |
| `QComboBox.InsertAtCurrent`     | Replace currently selected item |
| `QComboBox.InsertAtBottom`      | Insert after current item       |
| `QComboBox.InsertAfterCurrent`  | Insert after current item       |
| `QComboBox.InsertBeforeCurrent` | Insert before current item      |
| `QComboBox.InsertAlphabetiacally`                                |                 Insert in alphabetical order                |

使用 `.setInsertPolicy` 来控制插入行为，也可以通过 `setMaxCount` 来限制最大的插入数目。

## QListWidget

QListWidget 与 QComboBox 类似，但是外观与产生的信号不同。QListWidget 是一系列的选项，而不是下拉列表。QListWidget 提供了 `currentItemChanged` 信号发送 `QListWidgetItem` 变量，`currentTextChanged` 信号发送 item 的文本。

## QLineEdit

QLineEdit 是一个简单的单行文本编辑器，可以用于输入、设置这些不是限定文本的地方。单行编辑器有很多不同的信号，这里列举几个常用的：
- `returnPressed`：按下 Enter 后触发
- `selectionChanged`：改变焦点后触发
- `textChanged`：文本更改后触发
- `textEdited`：文本被用户修改后触发

## QSpinBox 与 QDoubleSpinBox

QSpinBox 提供了一个带箭头的小数字输入框，支持 Int 类型。QDoubleSpinBox 支持浮点类型。如果要限制输入框中的数字，可以使用 `.setMinimum()` 与 `.setMaximum()`，或者直接使用 `.setRange()`。数字的后缀和前缀分别使用 `.setPrefix` 和 `.setSuffix` 设置，例如设置单位与商品。

点击上箭头或者下箭头将会增加数字或者减小数字。要设置一次变化多少，可以使用 `setSingleStep`。

两个 Widget 都有 `valueChanged` 信号，发送一个数字。同时可以使用 `textChanged` 发送一个字符串，这个字符串将会包括前缀与后缀。

使用 `spinbox.lineEdit().setReadOnly(True)` 来使其不可编辑。

## QSlider

QSlider 提供了一个滚动条来选择数字，一般在设置音量时使用。它提供了 `.sliderMoved` 信号，触发时传递移动的位置，`.sliderPressed` 信号在被点击时触发。可以通过创建时使用 `Qt.Orientation.Vertical` 或者 `Qt.Orientation.Horizontal` 来设置滑动条的方向。
