# pandas数据类型与修改

pandas支持的数据类型如下：

|pandas数据类型|python数据类型|Numpy数据类型|
|--|--|--|
|object|str|string__，unicode__|
|int64|int|int__，uint__|
|float64|float|float__|
|bool|bool|bool__|
|datetime64|NA|NA|
|timedelta[ns]|NA|NA|
|category|NA|NA|

## 转换数据类型的方法

### astype()方法

使用astype()函数进行强制类型转换

### 自定义函数进行类型转换

自定义函数通过apply方法进行类型转换

### Pandas内置函数

pandas内置了一些辅助函数可以帮助我们快速实现类型和转换

|函数|描述|
|:--:|:--:|
|pd.to_numeric|将数据转换为数字|
|pd.to_datetime|将数据转换为datetime64类型，可以传入多个列转换为一个date|