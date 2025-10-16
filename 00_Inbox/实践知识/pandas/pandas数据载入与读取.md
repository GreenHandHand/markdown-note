## 2.pandas数据载入与读取

### 1.文本数据的读写

|       函数       |                     描述                     |
| :------------: | :----------------------------------------: |
|    read_csv    |      从文件、url或文件型对象读取分割好的数据，默认为','分隔符       |
|   read_table   |      从文件、url或文件型对象读取分割好的数据，默认为制表符'\t'      |
|    read_fwf    |           从特定宽度格式的文件中读取数据（无分隔符）            |
| read_clipboard |    read_table的剪切板版本，在将表格从Web页面上转换成数据是有用    |
|   read_excel   |          从Excel的XLS或XLSX文件中读取表格数据          |
|    read_hdf    |             读取pandas储存的HDF5文件              |
|   read_html    |              从HTML中读取所有的表格数据               |
|   read_json    | 从JSON（JavaScript Object Notation）字符串中读取数据  |
|  read_msgpack  |         读取MessagePack二进制中的pandas数据         |
|    read_sas    |         读取以Python pickle格式储存的任意对象          |
|    read_sql    | 将[[数据库/SQL|SQL]]的查询结果（使用SQLAlchemy）读取为pandas的DataFrame |
|   read_stata   |               读取Stata格式的数据集                |
|  read_feather  |               读取Feather二进制格式               |

这些函数的可选参数通常有以下类型：

- 索引：可以将一或多个列作为返回的DataFrame，从文件或用户处获得列名。或者没有列名。
- 类型推断和数据转换：包括用户自定义的值转换和自定义的缺失值符号列表。
- 日期时间解析：包括组合功能，也包括将分散在多个列上的日期和时间信息组合成结果中的单个列。
- 迭代：支持对大型文件的分块迭代。
- 未清洗数据问题：跳过行，页脚，注释，以及其他次要数据，比如使用逗号分隔千位符的数字。

由于现实世界的数据过于复杂，这些数据加载函数的可选参数十分复杂，可以在使用时针对性的查找。

### 2.read_csv与read_table的一些可选参数

|参数|描述|
|:--:|:--:|
|path|表明文件系统位置的字符串、url或文件型对象|
|sep或delimiter|用于分隔每行字段的字符序列或正则表达式|
|header|用作列名的行号，默认是0（第一行），如果没有列名的话应该为None|
|index_col|用作结果中行索引的列号或列名，可以是一个单一的名称或数字，也可以是一个分层索引。|
|names|结果的列名表名，和headers一起用|
|skiprows|从文件开头起处，需要跳过的行数或行号列表|
|na_values|需要用NA替换的值序列|
|comment|在行结尾处分隔注释的字符|
|parse_dates|尝试将数据解析为datatime，默认为False。如果为True，则尝试即系所有列，也可以指定列号或列名列表来进行解析。如果列表的元素是元组或者列表，则会将多个列组合在一起进行解析（例如将时间与日期拆分为两列）|
|keep_date_col|如果连接列到解析日期上，保留被连接的列，默认为False|
|converters|包含列名称映射到函数的字典（例如{'foo':f}会将函数f应用到foo列）|
|dayfirst|解析非明确日期时，按照国际格式处理（例如7/6/2012->June 7 2012），默认为False|
|data_parser|用于解析日期的函数|
|nrows|从文件开头处读入的行数|
|iterator|返回一个TextParser对象，用于零散地读入文件|
|chunksize|用于迭代的块大小|
|skip_footer|忽略文件尾部的行数|
|verbose|打印各种解析器输出的信息，比如位于非数值列中的缺失值数量|
|encoding|Unicode文本编码（例如'utf-8'用于表示UTF-8编码的文本）|
|squeeze|如果解析数据只包含一列，返回一个Series|
|thousands|千位分割符(例如','或者'.')|

### 3.分块读入文本

通过chunksize参数可以设定每块包含的行数。这将会返回一个chunk的类，通过该类可以按块读取文件中的文本。

### 4.更特殊的情况下的处理方式

当接受一个带有一行或者多行错误的文件时，可以通过csv库对文件进行处理。
csv.reader()的一些特殊参数：

|参数|描述|
|:--:|:--:|
|delimiter|一个用于分隔字符的字段，默认为','|
|lineterminator|行终止符，默认为'\r\n'，读取器将会忽略行终止符并识别跨平台行终止符|
|quotechar|用在含有特殊字段中的引号，默认是'"'|
|quoting|引用惯例。选项包含：csv.QUOTE_ALL（引用所有字段），csv.QUOTE_MINIMAL（只使用特殊字段，如分隔符），csv.QUOTE_NONNUMERIC和csv.QUOTE_NONE（不引用）。默认为QUOTE_MINIMAL|
|skipinitialspace|忽略每一个分隔符后的空白，默认为False|
|doublequote|如何处理字段内部的引号。如果为True，则是双引号|
|escapechar|当引用设置csv.QUOTE_NONE时用于转义分隔符的字符串，默认是禁用。|

当文件更加复杂时，只能使用split或者正则表达式方法re.split进行拆分与清理工作。

### 5.将数据写入文本

DataFrame对象通常含有 to_类型 的方法可以将DataFrame写入文件中，例如：to_csv(path)
Series也有相同的方法。

- 可选参数sep可以指定分隔符。
- 缺失值在输出是通常以空字符串的形式输出，通过na_rep可以指定空字符串的输出形式。
- 默认情况下行列标签都会被写入。通过index=False指定不写入行标签，通过header=False指定不写入列标签。
- columns可以指定写入的列。
