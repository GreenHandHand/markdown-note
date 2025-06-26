---
tags:
  - Rust
---

# 泛型、Trait 与生命周期

## 泛型数据类型

我们可以使用泛型为像函数签名或结构体这样的项创建定义，这样它们就可以用于多种不同的具体数据类型。

> [!note] 泛型代码的性能
> Rust 通过在编译时进行泛型代码的**单态化**（*monomorphization*）来保证效率。单态化是一个通过填充编译时使用的具体类型，将通用代码转换为特定代码的过程。因此，泛型代码与实际的多次编写类似函数得到的编译结果是相同的，泛型主要在编译期间进行转换，而在运行期间是没有损耗的。

### 在函数定义中使用泛型

如果要在函数体中使用参数，就必须在函数签名中声明它的名字，好让编译器知道这个名字指代的是什么。同理，当在函数签名中使用一个类型参数时，必须在使用它之前就声明它。Rust 中使用 `<>` 来表示一个类型参数。

```rust
fn largest<T>(list: &[T]) -> T {
	...
}
```

这里的类型 `T` 涵盖了 Rust 中的所有类型，因此我们无法直接对其使用一些特殊的方法。通过为 `T` 类型添加 `trait` 的方式，为泛型参数添加行为约束。

> [!note] Rust 通常可以自动推断类型参数。

### 在结构体中定义泛型

同样也可以用 `<>` 语法来定义结构体，它包含一个或多个泛型参数类型字段。

```rust
// x 和 y 是相同类型
struct Point<T> {
	x: T,
	y: T,
}

// x 和 y 是不同类型
struct Tuple<T, U> {
	x: T,
	y: U
}
```

> [!note] 可以在定义中使用任意多的泛型类型参数，不过太多的话，代码将难以阅读和理解。当你发现代码中需要很多泛型时，这可能表明你的代码需要重构分解成更小的结构。

### 在枚举中定义泛型

枚举中的泛型应用广泛，以常用的 `Option` 枚举和 `Result` 枚举为例。

```rust
enum Option<T> {
	Some(T),
	None,
}

enum Result<T, E> {
	Ok(T),
	Err(E),
}
```

### 泛型方法

泛型方法声明包含两种泛型类型，一种是结构体上的泛型，一种是针对方法的泛型。

```rust
struct Point<T1, T2> {
    x: T1,
    y: T2,
}

impl<T1, T2> Point<T1, T2> {
    fn x(&self) -> &T1 {
        &self.x
    }

	fn mixup<U1, U2>(&self, x_new: U1, y_new: U2) -> Point(T1, U2) {
		Point {
			x: self.x,
			y: y_new,
		}
	}
}
```

> [!warning]
> 注意必须在 `impl` 后面声明 `T`，这样就可以在 `Point<T>` 上实现的方法中使用 `T` 了。通过在 `impl` 之后声明泛型 `T`，Rust 就知道 `Point` 的尖括号中的类型是泛型而不是具体类型。

## Trait

trait 定义了不同的类型间共同的行为。可以通过 trait 以一种抽象的方式定义共同行为。可以使用 *trait bounds* 指定泛型是任何拥有特定行为的类型。

> [!warning] `trait` 类似于其他语言中的常被称为 **接口**（*interfaces*）的功能，虽然有一些不同。

### 定义 trait

trait 定义是一种将方法签名组合起来的方法，目的是定义一个实现某些目的所必需的行为的集合。

```rust
pub trait Summary {
	fn summarize(&self) -> String;
}
```

这里使用了 `trait` 关键字开始定义，声明为 `pub` 以便其他 crate 也可以实现这个 trait。在方法签名后跟分号，而不是在大括号中提供其实现。接着每一个实现这个 trait 的类型都需要提供其自定义行为的方法体。编译器会保证每个实现的签名都与 `trait` 中定义的一致。

> [!note] `trait` 中可以定义多个方法。

### 为类型实现 trait

使用 `impl for` 关键词为一个方法实现对应的 trait。编译器会保证实现了 `trait` 定义的所有方法。

```rust
pub struct NewsArticle {
    pub headline: String,
    pub location: String,
    pub author: String,
    pub content: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.headline, self.author, self.location)
    }
}
```

> [!note] 孤儿规则
> **孤儿规则** (*orphan rule*) 的含义为，至少在 `trait` 或类型中有一个属于当前作用域时，才能为其实现 `trait`。
>
> 这条规则确保了其他人编写的代码不会破坏你的代码，反之亦然。没有这条规则的话，两个 crate 可以分别对相同类型实现相同的 trait，而 Rust 将无从得知应该使用哪一个实现。

> [!note] 默认实现
> 我们在直接在 `trait` 方法中定义方法的默认实现。只需要在 `trait` 块中添加函数体，而不是一个分号。

### Trait Bound

由于 `trait` 定义了一系列的行为，将其作为函数的参数就表示该函数接受一系列拥有该行为的值。

```rust
pub fn notify(item: &impl Summary) {
    println!("Breaking news! {}", item.summarize());
}
```

Rust 还提供了 `trait bound` 语法用于更加复杂的场景。下面的定义与上面是等价的，或者说，上面的语法是下面语法的语法糖。

```rust
pub fn notify<T: Summary>(item: &T) {
    println!("Breaking news! {}", item.summarize());
}
```

`trait bound` 与泛型参数声明在一起，位于尖括号中的冒号后面。

> [!note] `trait bound` 的表达能力更强
> 考虑两个参数的情况。
> ```rust
> pub fn notify(item1: &impl Summary, item2: &impl Summary) {
> 	// 这种语法的含义是 item1, item2 都实现了 Summary trait
> }
> ```
> 如果我们需要 `item1` 与 `item2` 是同一个类型，这种方法就无法进行表达。此时，我们需要使用 `trait bound` 语法。
> ```rust
> pub fn notify<T: Summary>(item1: &T, item2: &T) {
> 	// 这种语法清晰的表达了 itme1 与 item2
> 	// 是相同的类型，且实现了 Summary trait
> }
> ```

> [!note] 多个 `trait bound`
> 当我们需要一个类型同时实现了多个 `trait` 时，我们可以使用 `+` 运算符进行连接。
> ```rust
> pub fn notify(item: &(impl Summary + Display)) {
> 	// 使用参数方法
> }
> pub fn notify<T: Summary + Display>(item: &T) {
> 	// 或者使用 trait bound
> }
> ```

> [!note] 使用 `where` 简化
> 当我们使用过多的 `trait bound` 时，函数的签名会变得无法直视。此时，使用 `where` 可以将 `trait bound` 声明后置，来增强可读性。
> ```rust
> fn some_function<T: Display + Clone, U: Clone + Debug>(t: &T, u: &U) -> i32 {
> 	// 非常复杂的签名，可读性较差
> }
> 
> fn some_function<T, U>(t: &T, u: &U) -> i32
> where
> 	T: Display + Clone,
> 	U: Clone + Debug,
> {
> 	// 后置可读性更好
> }
> ```

> [!note] 使用 `impl Trait` 作为返回值
> 也可以在返回值中使用 `impl Trait` 语法，表明函数返回了一个实现了 `trait` 的类型，但是调用者不清楚是什么类型。这是一个只有编译器知道的类型，或者是一个非常非常长的类型，在闭包或者迭代器中较为常见。
>
> 虽然可以通过 `impl Trait` 返回值返回一个实现了 `trait` 的类型，但是一个函数只能返回一个类型。假如你尝试返回多个不同的实现了 `trait` 的类型，编译器会报错。

> [!note] 在 `impl` 中使用 `trait bound`
> 可以在 `impl` 块中添加 `trait bound`，为满足条件的泛型实现对应的方法。^[其中的 `\` 符号需要删除，仅用于防止与 markdown 中的 html 嵌入块冲突]
>
> ```rust
> use std::fmt::Display;
> struct Pair\<T> {
>     x: T,
>     y: T,
> }
> 
> impl\<T> Pair\<T> {
>     fn new(x: T, y: T) -> Self {
>         Self { x, y }
>     }
> }
> 
> impl\<T: Display + PartialOrd> Pair\<T> {
>     fn cmp_display(&self) {
>         if self.x >= self.y {
>             println!("The largest member is x = {}", self.x);
>         } else {
>             println!("The largest member is y = {}", self.y);
>         }
>     }
> }
> ```

## 生命周期声明

