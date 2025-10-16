---
tags:
  - Rust
---

# 泛型、Trait 与生命周期

## 泛型数据类型

我们可以使用泛型为像函数签名或 [[00-笔记/Rust/结构体|结构体]] 这样的项创建定义，这样它们就可以用于多种不同的具体数据类型。

> [!note] 泛型代码的性能
> Rust 通过在编译时进行泛型代码的**单态化**（*monomorphization*）来保证效率。单态化是一个通过填充编译时使用的具体类型，将通用代码转换为特定代码的过程。因此，泛型代码与实际的多次编写类似函数得到的编译结果是相同的，泛型主要在编译期间进行转换，而在运行期间是没有损耗的。

### 在函数定义中使用泛型

如果要在函数体中使用参数，就必须在函数签名中声明它的名字，好让编译器知道这个名字指代的是什么。同理，当在函数签名中使用一个类型参数时，必须在使用它之前就声明它。Rust 中使用 `<>` 来表示一个类型参数。

```rust
fn largest<T>(list: &[T]) -> T {
	...
}

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

### 默认泛型类型参数

当使用泛型类型参数时，可以为泛型指定一个默认的具体类型。

```rust
trait Add<Rhs=Self> {
	type Output;

	fn add(self, rhs: Rhs) -> Self::Output;
}
```

## Trait

**特征** (*trait*) 定义了不同的类型间共同的行为。可以通过 trait 以一种抽象的方式定义共同行为。可以使用 *trait bounds* 指定泛型是任何拥有特定行为的类型。

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

> [!tip] supertrait
> trait bound 语法除了可以用于泛型中，还可以用于 trait 的声明中，此时要求实现该 trait 的类型必须实现其 supertrait。
> ```rust
> use std::fmt;
> 
> trait OutlinePrint: fmt::Display {
>     fn outline_print(&self) {
> 	    // pass
>     }
> }
> ```

### 特征对象

在 Rust 中，当我们使用 `impl Trait` 返回类型时，实际上返回的是一个**具体的泛型类型**，编译器在编译期会为每个具体的类型生成代码。然而，有时我们希望表达的是“**任意实现了某个 trait 的类型**”，而不是一个固定类型。特别是在以下场景中尤为常见：
- 一个 `Vec` 存储多个不同类型的值，但它们都实现了相同的 trait；
- 一个函数返回多种不同类型，但它们具有统一的行为接口。

这时，就需要使用 **特征对象**（*trait object*）。

**特征对象**是运行时的动态分发机制，它允许我们通过 trait 定义的接口调用方法，而不关心具体的类型实现。使用 `dyn Trait` 表示特征对象，可以通过以下两种方式创建：
- `&dyn Trait`：特征对象的不可变引用；
- `Box<dyn Trait>`：特征对象的堆分配所有权指针。

```rust
trait Draw {
    fn draw(&self);
}

// 接收任何实现了 Draw trait 的类型（所有权方式）
fn draw_boxed(x: Box<dyn Draw>) {
    x.draw();
}

// 接收任何实现了 Draw trait 的类型（引用方式）
fn draw_ref(x: &dyn Draw) {
    x.draw();
}

// 一个可以存放任意实现了 Draw trait 的类型的容器
let mut drawables: Vec<Box<dyn Draw>> = Vec::new();
```

> [!note] 为什么需要 `Box` 或 `&`
> Rust 的类型系统要求每个变量在编译时必须具有**已知大小**（`Sized`）。然而，`dyn Trait` 是一个**不确定大小的类型**，因为它代表的是“所有实现了某个 trait 的类型”，每个实现的实际大小可能不同。
>
> 因此，必须使用像 `Box<dyn Trait>` 或 `&dyn Trait` 这样的**指针类型**来包裹 trait object，从而为它提供一个已知大小的包装。

特征对象背后的机制是 **虚函数表**（*vtable*）。当你创建一个特征对象时，Rust 会在内存中维护两部分信息：
- 指向实际数据的指针；
- 指向该类型对应的 vtable 的指针。

vtable 是一个函数指针表，它记录了 trait 中定义的每个方法在当前具体类型中的实现。当你调用特征对象的方法时，Rust 会通过 vtable 动态分发，找到正确的实现方法并执行。

> [!tip] 这类似于 C++ 中使用基类指针来指向继承类对象，达到运行时多态的效果 (虚函数表)。

> [!note] 动态分发的代价
> 与静态分发相比，特征对象具有轻微的运行时开销，包括：
>
> - 一次间接跳转（函数指针调用）；
> - 禁用内联和某些优化。
> 
> 因此在性能敏感代码中应避免不必要的动态分发。

### 关联类型

**关联类型**（*associated types*）将一个类型占位符与 trait 相关联，使得该 trait 的方法定义可以在签名中使用这些占位符类型。该 trait 的实现者会为每个具体实现指定要使用的具体类型来替代占位符类型。这样，我们就能在定义 trait 时使用占位符类型，而无需预先知道这些类型的具体内容，直到实现该 trait 时再进行指定。

```rust
pub trait Iterator {
	type Item;

	fn next(&mut self) -> Option<Self::Item>;
}
```

上面的 `Iterator` 的例子就是一个常用的带有关联类型的 trait，其中 `Item` 是一个占位符类型，同时 `next` 方法的定义表明它返回 `Option<Self::Item>` 类型的值。

> [!note] 关联类型与泛型
> 关联类型可能看上去与泛型类似，但是无需在使用类型时标注，而是在实现 trait 时标注的。
>
> 换句话说，当使用泛型参数时，可以多次实现这个 trait，每次使用不同的具体泛型参数类型来指定使用哪一个 trait 的实现。
>
> 而使用关联类型时，无需标注类型，因为无法对同一个类型实现多个 trait。

### 在同名方法之间消除歧义

Rust 既不能避免一个 trait 与另一个 trait 拥有相同的名称的方法，也不能阻止为同一类型同时实现这两个 trait，同时还可以在该类型上在定义一个同名方法。

当调用这些方法时，需要明确告诉 Rust 我们想要使用哪一个。默认情况下，会先使用自身定义的方法。考虑下面的例子：

```rust
trait Pilot {
    fn fly(&self);
}

trait Wizard {
    fn fly(&self);
}

struct Human;

impl Pilot for Human {
    fn fly(&self) {
        println!("This is your captain speaking.");
    }
}

impl Wizard for Human {
    fn fly(&self) {
        println!("Up!");
    }
}

impl Human {
    fn fly(&self) {
        println!("*waving arms furiously*");
    }
}
```

当我们对一个 `Human` 实例调用 `fly` 方法时，默认会使用它自己的方法。如果需要使用另外两个 trait 定义的 `fly` 方法，则需要特别指定

```rust
let person = Human;
Pilot::fly(&person);   // 使用 Pilot 的
Wizard::fly(&person);  // 使用 Wizard 的
person.fly();          // 使用 Human 的
```

> [!note]
> 对于静态关联函数，没有 `self` 参数，此时无法通过上面的方法区分。此时需要使用**完全限定语法** (*flly qualified syntax*)。
> ```rust
> trait Animal {
>     fn baby_name() -> String;
> }
> 
> struct Dog;
> 
> impl Dog {
>     fn baby_name() -> String {
>         String::from("Spot")
>     }
> }
> 
> impl Animal for Dog {
>     fn baby_name() -> String {
>         String::from("puppy")
>     }
> }
> 
> fn main() {
> 	let s1 = Dog::baby_name();
> 	let s2 = \<Dog as Animal>::baby_name();
> }
> ```

## 生命周期

生命周期是一种特殊的泛型。不同于确保类型有期望的行为，生命周期用于保证引用在我们需要的整个期间内都是有效的。

在 Rust 中，每一个引用都有自己的**生命周期** (*lifetime*)，也就是引用保持有效的作用域。
- 大部分时候生命周期与类型一样，是可以推断的。
- 类似于当存在多种可能的类型时需要类型注解，也会出现引用的生命周期存在多种关联方式，需要进行生命周期注解。

> [!note] 生命周期注解与类型注解一样，是帮助编译器正确理解程序的，对实际代码的运行不会造成影响。

> [!tip] 生命周期注解甚至不是一个大部分语言都有的概念。

> [!note] 避免悬垂引用
> 生命周期最最重要的目的是避免**悬垂引用** (*dangling references*)，这会导致程序引用非预期的数据。
> ```rust
> fn main() {
>     let r;
>     {
>         let x = 5;
>         r = &x;
>     }
>     println!("r: {r}");
> }
> ```
> 由于 cpp 没有生命周期的机制，导致无法判断一个指针是否有效。当一个指针指向的对象被释放后，再次解引用这个指针就会导致预期之外的错误。Rust 通过生命周期机制，在编译器预防了悬垂引用。
>
> Rust 通过**借用检查器** (*borrow checker*) 来实现这一点，它通过比较作用域来保证所有的借用都是有效的。当且仅当借用的生命周期短于或等于被借用的变量时，编译通过。下面的例子就是一个编译不通过的例子：
> ```rust
> fn main() {
>     let r;                // ---------+-- 'a
>                           //          |
>     {                     //          |
>         let x = 5;        // -+-- 'b  |
>         r = &x;           //  |       |
>     }                     // -+       |
>                           //          |
>     println!("r: {r}");   //          |
> }                         // ---------+ 
> // 编译不通过, 'b 比 'a 更早释放
> ```
> 在这个例子中，使用 `'a` 表示 `r` 的生命周期，使用 `'b` 表示 `x` 的生命周期，`'b` 的生命周期只持续到内部括号结束，而 `'a` 的生命周期持续到函数结束，因此生命周期为 `'a` 到 `r` 无法借用生命周期为 `'b` 的 `x`。
>
> 下面是可以通过编译的版本。
> ```rust
> fn main() {
>     let x = 5;            // ----------+-- 'b
>                           //           |
>     let r = &x;           // --+-- 'a  |
>                           //   |       |
>     println!("r: {r}");   //   |       |
>                           // --+-------+
> }                         
> // 或者
> fn main() {
>     let r;                          // ----------+--'b
>     {                               //           |
> 		let x = 5;                  // --+--'a   |
>         r = &x;                     //   |       |
>         println!("x: {x}, r: {r}"); //   |       |
>     } // r droped here              // --+-------+
> }
> ```
> 在 `x` 被丢弃后，`r` 的生命周期也终止了。

### 生命周期注解

编译器在大多数情况下可以自动推断生命周期，但是对于一些特殊情况，需要程序员手动注释生命周期。例如，在函数的参数与返回值都为引用的情况下，可能存在多种联系：

```rust
fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() { x } else { y }
}
```

这里，返回值可能是 `x`，也可能是 `y`，编译器无法推断到底是哪一个。这种时候，就需要手动注解生命周期。

生命周期的注解使用 `'name` 的形式，添加在类型的前面，并与类型中间间隔一个空格。

```rust
&i32        // 引用
&'a i32     // 带有显式生命周期的引用
&'a mut i32 // 带有显式生命周期的可变引用
```

为之前的函数添加显式的生命周期声明。

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

在 Rust 中，当我们为两个参数标注了相同的生命周期注释时，Rust 本着安全的原则，总是认为该生命周期等于最短的生命周期。因此，函数中 `'a` 生命周期总是等于 `x` 与 `y` 中较短的那个生命周期。

> [!warning] 生命周期注释是一种特殊的泛型。

> [!note]
> 生命周期注解并不改变任何引用的生命周期的长短。相反它们描述了多个引用生命周期相互的关系，而不影响其生命周期。与当函数签名中指定了泛型类型参数后就可以接受任何类型一样，当指定了泛型生命周期后函数也能接受任何生命周期的引用。

> [!tip] 合同工
> 这里使用*合同工*的类比来阐述生命周期的含义。
> - **基本设定：**
>   - 每个引用就像是一位合同工，合同期为 `'a`、`'b`、`'static` 等。
>   - 分配给合同工的任务必须在其合同期（生命周期）内完成。
>   - Rust 编译器的职责就像 HR，它会严格审核任务是否能在合同到期前完成，以防止“超期用工”。
> - 我们作为项目经理（函数），负责调度这些合同工，合理规划他们的工作任务。
>
> 理解这个设定后，我们来看看几个典型场景：
>
> > [!example]- 找来一个人，将任务分配给他
> >
> > ```rust
> > fn assign(x: &'a str) -> &'a str {
> >     x
> > }
> > ```
> > 我们找来了合同工 `x`，他拥有生命周期 `'a`。任务直接交给了他，所以他必须在 `'a` 合同期内完成任务。我们只有在他离职前，才能使用任务的结果。(当他离职后，如果没有保存任务的结果，就无法找回了)
>
> > [!example]- 找来两个人，但将任务分配给其中一个
> >
> > ```rust
> > fn pick(a: &'a str, b: &'b str) -> &'a str {
> >     a
> > }
> > ```
> > 你召集了合同工 `a` 和 `b`，只是和 `b` 简单交流了一下，最终将任务交给了 `a`。由于任务只和 `a` 有关，所以只需确保 `a` 的合同期 `'a` 足够，`b` 的合同期 `'b` 不影响任务的完成。
>
> > [!example]- 找来两个人，分配他们一个合作任务
> >
> > ```rust
> > fn longest(a: &'a str, b: &'a str) -> &'a str {
> >     if a.len() > b.len() { a } else { b }
> > }
> > ```
> > 你找来 `a` 和 `b` 两位合同工，让他们商量谁来完成任务。由于你事先并不知道结果到底是谁完成的，那么这个任务必须安排在两人都在岗的那段时间内（即生命周期 `'a` 的交集），以确保无论谁负责都不会“超期用工”。
>
> 生命周期的约束体现的是函数的责任声明。通过生命周期注释，我们可以清晰地传达：
>
> - 这个任务（引用）会持续到什么时候（生命周期），
> - 在这个时间段内，外部可以通过返回的任务成果找到负责的合同工，
> - 超出这个时间，就可能造成“非法用工”（悬垂引用）。

综上，生命周期语法是用于将函数的多个参数与其返回值的生命周期进行关联的。一旦它们形成了某种关联，Rust 就有了足够的信息来允许内存安全的操作并阻止会产生悬垂指针亦或是违反内存安全的行为。

### 结构体定义中的生命周期注解

我们在结构体中创建引用变量时，也需要申明其生命周期。

```rust
struct ImportantExcerpt<'a> {
	part: &'a str,
}
```

该声明表示一个 `ImportantExcerpt` 类型的存在时间不能比其中 `part` 引用的变量存在的时间长。

### 生命周期注解省略

在特定情况下，生命周期注解的模式是固定的，此时编译器可以自动推断出生命周期而无需程序员指明。被编码进 Rust 引用分析的模式被称为**生命周期省略规则** (lifetime elision rules)，其内容如下：
1. 编译器为每一个引用参数都分配一个生命周期参数。
2. 如果只有一个生命周期参数，则将其分配给输出生命周期参数。
3. 如果方法有多个输入生命周期参数，并且其中一个是 `&self` 或者 `&mut self`，说明这是一个方法，那么所有输出生命周期参数都被赋予 `&self` 的生命周期。

> [!note] 方法定义的生命周期注解
> 由于生命周期注解是一种特殊的泛型，因此为带有生命周期的结构体实现方法的方式与带有泛型的结构体实现方法的方式相同。又由于生命周期省略规则第三条的存在，使得我们不需要在方法声明的函数签名中声明生命周期。
> ```rust
> impl<'a> ImportantExcerpt<'a> {
> 	fn level(&self) -> i32 {
> 		3
> 	}
> 	fn announce_and_return_part(&self, announcement: &str) -> &str {
> 		println!("Attention please: {announcement}");
> 		self.part
> 	}
> }
> ```

### 静态生命周期

静态生命周期是一种特殊的生命周期生命，它表示引用的值在整个程序运行期间都是有效的。最常见的静态生命周期引用就是字符串字面量。

```rust
let s: &'static str = "I have a static lifetime";
```

---

< [[00-笔记/Rust/错误处理|错误处理]] | [[00-笔记/Rust/自动化测试|自动化测试]] >
