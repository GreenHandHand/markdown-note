---
tags:
  - Rust
---

# 不安全 Rust

不安全 Rust 之所以存在，是因为静态分析本质上是保守的。当编译器尝试确定一段代码是否支持某个保证时，拒绝一些合法的程序比接受无效的程序要好一些。这必然意味着有时代码**可能**是合法的，但如果 Rust 编译器没有足够的信息来确定，它将拒绝该代码。在这种情况下，可以使用不安全代码告诉编译器，“相信我，我知道自己在干什么。” 不过千万注意，使用不安全 Rust 风险自担：如果不安全代码出错了，比如解引用空指针，可能会导致不安全的内存使用。

另一个 Rust 存在不安全一面的原因是底层计算机硬件固有的不安全性。如果 Rust 不允许进行不安全操作，那么有些任务则根本完成不了。Rust 需要能够进行像直接与操作系统交互甚至于编写你自己的操作系统这样的底层系统编程。底层系统编程也是 Rust 语言的目标之一。

## `unsafe` 块

Rust 可以使用 `unsafe` 关键词开启一个不安全的 Rust 代码块。这里有五类可以在不安全的 Rust 代码中使用的操作：
- 解引用裸指针
- 调用不安全的函数或方法
- 访问或修改可变静态变量
- 实现不安全 trait
- 访问 `union` 的字段

> [!note]
> `unsafe` 并不会关闭借用检查器或禁用任何其他 Rust 安全检查：如果在不安全代码中使用引用，它仍会被检查。`unsafe` 关键字只是提供了那五个不会被编译器检查内存安全的功能。你仍然能在不安全块中获得某种程度的安全。

## 解引用裸指针

Rust 中有两类裸指针，`*const T` 和 `*mut T`，这里的 `*` 不是解引用的意思，而是类型名称的一部分。在裸指针的上下文中，不可变的意思是在解引用是否可以直接赋值。

> [!note] 裸指针与引用和智能指针的区别
> - 允许忽略借用规则，可以同时拥有不可变和可变的指针，或多个指向相同位置的可变指针
> - 不保证指向有效的内存
> - 允许为空
> - 不能实现任何自动清理功能
> 
> 总之，裸指针的使用方式与 C 语言类似。

我们可以使用下面的方式申明裸指针：

```rust
let mut x = 5;

let r1: *const i32 = &raw const x;
let r2: *mut i32 = &raw mut x;

let r3 = 0x012345usize as *const i32; // 基于地址创建

// 还可以使用指针指针提供的方法创建
let mut y = Box::new(6);
let r4: *const i32 = &raw const *y;   // 先解引用后创建
let r5: *const i32 = Box::into_raw(y);// 或者使用 into_raw 静态方法
```

创建一个裸指针是安全的行为，因为这不会造成任何未定义的后果。但是解引用一个裸指针的安全性就无法得到保证，该操作需要在 `unsafe` 块中进行。

> [!tip] 这两种创建方式是下面的方式的语法糖
>
> ```rust
> let mut x = 5;
> 
> let r1: *const i32 = &x as *const i32;
> let r2: *mut i32 = &mut x as *mut i32;
> ```

## 使用不安全函数

不安全函数和方法与常规函数方法十分类似，除了其开头有一个额外的 `unsafe`。在此上下文中，关键字 `unsafe` 表示该函数具有调用时需要满足的要求，而 Rust 不会保证满足这些要求。通过在 `unsafe` 块中调用不安全函数，表明我们已经阅读过此函数的文档并对其是否满足函数自身的契约负责。

```rust
unsafe fn dangerous() {
	// 一些不安全的操作
}
```

不安全函数体也是有效的 `unsafe` 块，所以在不安全函数中进行另一个不安全操作时可以不新增额外的 `unsafe` 块。

> [!tip]
> 但是从 2024 edition 开始，直接使用 `unsafe` 操作会导致编译器会产生警告。在不安全函数的函数体内部执行不安全操作时，建议同样使用 `unsafe` 块，就像在普通函数中一样。这有助于将 `unsafe` 块保持得尽可能小，因为 `unsafe` 操作并不一定需要覆盖整个函数体。

一个函数包含了 `unsafe` 代码不代表我们需要将整个函数都定义为 `unsafe fn`。事实上，在标准库中有大量的安全函数，它们内部都包含了 `unsafe` 代码块。

## 外部函数接口

Rust 提供了一个关键字 `extern`，用于创建和使用 **外部函数接口**（*Foreign Function Interface*，FFI）以实现与其他语言编写的代码交互。`extern` 块中声明的函数在 Rust 代码中通常是不安全的，因此 `extern` 块本身也必须标注 `unsafe`。之所以如此，是因为其他语言不会强制执行 Rust 的规则，Rust 也无法检查这些约束，因此程序员有责任确保调用的安全性。

```rust
unsafe extern "C" {
    fn abs(input: i32) -> i32;
}

fn main() {
    unsafe {
        println!("Absolute value of -3 according to C: {}", abs(-3));
    }
}
```

> [!note] ABI
> 在 `extern "C"` 代码块中，我们列出了想要调用的外部函数的签名。其中 `"C"` 定义了外部函数所使用的**应用二进制接口**`ABI` (Application Binary Interface)，`ABI` 定义了如何在汇编层面来调用该函数。在所有 `ABI` 中，C 语言的是最常见的。有关 Rust 支持的所有 ABI 的信息请参见 [the Rust Reference](https://doc.rust-lang.org/reference/items/external-blocks.html#abi)。

> [!note] 从其他语言调用 Rust
> 也可以使用 `extern` 来创建一个允许其它语言调用 Rust 函数的接口。不同于创建整个 `extern` 块，就在 `fn` 关键字之前增加 `extern` 关键字并为相关函数指定所用到的 ABI。还需增加 `#[no_mangle]` 注解来告诉 Rust 编译器不要 mangle 此函数的名称。
>
>  `Mangling` 的定义是：当 Rust 因为编译需要去修改函数的名称，例如为了让名称包含更多的信息，这样其它的编译部分就能从该名称获取相应的信息，这种修改会导致函数名变得相当不可读。
> ```rust
> // 可以使用 #[unsafe(no_mangle)] 声明 unsafe 的函数
> #[no_mangle]
> pub extern "C" fn call_from_c() {
>     println!("Just called a Rust function from C!");
> }
> ```

## 访问或修改静态变量

全局变量在 Rust 中被称为 **静态**（*static*）变量。可以使用 `static` 申明一个静态变量。

```rust
static HELLO_WORLD: &str = "Hello, world!";

fn main() {
    println!("name is: {HELLO_WORLD}");
}
```

静态变量名称通常全大写，只能存储 `'static` 生命周期的变量或引用。访问不可变的静态变量是安全的。

只有在 `unsafe` 块中才可以访问和修改可变的静态变量。因为这种使用方式往往并不安全，当在多线程中同时去修改时，会不可避免的遇到脏数据。

> [!tip]
> 每当我们编写一个不安全函数，惯常做法是编写一个以 `SAFETY` 开头的注释并解释调用者需要做什么才可以安全地调用该方法。同理，当我们进行不安全操作时，惯常做法是编写一个以 `SAFETY` 开头并解释安全性规则是如何维护的。

> [!note] 常量
> Rust 可以使用 `const` 关键词申明一个常量，其使用方式与不可变的静态变量类似，都需要能够在编译期就计算出来，但是有以下区别：
> - 静态变量不会被内联，在整个程序中，静态变量只有一个实例，所有的引用都会指向同一个地址
> - 存储在静态变量中的值必须要实现 Sync trait

## 实现不安全的 trait

我们可以使用 `unsafe` 来实现一个不安全 trait。当 trait 中至少有一个方法中包含编译器无法验证的不变式（invariant）时该 trait 就是不安全的。

```rust
unsafe trait Foo {
    // 方法在这里
}

unsafe impl Foo for i32 {
    // 方法实现在这里
}
```

> [!example]
> 一个例子是 [[00_Inbox/Rust/无畏并发#`Sync` 与 `Send` Trait|手动实现Sync与Send Trait]]。

## 访问 `union` 中的字段

`union` 创建的类型主要用于和 C 语言的交互，这里不介绍 `union` 的使用。访问一个 `union` 是不安全的，因为 Rust 无法保证访问到 `union` 字段时是什么类型。

## 使用 Miri 工具检查不安全代码

当编写不安全代码时，你可能会想要检查编写的代码是否真的安全正确。最好的方式之一是使用 Miri，一个用来检测未定义行为的 Rust 官方工具。鉴于借用检查器是一个在编译时工作的**静态**工具，Miri 是一个在运行时工作的**动态**工具。它通过运行程序，或者测试集来检查代码，并检测你是否违反了它理解的 Rust 应该如何工作的规则。

> [!tip] 有点像 c++ 中提供的 sanitizer 的作用。

你可以通过输入 `rustup +nightly component add miri` 来同时安装 nightly 版本的 Rust 和 Miri。这并不会改变你项目正在使用的 Rust 版本；它只是为你的系统增加了这个工具所以你可以在需要的时候使用它。你可以通过输入 `cargo +nightly miri run` or `cargo +nightly miri test` 在项目中使用 Miri。

---

< [[00_Inbox/Rust/异步编程|异步编程]] | [[00_Inbox/Rust/宏|宏]] >
