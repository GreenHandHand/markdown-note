---
tags:
  - Rust
---

# Cargo

本节记录一些 Cargo 的使用方法。

## 配置文件

Cargo 使用 `cargo.toml` 文件进行项目配置，其中有两种主要配置：
- `dev` 配置：开发环境配置，运行 `cargo build` 和 `cargo run` 时采用的配置。
- `release` 配置：发布时使用的配置，运行 `cargo build --release` 时采用。

当项目的 `Cargo.toml` 文件中没有显式增加任何 `[profile.*]` 部分的时候，Cargo 会对每一个配置都采用默认设置。通过增加任何希望定制的配置对应的 `[profile.*]` 部分，我们可以选择覆盖任意默认设置的子集。
- `[profile.dev]` 用于 `dev` 环境下的项目配置。
- `[profile.release]` 用于 `release` 环境下的项目配置。

> [!note]
> 在没有进行指定的情况下，`dev` 配置采用 `opt-level=0` 优化，`release` 配置采用 `opt-level=3` 优化。一般而言，保持默认就是最好的用法。

## 文档注释

准确的包文档有助于其他用户理解如何以及何时使用它们，所以花一些时间编写文档是值得的。**文档注释**（*documentation comments*）是 Rust 中一种用于文档的注释类型，它们会生成 HTML 文档。

```rust
/// Adds one to the number given.
///
/// # Examples
///
/// ```
/// let arg = 5;
/// let answer = my_crate::add_one(arg);
///
/// assert_eq!(6, answer);
/// ```
pub fn add_one(x: i32) -> i32 {
    x + 1
}
```

文档注释使用 `///` 而不是 `//` 以支持 `markdown` 文本格式。使用 `cargo doc --open` 会构建当前 crate 文档。

> [!note] 经常在文档注释中使用的标注
> - **Panics**：这个函数可能会 `panic!` 的场景。并不希望程序崩溃的函数调用者应该确保他们不会在这些情况下调用此函数。
> - **Errors**：如果这个函数返回 `Result`，此部分描述可能会出现何种错误以及什么情况会造成这些错误，这有助于调用者编写代码来采用不同的方式处理不同的错误。
> - **Safety**：如果这个函数使用 `unsafe` 代码，这一部分应该会涉及到期望函数调用者支持的确保 `unsafe` 块中代码正常工作的不变条件。

> [!note] 文档示例代码
> 在文档中编写的代码块也会被作为测试函数运行，对于上面的例子，如果我们使用 `cargo test` 也会将其中的代码作为一个测试函数运行。

除了函数注释，还可以使用 `//!` 为整个 crate 书写注释，通常放置在 `lib.rs` 的开头部分。

```rust
//! # My Crate
//!
//! `my_crate` is a collection of utilities to make performing certain
//! calculations more convenient.
```

> [!note] 重导出
> 用户在使用库文件时，一般不希望引用的层级过多，此时 [[00-笔记/Rust/包与模块#use 关键字|重导出]] 方式就可以起到作用。我们不必为这些重导出模块编写额外的文档注释，Cargo 会自动生成。

### 发布

在你可以发布任何 crate 之前，需要在 [crates.io](https://crates.io/) 上注册账号并获取一个 API token。为此，访问位于 [crates.io](https://crates.io/) 的首页并使用 GitHub 账号登录。一旦登录之后，查看位于 [https://crates.io/me/](https://crates.io/me/) 的账户设置页面并获取 API token。然后运行 `cargo login` 命令，使用得到的 API token 进行登陆。这个命令会通知 Cargo 你的 API token 并将其储存在本地的 *~/.cargo/credentials* 文件中。

> [!warning]
> 这个 token 是一个**秘密**（**secret**）且不应该与其他人共享。如果因为任何原因与他人共享了这个信息，应该立即到 [crates.io](https://crates.io/) 撤销并重新生成一个 token。

在发布一个 crate 之前，你需要在 crate 的 *Cargo.toml* 文件的 `[package]` 部分增加一些本 crate 的元数据（metadata）。例如，你需要添加 `name` 字段指定其他人可以通过什么搜索到你的 crate。其次，你需要 `license` 字段指定包的使用协议。

> [!note] 元数据
> 常用的元数据包括 `name, license, version, description` 等。

确定项目正常后，就可以使用 `cargo publish` 发布。

> [!warning]
> crate 一旦发布，就无法撤回了。如果你发现你发布的包中存在问题，可以使用 `cargo yank --vers xxx` 撤回版本。此时，已经使用该版本的项目仍然可以继续使用，但是新版本的项目就无法再添加撤回版本的包。

## 工作空间

Cargo 还提供了**工作空间** (*workspace*) 的功能，它可以帮助我们管理多个相关的协同开发的包。工作空间是一系列共享 `cargo.lock` 和 `target/` 的包。

要创建一个工作空间，首先，在工作目录下建立一个 `cargo.toml` 文件：
```shell
mkdir add
cd add
touch cargo.toml
```
并在其中添加 `[workspace]` 配置：
```toml
[workspace]
resolver = "3"                   # 解析算法，3 是目前最新的算法
member = ["adder, adder_two"]    # 这里填写工作空间中的 crate
```
上面的配置文件对应的工作空间目录应该是像下面的样子：
```shell
├── Cargo.lock
├── Cargo.toml
├── adder
│   ├── Cargo.toml
│   └── src
│       └── main.rs
├── adder_two
│   ├── Cargo.toml
│   └── src
│       └── lib.rs
└── target
```

如果我们想要在 `adder` 中使用 `adder_two` 模块中的内容，则需要在 `adder/Cargo.toml` 中添加依赖：
```toml
[dependencies]
adder_two = { path = "../adder_two" }
```

之后，使用 `cargo build` 来编译整个工作空间，并使用 `-p` 参数指定执行的二进制包的输出内容，即 `cargo run -p adder`。

> [!note]
> 在工作空间中，不同的 crate 互相依赖，但是共享一个输出目录。当我们在工作空间中使用 `cargo build` 命令时，会直接编译工作空间下的所有包，并在 `target` 中输出。

> [!note] 测试
> 与运行二进制输出相同，也可以使用参数 `-p` 指定要运行的 crate 的测试。如果不指定，那么 `cargo test` 将会运行工作空间中所有的测试。

## 其他命令

`Cargo install` 命令与 Rust 项目本身没有关系，而是一个用于管理二进制包的程序。用于可以利用 `Cargo install` 安装任意 Cargo 管理的项目。

如果 `$PATH` 中存在任意名字为 `cargo-something` 的二进制文件，那么就可以利用 `cargo something` 的方式来执行。像这样的命令可以通过 `cargo --list` 来列出。

---
< [[00-笔记/Rust/函数式特性|函数式特性]] | [[00-笔记/Rust/智能指针|智能指针]] >
