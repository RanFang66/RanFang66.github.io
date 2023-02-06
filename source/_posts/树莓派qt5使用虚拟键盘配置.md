---
title: 树莓派qt5使用虚拟键盘配置
date: 2023-01-19 18:12:00
categories: Notes
tags: 
  - raspberryPi 
  - Qt
---

## Qt5虚拟键盘功能的实现

在树莓派上使用Qt开发触摸屏UI软件时，由于使用触摸屏不使用物理键盘，在需要进行文本输入时需要在UI软件中提供虚拟键盘功能。在Qt5中有两种方式提供虚拟键盘功能，一种是自己按照自己的需求开发相应的虚拟键盘模块，这种方式可定制性强，灵活性好，但是需要自己做很多开发调试工作。另一种就是直接使用Qt5提供的QVirtualKeyboard虚拟键盘模块，这种方式直接使用qt官方提供的虚拟键盘模块，省去了自己开发的工作，同时官方的虚拟键盘模块整体UI设计还是不错的。

在Qt中使用官方虚拟键盘模块的步骤如下：

1. 首先要确认系统中已经安装了Qt5对应的虚拟键盘模块，debian系统下可以用如下指令安装：

   ```shell
   sudo apt-get install libqt5virtualkeyboard5-dev
   ```

2. 在Qt工程配置文件(.pro文件)中引用虚拟键盘模块：

   ```c++
   // 在.pro文件开始处添加
   Qt += virtualkeyboard
   ```

3. 在main函数的最开始将整个工程的输入模块环境变量定义为虚拟键盘模块：

   ```c++
   // main函数开始处添加qt输入模块环境变量
   qputenv("QT_IM_MODULE", QByteArray("qtvirtualkeyboard"));
   ```

4. 在需要调用虚拟键盘中的控件中设置输入模式

   输入控件(QLineEdit, QSpinBox等)有多种输入方式(InputMethodHints)可以设置，可以直接通过UI设计器或者代码的方式对相应的输入空间的输入方式进行设置。下面列出了常用的几种，具体可以参考Qt官方文档：

   <!--more-->
   
   | inputMethodHints       | 说明                       |                    |
   | ---------------------- | -------------------------- | ------------------ |
   | imhNone                | 没有特殊设置，出现完整键盘 | 默认               |
   | imhHiddenText          | 输入时隐藏输入文本         | 密码输入框常用     |
   | imhDate                | 日期选择键盘               | 日期输入时常用     |
   | imhTime                | 时间选择键盘               | 时间输入时常用     |
   | imhDigitsOnly          | 只有数字小键盘             | 纯数字输入时常用   |
   | imhEmailCharactersOnly | 邮件地址输入键盘           | 邮箱地址输入时常用 |
   | imhLatinOnly           | 只有字母键盘               | 纯字母输入时常用   |

## 树莓派Qt5虚拟键盘使用出现的问题

在经过如上的配置之后，重新编译运行qt5工程，就能够在qt项目中使用虚拟键盘了，在PC上使用效果一切正常。但是在树莓派上运行时却出现了问题。

![在树莓派上的qt虚拟键盘显示bug](https://i.stack.imgur.com/eJUjm.png)

在树莓派上点击输入控件(如lineEdit)，会发现虚拟键盘会覆盖整个界面，屏幕下方是虚拟键盘，整个其他部分都是黑色界面。

针对这个问题在网络上找了很久，也没找到有效的措施。最后还是通过科学上网，在stackoverflow[上找到了正确的解答](https://stackoverflow.com/questions/63955568/how-to-find-the-window-that-contains-the-qtvirtualkeyboard)。这个问题其实来自QT5虚拟键盘模块在arm架构下的一个内部bug。

如果要修复该bug，要么直接修改qt5虚拟键盘模块的源码，要么在使用虚拟键盘模块的工程中进行补丁修改。由于对qml编程不是很熟悉，因此采用在工程中打补丁的方式对该问题进行修复。

```c++
static void handleVisibleChanged() {
    if (!QGuiApplication::inputMethod()->isVisible())
        return;
    for (QWindow * w: QGuiApplication::allWindows()) {
        if (std::strcmp(w->metaObject()->className(), "QtVirtualKeyboard::InputView") == 0) {
            if (QObject *keyboard = w->findChild<QObject *>("keyboard")) {
                QRect r = w->geometry();
                r.moveTop(keyboard->property("y").toDouble());
                w->setMask(r);
                return;
            }
        }
    }
}
```

然后将该函数通过信号-槽机制绑定到弹出键盘时执行。

```c++
int main(int argc, char *argv[])
{
    qputenv("QT_IM_MODULE", QByteArray("qtvirtualkeyboard"));
    QApplication a(argc, argv);
    QObject::connect(QGuiApplication::inputMethod(), &QInputMethod::visibleChanged, &handleVisibleChanged);
    // ..
```

在增加了该补丁后，虚拟键盘模块在树莓派上的应用bug得到完美解决，显示结果如下图所示。

![修复后的qt虚拟键盘显示](https://i.stack.imgur.com/xo8E9.png)

## 总结经验：

最好还是能多读读源码，能够自己定位问题，不行再求助搜索引擎。

很多技术问题还是多尝试去外网上检索以下，国外的技术论堂氛围还是要好很多的。
