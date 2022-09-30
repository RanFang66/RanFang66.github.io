---
title: Makefile Learning Notes
date: 2022-09-29 19:11:06
categories: Learning
tags: 
  - embedded
  - Linux
---

## 基本语法与规则

Makefile 描述的是文件编译的规则，一条规则主要由两部分组成，依赖关系和命令：

```makefile
targets : prerequisites
	command
```

- targets: 规则的目标，可以是.o，可以是可执行文件，还可以是一个标签
- prerequisites: 生成目标的依赖文件，可以是多个或者没有
- command: make时需要执行的命令，可以是任意的shell命令，可以有多条命令，每条命令占一行。

<!--more-->

Makefile中主要包含的内容：

1. 显示规则：显式说明如何生成一个目标文件
2. 隐晦规则：make命令支持自动推导功能
3. 变量的定义：Makefile中可以定义一系列变量，其用法与shell中的变量类似。
4. 文件指示：可以在一个makefile中引用另一个makefile，类似c中的include。可以根据某些情况制定makefile中的有效部分，类似c中的条件编译。还可以自定义多行的命令。
5. 注释：与shell相同，用#号注释。

make的执行过程：

在shell中执行make之后，会自动搜索Makefile文件，并且将脚本中的第一条规则

定义的目标作为最终目标，然后检查目标的依赖，这时有三种情况：

1. 如果依赖的中间文件不存在，则会根据规则生成中间文件
2. 如果依赖的中间文件存在，并且没有它所依赖的生成文件新，则会重新生成这个中间文件。
3. 如果依赖的中间文件存在，并且它所依赖的生成文件也没有更新，则不会重新生成。

## Makefile 中的通配符

Makefile 支持shell中的通配符，如下：

| 通配符 | 说明                   |
| ------ | ---------------------- |
| *      | 匹配0个或多个任意字符  |
| ?      | 匹配任意一个字符       |
| []     | 匹配方括号中制定的字符 |

通配符在引用变量时的使用，在写依赖规则时，如果直接使用有通配符的变量，会出现错误。如

``` makefile
obj = *.c
test : $(obj)
	gcc -o $@ $^
```

当按如上规则make时，会报错找不到*.c。这是因为变量引用中的通配符并不会被展开。而如果想在引用的变量中使用通配符，则需要借助wildcard函数，这个函数会在引用变量时进行通配符的展开，如：

``` makefile
obj = $(wildcard *.c)
test : $(obj)
	gcc -o $@ $^
```

Makefile 中还支持 % 通配符，也是匹配任意个字符，

``` makefile
test : test1.o test2.o
	gcc -o $@ $^
%.o : %.c
	gcc -o $@ $^
```

%.o会把所有.o文件组合成一个列表，每次从列表中取出一个文件，然后找到文件中和%名称相同的.c文件，然后执行下面的文件，直至列表中的文件取完。这个属于Makefile中静态模规则：规则存在多个目标，规则存在多个目标，并且不同的目标可以根据目标文件的名字自动构造出依赖文件。

## Makefile中的变量

### 变量的定义

`变量名称 = 值列表`

变量名称可以由字母数字和下划线构成，等号左右的空格没有要求，值列表可以是0项，也可以是多项，中间用空格分隔。

``` makefile
OBJ = a.o b.o c.o
test : $(OBJ)
	gcc -o test $(OBJ)
```

### 变量的基本赋值

- 简单赋值(:=)   编程语句中的常规赋值方式，只对当前语句的变量有效
- 递归赋值(=)    所有目标变量相关的变量都会受影响，赋值语句可能影响多个变量
- 条件赋值(?=)   如果变量未定义，则使用符号中的值定义变量，
- 追加赋值(+=)   在变量的值列表后面追加一项新的值

``` makefile
x = abc
y = $(x)d
x := efg	# 简单赋值 x=efg, y=abcd
x = efg		# 递归赋值 x=efg, y=efgd
x ?= efg	# 条件赋值 x=abc, y=abcd
x += efg	# 追加赋值 x=abc efg, y=abcd

```

### 自动化变量

| 自动化变量 | 说明                                                         |
| :--------: | ------------------------------------------------------------ |
|     $@     | 表示规则的目标文件名。如果目标是一个文档文件（Linux 中，一般成 .a 文件为文档文件，也成为静态的库文件）， 那么它代表这个文档的文件名。在多目标模式规则中，它代表的是触发规则被执行的文件名。 |
|     $%     | 当目标文件是一个静态库文件时，代表静态库的一个成员名。       |
|     $<     | 规则的第一个依赖的文件名。如果是一个目标文件使用隐含的规则来重建，则它代表由隐含规则加入的第一个依赖文件。 |
|     $?     | 所有比目标文件更新的依赖文件列表，空格分隔。如果目标文件时静态库文件，代表的是库文件（.o 文件）。 |
|     $^     | 代表的是所有依赖文件列表，使用空格分隔。如果目标是静态库文件，它所代表的只能是所有的库成员（.o 文件）名。 一个文件可重复的出现在目标的依赖中，变量$^ 只记录它的第一次引用的情况。就是说会去掉重复的依赖文件。 |
|     $+     | 类似“$^”，但是它保留了依赖文件中重复出现的文件。主要用在程序链接时库的交叉引用场合。 |
|     $*     | 在模式规则和静态模式规则中，代表“茎”。“茎”是目标模式中“%”所代表的部分（当文件名中存在目录时， “茎”也包含目录部分）。 |

## Makefile 目标文件搜索路径

有两种方式可以制定Makefile脚本搜索目标文件的目录路径：一般搜索`VPATH` 和选择搜索`vpath`.

VPATH和vpath的区别在于，VPATH是一个变量，而且是环境变量，使用时需要指定文件的路径；vpath是关键字，按照指定的模式搜索，搜索时不仅要加上路径，还要加上限值条件。

### VPATH

在Makefile中按如下形式使用VPATH指定搜索路径：

``` makefile
VPATH := src
```

当要指定多个路径时，不同的路径之间用空格或者冒号分隔。

``` makefile
VPAHT := path1:path2
```

 ### vpath

vpath按照给定模式在指定目录中搜索，使用方法如下：

```makefile
1. vpath PATTERN DIRECTORIES
2. vpath PATTERN
3. vpath
```

用法二的意思是清除符合模式 的搜索目录，用法3 单独使用vpath是清楚所有已设置的文件搜索路径。

### 应用

假设现在有一个工程，包含两个目录，src目录下包含main.c, module1.c, module2.c源文件，inc目录下包含module1.h, module2.h头文件。通过如下方式先声明文件搜索路径。

``` makefile
vpath %.c 	src
vpath %.h	inc		## 或者使用 VPATH = src inc
main : main.o module1.o module2.o
	gcc -o $@ $<
main.o : main.c
	gcc -o $@ $^
module1.o : module1.c module1.h
	gcc -o $@ $<
module2.o : module2.c module2.h
	gcc -o $@ $<
```

## Makefile 隐含规则



## Makefile 条件判断

在实际工程中，经常会遇到要根据某个条件执行不同的编译操作的情况。这是就需要在Makefile中能够实现分条件执行语句的功能。Makefile中提供以下条件判断关键字：

| 关键字 | 说明             |      |
| ------ | ---------------- | ---- |
| ifeq   | 判断参数是否相等 |      |
| ifneq  | 判断参数是否不等 |      |
| ifdef  | 判断是否有值     |      |
| ifndef | 判断是否没有值   |      |

### ifeq 和 ifneq

使用语法

``` makefile
ifeq (ARG1, arg2)
# ifeq "ARG1" "ARG2"
# ifeq 'ARG1' 'ARG2'
	xxx
else
	xxx
endif
```

举例

``` makefile
gcc_libs = -lgnu
default_libs = 
ifeq ($(CC), gcc)
	libs = $(gcc_libs)
else
	libs = $(default_libs)
endif
foo : $(objects)
	$(CC) -o foo $(objects) $(libs)
```

### ifdef 和 ifndef

使用语法：

``` makefile
ifdef VARIABLE_NAME
```

## Makefile 伪目标

伪目标的含义是它并不会创建目标文件，但是会去执行这个目标下面的命令。使用伪目标有两点原因：

1. 避免Makefile中只用来执行命令的目标与时间的文件出现名字冲突
2. 提高执行make时的效率

3. 文件清理操作clean作为伪目标的实现：

``` makefile
.PHONY : clean
clean :
	rm -rf *.o $(PROGRAM)
```

2. make 命令对于多个目录的并行和递归操作：

``` makefile
SUBDIRS = foo bar baz
.PHONY : subdirs $(SUBDIRS)
subdirs : $(SUBDIRS)
$(SUBDIRS) : 
	$(MAKE) -C $@
foo: baz
```

​			上述脚本实现了递归的调用三个子目录下的make命令。

3. 伪目标实现多目标文件的生成

   如果想要在一个Makefile里控制生成多个可执行文件，也可以借助伪目标实现：

   ``` makefile
   .PHONY all
   all : test1 test2 test3
   test1 : test1.o
   	gcc -o $@ $^
   test2 : test2.o
   	gcc -o $@ $^
   test3 : test3.o
   	gcc -o $@ $^
   ```

   上述脚本通过声明一个all伪目标，实现同时生成test1, test2, test3三个目标可执行文件。

## Makefile中的字符串处理函数

Makefile中调用函数的语法：

`$(<function> <arguments>)` 或者 `${<function> <arguments>}` 

其中function是函数名，arguments是参数列表。函数名和参数列表之间用空格分开，参数列表中的多个参数用逗号分隔。

### 模式字符串替换函数

使用格式：

```makefile
$(patsubst <pattern>, <replacement>, <text>)
```

说明：查找text中符合模式pattern的部分，如果找到匹配的，则用replacement替换。返回替换后的新字符串。

常见用法示例：

```makefile
C_SOURCE = $(shell find . -name *.c)
C_OBJECTS=$(patsubst %.c, %.o, $(C_SOURCE))
main : $(C_OBJECTS)
	gcc -o $@ $^
$(C_OBJECTS) : $(C_SOURCE)
```

### 字符串替换函数

``` makefile
$(subst <from>, <to>, <text>)
```

将text字符串中的from替换成to，返回替换后的新字符串

用法示例：

``` makefile
OBJ = $(subst Beijing, Shanghai, I love Beijing)
all:
	@echo $(OBJ)	# 显示 I love Shanghai
```

### 去空格函数

```makefile
$(strip <string>)
```

去掉string开头和结尾的空格，并且将中间连续多个空格合并为一个空格。返回去空格后的字符串。

### 字符串查找函数

```makefile
$(findstring <find>, <in>)
```

在in中查找find，如果查找的目标字符串存在，返回目标字符串，如果不存在返回空。

用法示例

```makefile
OBJ = $(findstring you, I hate you)
all:
	@echo $(OBJ)	# 显示 you
```

### 模式过滤函数

``` makefile
$(filter <pattern>, <text>)
```

过滤出text中符合模式pattern的字符串，可以有多个pattern，返回所有符合pattern的字符串。

用法示例：

``` makefile
OBJ = $(filter %.c %.o, foo1.c foo2.o foo3.s)
all:
	@echo $(OBJ)	#显示foo1.c foo2.o
```

### 反向过滤函数

``` makefile
$(filter-out <pattern>, <text>)
```

与filter函数相反，滤除所有符合pattern模式的字符串，返回所有不符合pattern的字符串。

``` makefile
OBJ = $(filter-out %.c %.o, foo1.c foo2.o foo3.s)
all:
	@echo $(OBJ)	#显示foo3.s
```

### 排序函数

``` makefile
$(sort <list>)
```

将list中的单词按照单词升序排序，返回排序后的字符串

示例：

``` makefile
OBJ = $(sort orange apple pear apple)
all: 
	@echo $(OBJ) # 显示 apple orange pear
```

**notice** : sort会去掉重复的字符串

### 去单词函数

``` makefile
$(word <n>, <text>)
```

取出字符串text中的第n个单词，返回取出的第n个单词。

示例：

```makefile
OBJ = $(word 2, foo1.c foo2.c foo3.c)
all:
	@echo $(OBJ)	# 显示 foo2.c
```

## Makefile 中的文件名操作函数

### 取目录函数

``` makefile
$(dir <names>)
```

从文件名序列names中取出所有文件的目录部分，如果文件名不包含任何路径，则取出的是“./”。返回文件序列的目录部分。

### 取文件函数

``` makefile
$(notdir <names>)
```

从文件名序列names中取出所欲的非目录部分，返回取出的非目录部分。

### 取后缀函数

``` makefile
$(suffix <names>)
```

从文件名序列names中取出各个文件的后缀名，如果文件名没有后缀，则取出结果为空，返回取出的后缀名序列。

### 取前缀函数

``` makefile
$(basename <names>)
```

从文件名序列names中取出各个文件名的前缀部分（包含路径），返回取出的前缀名序列。

### 添加后缀名函数

``` makefile
$(addsuffix <suffix>, <names>)
```

将后缀suffix添加到文件名列表names中的每个单词后面。返回添加后缀后的文件名序列。

### 添加前缀名函数

```makefile
$(addprefix <prefix>, <names>)
```

将前缀prefix添加到文件名序列names中的每个单词前面，返回添加前缀之后的文件名序列。

### 链接函数

```makefile
$(join <list1>, <list2>)
```

将list2中的单词一一对应的拼接到list1中的单词后，如果list1的单词比list2多，那么list1后面多出来的单词保持不变。如果list1的单词比list2少，那么list2后面多出来的单词保持不变。返回拼接之后的单词序列

### 获取匹配模式文件名函数

```makefile
$(wildcard PATTERN)
```

列出当前目录下所有符合模式的PATTERN格式的文件名，返回由空格分隔的当前目录下所有符合PATTERN模式的文件名。

### Exam

```makefile
OBJ = $(dir src/main.c src/module1.c inc/module1.h bsp.h)
all:
	@echo $(OBJ)	#显示 src/ src/ inc/ ./
	
OBJ = $(notdir src/main.c src/module1.c inc/module1.h bsp.h)
all:
	@echo $(OBJ)	#显示main.c module1.c module1.h bsp.h
	
OBJ = $(suffix src/main.c src/module1.c inc/module1.h bsp.h)
all:
	@echo $(OBJ)	#显示.c .c .h .h
	
OBJ = $(basename src/main.c src/module1.c inc/module1.h bsp.h)
all:
	@echo $(OBJ)	#显示 src/main src/module1 inc/module1 bsp

OBJ = $(addsuffix .c, src/main src/module1)
all:
	@echo $(OBJ)	#显示 src/main.c src/module1.c
	
OBJ = $(addprefix src/, main.c module1.c)
all:
	@echo $(OBJ)	#显示 src/main.c src/module1.c
	
OBJ = $(join main module1 module2, .c .h)
all:
	@echo $(OBJ)	#显示 main.c module1.h module2
	
OBJ = $(wildcard *.c *.h)
main : $(OBJ)
	gcc -o $@ $^
	
```

## Makefile其他常用函数

### 遍历函数

``` makefile
$(foreach <var>, <list>, <text>)
```

把参数list中的单词逐一取出放到参数var所指定的变量中，然后再执行text所包含的表达式，最终遍历完成后返回由空格分隔的遍历执行结果。

**Notice** foreach函数中的var是一个临时变量，作用域只在该函数中，执行结束后就不再起作用。

```makefile
name = main module1 module2
files = $(foreach n, $(name), $(n).o)
all:
	@echo $(files)  #显示 main.o module1.o module2.o
```

### 条件执行函数

```makefile
$(if <condition>, <then-part>)
$(if <condition>, <then-part>, <else-part>)
```

当condition为真，则执行then-part部分，否则执行else-part部分，返回执行结果，如果condition为假且else-part为空，则返回空字符串。condition为真的条件为condition为非空字符串。

```makefile
OBJ = foo.c
OBJ = $(if $(OBJ), $(OBJ), main.c)
all:
	@echo $(OBJ)		#显示 foo.c
```

### 参数替换函数

```makefile
$(call <expression>, <parm1>, <parm2>, <parm3>, ...)
```

expression是一个包含参数的表达式，但call被执行时，expression中的参数变量$(1)，\$(2)，\$(3)等会被后面的参数parm1， parm2, parm3依次取代。最终返回替换完之后expression的值。

``` makefile
files = $(2).c $(1).c
obj = $(call files, main, module)
all:
	@echo $(obj)	#显示 module.c main.c
```

### 变量属性函数

```makefile
$(origin <variable>)
```

origin 函数不会操作变量的值，他只会返回这个变量的来源。这里variable是变量的名字，不应该是引用，最好不要在variable中使用$字符。下面是origin函数的返回值：

| 返回值       | 说明                                                 |
| ------------ | ---------------------------------------------------- |
| undefined    | variable从没有定义过                                 |
| default      | variable是默认定义的变量，如CC                       |
| environment  | variable是一个环境变量，并且makefile执行时没有-e选项 |
| file         | variable是在Makefile中定义的变量                     |
| command line | variable这个变量是被命令执行的                       |
| override     | variable是被override指示符重新定义的                 |
| automatic    | variable是一个命令行中的自动化变量                   |

## Makefile的文件包含与嵌套执行

 Makefile中可以通过include关键字来包含其他文件，当make命令遇到include关键字时，会暂停读取当前的Makefile，而是取读取include包含的文件，读取结束后再继续运行Makefile文件。具体用法如下：

```makefile
include <filenames>
-include <filenames>
```

上述两种使用方式的区别在于：

- 使用 `include <filenames>` ，make 在处理程序的时候，文件列表中的任意一个文件不存在的时候或者是没有规则去创建这个文件的时候，make 程序将会提示错误并保存退出。
- 使用 `-include <filenames>`，当包含的文件不存在或者是没有规则去创建它的时候，make 将会继续执行程序，只有真正由于不能完成终极目标重建的时候我们的程序才会提示错误保存退出。

**Notice** :使用include包含进来的 Makefile 文件中，如果存在函数或者是变量的引用，它们会在包含的 Makefile 中展开。

### Makefile嵌套执行的两种方式

```makefile
subsystem:
	cd subdir && $(MAKE)
```

上述脚本使当前的make命令切换目录到指定的目录subdir，该目录下也有一个Makefile文件用于描述subdir下文件的编译规则，然后会再该目录下执行make命令，执行完之后再返回外层的make执行中。这样实现了make的嵌套执行，最外层的Makefile一般称为总控Makefile。

另一种写法是：

```makefile
subsystem:
	$(MAKE) -C subdir
```

make的嵌套执行过程中，有一个系统变量“CURDIR”，它表示make的工作目录。当使用-C选项时，命令就会进入指定的目录中，此变量也会被重新赋值。

### Make嵌套执行时的参数传递

使用make嵌套执行时，如果需要传递变量，可以如下使用：

```makefile
export <variable>
```

如果需要传递所有变量，直接使用export不添加变量名即可。

有两个变量SHELL和MAKEFLAGS在不管是否使用export关键字的情况下都会传递给被嵌套的Makefile。

