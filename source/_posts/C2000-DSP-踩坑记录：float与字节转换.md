---
title: C2000 DSP 踩坑记录：float与字节转换
date: 2022-12-24 01:43:18
categories: Notes
tags:
    - DSP
    - embedded software
---

## 问题来源

由于在项目中需要用到E2PROM来保存一个浮点型数据，E2PROM读写的驱动都是按照字节来读写的，所以需要将float数据类型转换为字节序列，再写入E2。同理读取float数据时也要先读出字节序列，然后转换为float类型数据。

## 原程序

开始编写时图代码简单，直接使用指针强制类型转换的方式，代码如下：

<!--more-->

```c
#include <stdio.h>
#include <stdlib.h>

// float 类型转换为字符数组
void
float_2_byte(float f, unsigned char *s)
{
    unsigned char *byte;
    int i = 0;

    byte = (unsigned char*)&f;
    for (i = 0; i < 4; i++) {
        *(s+i) = *(byte+i);
    }
}

// 字符数组转换为float类型
float
byte_2_float(unsigned char *s)
{
    float *fp;

    fp = (float*)s;

    return *fp;
}

// 测试程序
int main(void)
{
    float f = 3.141582657;
    float f2 = 0.0;
    unsigned char s[4];

    float_2_byte(f, s);
    printf("%x\t%x\t%x\t%x\n", s[0], s[1], s[2], s[3]);

    f2 = byte_2_float(s);
    printf("float = %f\n", f2);

    return 0;
}

```

将上述代码再PC上测试了一下，输出结果如下：

![](/home/ran/gitRepos/RanFang66.github.io/source/_posts/C2000-DSP-踩坑记录：float与字节转换.assets/image-20211012154414486.png)

说明字节序和float类型之间的相互转换没有问题。

## 问题现象与原因

将上述程序移植到28335平台时，通过E2读写数据时，发现读出的数据与写入的不同。经过调试发现，实际写入的数据并没有按照预想的那样写入float型数据的各个字节。

原因分析：dsp28335的最小数据类型单位是16位的，28335平台上的unsigned char类型实际上是16位！当采用如上代码运行在28335平台上，将一个float类型数据0x12345678转换为unsigned char类型数组时，实际上得到的unsigned char类型数组为：

s[0] = 0x5678, s[1] = 0x1234, s[2] = 0x????, s[3] = 0x????。也就是说，直接采用unsigned char类型指针运算的方式，指针加1并不是移动8位，而是16位，因此采用指针去取字节出现了错误。

## 修改程序

采用指针进行暴力类型转换的方法在28335上行不通了。还可以采用位运算或者联合体的方式。这里直接采用位运算和类型转换的方式。

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void
float_2_byte(float f, unsigned char *s)
{
    uint32_t longdata = 0;

    longdata = *(uint32_t*)&f;

    s[0] = (longdata & 0xFF000000) >> 24;
    s[1] = (longdata & 0x00FF0000) >> 16;
    s[2] = (longdata & 0x0000FF00) >> 8;
    s[3] = (longdata & 0x000000FF);
}

float
byte_2_float(unsigned char *s)
{
    uint32_t longdata = 0;
    float f = 0.0;
	int i = 0;
    
    for (i = 0; i < 3; i++) {
        longdata += s[i];
        longdata <<= 8;
    }
    longdata += s[3];
    
    f = *(float*)&longdata;
    return f;
}

int main(void)
{
    float f = 3.1415;
    float f2 = 0.0;
    unsigned char s[4];

    float_2_byte(f, s);
    printf("%x\t%x\t%x\t%x\n", s[0], s[1], s[2], s[3]);

    f2 = byte_2_float(s);
    printf("float = %f\n", f2);

    return 0;
}

```

上述代码在移植到DSP28335平台之后，能够正确的实现float类型数据和字节序列之间的转换。

## 总结

之所以源程序在PC上没问题，而在DSP28335平台上出现bug的原因在于对于unsigned char类型的指针操作。在DSP28335上对于8位数据的指针操作与预想的是不同的，因为DSP28335中实际上是以16位的形式存储char和unsigned char类型数据。在对8位数据指针类型进行加减操作时，指针移动也是16位而不是8位。

