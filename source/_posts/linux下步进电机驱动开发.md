---
title: linux下步进电机驱动开发
date: 2023-03-23 22:51:56
categories: Learning
tags: 
  - Linux
  - embedded software
---

## 功能需求

​	最近项目中需要在嵌入式Linux系统中直接控制步进电机，需要开发Linux系统下的步进电机驱动，实现控制步进电机向前，向后运动指定步数的驱动API接口。每个步进电机有三个硬件接口，一个使能IO口通过电平控制步进电机的使能， 一个方向控制IO口通过电平控制步进电机的运动方向，一个PWM驱动接口驱动电机运动。在项目中的设备需要同时控制多个步进电机，因此开发的驱动也需要支持多个步进电机设备同时挂载。

##  分析

​		步进电机在Linux中属于字符设备，控制步进电机的运动(向前，或向后)和停止等API可以通过实现字符设备的文件操作函数来实现。这里选择实现文件操作中的ioctl函数，通过定义ioctl不同的命令来对应我们需要的不同操作。当然也可以通过字符设备文件的写函数write来实现，通过向步进电机的字符设备文件中写入相应的命令实现需要的操作。

​		其次，针对驱动程序需要同时支持多个步进电机的需求，可以采用设备树加platform总线的形式。将步进电机挂载到platform总线上，通过platform的probe函数自动解析设备树文件，得到挂载的步进电机设备信息，从而根据设备树文件中的定义加载指定数量的驱动，配置好驱动需要用到的硬件资源。

## 设备树文件的编写

前面提到了每个步进电机有三个硬件接口需要指定，其中包括1个使能GPIO口，1个方向GPIO口，1个PWM输出。这里PWM输出可以直接使用硬件PWM模块，也可以通过软件定时器去控制GPIO口实现模拟PWM功能。

因此设备树文件中对于一个步进电机的定义如下：

```c
\ {
    stepper_motor {
        compatible = "stepper-motor";
        status = "okay";
        enable-gpio = <&gpio0 0 GPIO_ACTIVE_HIGH>;
        direction-gpio = <&gpio0 1 GPIO_ACTIVE_HIGH>;
        pwm = <&pwm0 0 1000000 0>;					// 使用硬件PWM模块
        // pwm-gpio = <&gpio0 2 GPIO_ACTIVE_HIGH>;	// 使用GPIO模拟PWM
    }；
}；
```

<!--more-->

考虑到要同时支持多个步进电机驱动，可以进一步将步进电机虚拟成一个总线设备，上面挂载多个步进电机设备子节点。例如同时挂载三个步进电机的设备树文件如下:

```c
\ {
    stepper_motor {
        compatible = "stepper-motor";
		status = "okay";
		#address-cells = <1>;
		#size-cells = <0>;
   		stepper_motor0@0 {
            enable-gpio = <&gpio0 0 GPIO_ACTIVE_HIGH>;
        	direction-gpio = <&gpio0 1 GPIO_ACTIVE_HIGH>;
        	pwm = <&pwm2 0 1000000 0>;	
        };
        stepper_motor1@1 {
            enable-gpio = <&gpio1 0 GPIO_ACTIVE_HIGH>;
        	direction-gpio = <&gpio1 1 GPIO_ACTIVE_HIGH>;
        	pwm = <&pwm2 0 1000000 0>;	
        }；
        stepper_motor2@2 {
            enable-gpio = <&gpio2 0 GPIO_ACTIVE_HIGH>;
        	direction-gpio = <&gpio2 1 GPIO_ACTIVE_HIGH>;
        	pwm = <&pwm2 0 1000000 0>;	
        };  
    }；
}；
```

在上面的设备树文件中，三个步进电机stepper_motor0,1,2作为stepper_motor节点下的3个子节点挂载到设备树中。这样就可以在stepper_motor的驱动程序中同时对设备树文件中定义的多个设备进行解析。上面写的GPIO口编号，PWM模块都是随便写的例子，使用时根据实际的硬件连接情况设置即可。

## platform设备probe函数的实现

将步进电机做为platform设备挂载到platform总线上，最重要的步骤就是实现步进电机platform 驱动的probe函数。通过设备树文件中stepper_motor节点的compatible属性，platform总线可以自动将步进电机驱动match到该设备节点，从而调用驱动的probe函数，在probe函数中需要完成设备节点的解析，获取和分配设备需要的各种资源并对设备进行初始化。

针对上面的设备树文件，可以编写步进电机驱动的probe函数如下：

```c
static int stepper_motor_probe(struct platform_device *pdev)
{
    int ret;
    struct stepper_motor_platform *plat;
    struct device_node *np, *pp;
    int n_stepper_motors;
    unsigned int stepper_motor_major;
    int i;
    dev_t devno;

    /* 获取设备树节点 */
    np = pdev->dev.of_node;
    printk("probe stepper motor!\n");
	
    /* 获取挂在的步进电机的个数(字节点个数) */
    n_stepper_motors = of_get_child_count(np);
    if (n_stepper_motors == 0) {
        printk("no device node!\n");
        return -ENODEV;
    }

    /* 申请总线设备结构体内存并初始化 */
    plat = devm_kzalloc(&pdev->dev, sizeof(struct stepper_motor_platform), GFP_KERNEL);
    if (!plat)
        return -ENOMEM;
    plat->count = n_stepper_motors;
    plat->pdev = pdev;
	
    /* 申请步进电机设备结构体内存 */
    stepper_motor_devp = kzalloc(n_stepper_motors * sizeof(struct stepper_motor_device), GFP_KERNEL);
    if (!stepper_motor_devp) {
        printk("stepper motor: failed to alloc stepper motor.\n");
        return -ENOMEM;
    }       
    plat->data = stepper_motor_devp;

    /* 保存设备结构体指针，在释放设备时需要使用 */
    platform_set_drvdata(pdev, plat);

    /* 将每一个步进电机注册为字符设备 */
    /* 申请字符设备号 */
    ret = alloc_chrdev_region(&devno, 0, n_stepper_motors, STEPPER_MOTOR_NAME);
    stepper_motor_major = MAJOR(devno);
    if (ret < 0) {
        dev_err(&pdev->dev, "Failed to allocate chrdev region\n");
        return ret;
    }    
    for (i = 0; i < n_stepper_motors; i++) {
        struct stepper_motor_device *devp;       
        devp = &(stepper_motor_devp[i]);
        devp->devno = MKDEV(stepper_motor_major, i);
        stepper_motor_setup(devp, i);			// 注册为字符设备
    }
    
    /* 根据设备树中的步进电机节点信息申请和分配相应的资源 */
    i = 0;
    for_each_child_of_node(np, pp) {
        stepper_motor_get_data(&(stepper_motor_devp[i]), pp);	// 根据字节点信息配置设备
        i++;
    }
    
    dev_info(&pdev->dev, "stepper motor driver initialized\n");
    return 0;
}
```



## 步进电机字符设备文件操作接口的实现

前面在功能需求中提到了，步进电机的驱动程序需要提供如下操作函数接口：

1. 向前运动指定步数
2. 向后运动指定步数
3. 停止运动

步进电机控制实现的逻辑很简单，控制步进电机运动时，只要将步进电机的使能GPIO设置为有效电平，将方向GPIO按照指定方向设置高低电平，然后使能PWM模块输出脉冲。延时指定步数的PWM脉冲周期即可。

当然，下面的驱动程序中使用阻塞的方式等待步进电机运动完指定的步数，在步进电机运动过程中其实是无法停止的。其实可以采用非阻塞的方式去实现，而不用在驱动程序中去等待， 下次再写了。 

通过字符设备的文件操作接口，ioctl即可实现以上三个功能，ioctl函数实现如下：

```c
/* 按照ioctl的规范定义我们需要实现的操作对应的命令 */
#define STEPPER_IOC_MAGIC               's'
#define STEPPER_MOTOR_CMD_STOP          _IO(STEPPER_IOC_MAGIC, 0)
#define STEPPER_MOTOR_CMD_FORWARD       _IOW(STEPPER_IOC_MAGIC, 1, int)
#define STEPPER_MOTOR_CMD_BACKWARD      _IOW(STEPPER_IOC_MAGIC, 2, int)


/* 实现步进电机按照指定方向运动指定步数 */
static void stepper_motor_run(struct stepper_motor_device *dev, unsigned int dir, unsigned long nsteps)
{
    unsigned long delay_time_us;
    unsigned long delay_time_ms;
    
    gpiod_set_value(dev->direction_gpio, dir);
    gpiod_set_value(dev->enable_gpio, STEPPER_MOTOR_ENABLE);
    pwm_config(dev->pwm, PWM_DUTY_NS, PWM_DUTY_NS * 2);
        
    delay_time_us = PWM_DUTY_NS / 1000 * nstpes;
    if (delay_time_us > 5000) {
        delay_time_ms = delay_time_us / 1000;
        delay_time_us = delay_time_us - delay_time_ms * 1000;
    }
    pwm_enable(dev->pwm);
    
    if (delay_time_ms > 0)
        msleep(delay_time_ms);
    if (delay_time_us > 0)
        udelay(delay_time_us);
    
    pwm_disable(dev->pwm);

    gpiod_set_value(dev->enable_gpio, STEPPER_MOTOR_DISABLE);
}

/* 停止步进电机运动 */
static void stepper_motor_stop(struct stepper_motor_device *dev)
{
    pwm_disable(dev->pwm);
    gpiod_set_value(dev->enable_gpio, STEPPER_MOTOR_DISABLE);
}

/* 字符设备驱动的ioctl函数实现 */
static long stepper_motor_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct stepper_motor_device *dev = filp->private_data;
    int ret = 0;
    switch (cmd) {
    case STEPPER_MOTOR_CMD_STOP:
        stepper_motor_stop(dev);
        break;
    case STEPPER_MOTOR_CMD_FORWARD:
        stepper_motor_run(dev, STEPPER_MOTOR_DIR_FORWARD, arg);
        break;
    case STEPPER_MOTOR_CMD_BACKWARD:
        stepper_motor_run(dev, STEPPER_MOTOR_DIR_BACKWARD, arg);
        break;
    default:
        return -EINVAL;
    }
    return 0;
}
```

这里需要注意的有两点：

1. 由于需要支持同时驱动多个步进电机设备，因此需要在ioctl中获取到设备文件对应的步进电机设备结构体。一般驱动程序都会把设备结构体作为设备文件的private_data，这样在设备文件的操作函数中就可以获取到设备对应的设备结构体了。因此在实现步进电机的open函数时，需要将驱动中定义的步进电机设备结构体绑定到文件的private_data。如下所示：

```c
static int stepper_motor_open(struct inode *inode, struct file *filp)
{
    /* 根据文件inode节点获取到对应设备驱动中的设备结构体 */
    struct stepper_motor_device *dev = container_of(inode->i_cdev, struct stepper_motor_device, cdev);
    filp->private_data = dev;
    printk("open stepper motor%d\n", MINOR(dev->devno));
    return 0;
}
```

2. ioctl的控制命令的定义不要随便define，要按照ioctl接口的规范去定义，可以直接通过Linux提供的ioctl命令定义宏来很方便的定义不同类型的命令，如前面程序中用到的_IO, _IOW宏 。具体的ioctl命令定义规范可以查看相关文档资料。
