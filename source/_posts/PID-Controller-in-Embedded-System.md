---
title: PID Controller in Embedded System
date: 2022-10-01 15:37:32
categories: work
mathjax: true
katex: true
tags: [embedded, C/C++]
---

## PID Introduction

### The model of PID controller

$$
u(t)=K_pe(t)+K_i\int e(t)dt+K_d \frac{\mathrm{d}e(t)}{\mathrm{d}t}
$$

Where u is the output of PID controller and e is the error of reference and feedback. Kp, Ki and Kd are the parameters of controller, which represents propotion, integral and derivative.

<!--more-->

### Discretization of PID controller

There are many methods to discretize a controller. We deploy the backward discretization method to the PID controller as following equation: 
$$
u(k) = K_pe(k)+K_iT_s\sum_{i=1}^k e(i) + \frac{K_d}{T_s}(e(k)-e(k-1)) \\
u(k)= K_pe(k)+U_i(k) + \frac{K_d}{T_s}(e(k)-e(k-1))	\\
u(k)	= K_pe(k)+K_iT_se(k) + \frac{K_d}{T_s}(e(k)-e(k-1))+U_i(k-1)
$$
where Ui represent the integral or sum of the error of past. And the controller after discretization as equation (4) is called "Position type PID controller". As the equation show, the output of controller is relevant with all of the past errors, so  it's not convenient to implement this controller in embedded software.

### From integral to increment

To simplifiy the controller's implementation, we calculate the difference of output of kth step and (k-1)th step as following equation:
$$
\Delta u(k) = u(k)-u(k-1)  \\
\Delta u(k) = K_p[e(k)-e(k-1)]+K_iT_se(k)+ \frac{K_d}{T_s}(e(k)-2e(k-1)+e(k-2)) \\
\Delta u(k) = (K_p+K_iT_s+\frac{K_d}{T_s})e(k)-(K_p+\frac{2K_d}{T_s})e(k-1)+\frac{K_d}{T_s}e(k-2)
$$
According to above equation, we get:
$$
\begin{align}
u(k) &=u(k-1)+(K_p+K_iT_s+\frac{K_d}{T_s})e(k)-(K_p+\frac{2K_d}{T_s})e(k-1)+\frac{K_d}{T_s}e(k-2)\\
	&=u(k-1)+\alpha e(k) - \beta e(k-1) + \gamma e(k-2)   \\
\end{align}
$$
Where:
$$
\begin{align}
\alpha &= K_p+K_iT_s+\frac{K_d}{T_s}\\
\beta &=K_p+\frac{2K_d}{T_s} \\
\gamma &= \frac{K_d}{T_s}
\end{align}
$$
As equation (14) shows, we got a controller model and the output is only relevant to the output of  last step and the error of current step and last 2 steps. Wc can call it "increment type pid controller" We don't need to calculate the sum of error of all past steps, so it's simple and costless to impement this controller in our embedded software.

## PID controller implementation

### Interface

We can define a PID controller module in C:

```c

typedef struct PID_REGULATOR
{
    float   ref;        /*!< reference value */
    float   fed;        /*!< feedback value  */
    float   err;        /*!< error of reference and feedback value */
    float   err_2;			/*!< error of reference and feedback of last time*/
    float   out;        /*!< output of pi regulator */

    float   alpha;      /*!< alpha parameter */
    float   beta;       /*!< beta parameter */
    float   gamma;			/*!< gama parameter */
  
  	float   step_max;		/*!< step increment upper limit>*/
    float		step_min;		/*!< step increment lower limit>*/
    float   out_max;    	/*!< output upper limit */
    float   out_min;    	/*!< output lower limit */
} pid_regu_type;
```

And we need to implement following operation interface of PID controller:

```c
/**
* func: create a pid controller 
* ret: a pid controller pointer
* param: none
*/
pid_regu_type* create_pid_regu(void);

/**
* func: init a pid controller 
* ret: none
* param: 
*				pid: PID controller pointer to be initialized
* 			others: thoes parameter of controller 
*/
void init_pid_regu(pid_regu_type *pid, float alpha, float beta, float gamma, float step_max, 
float step_min, float out_max, float out_min);

/**
* func: set control parameters of  pid controller 
* ret: none
* param: 
*				pid: PID controller pointer to be set
* 			alpha, beta, gamma: control parameters of controller as PID model defined 
*/
void  set_pid_para(pid_regu_type *pid, float alpha, float beta, float gamma);

/**
* func: set limit parameters of  pid controller 
* ret: none
* param: 
*				pid: PID controller pointer to be set
* 			step_max, step_min: the upper and lower limit of increment of controller output
* 			out_max, out_min: the upper and lower limit of controller output
*/
void set_pid_limits(pid_regu_type *pid, float step_max, float step_min, float out_max, float out_min);

/**
* func: calculate the output of pid controller
* ret: output of controller of current step
* param: 
* 		pid: PID controller pointer
* 		ref: reference value(target) of controller
*.    fed: feedback value
*/
float calc_pid_output(pid_regut_type *pid, float ref, float fed);
```

We can encapsulate the definition and interface in a header file which may be named "pid_regu.h". And then we can implement those interface in a source file "pid_regu.c". Finnally we got a pid contoller module for common use. 

May me we can use the PID controller like following:

```c
#include "pid_regu.h"

// we define a pid controller handler
pid_regu_type *speed_ctrl;

// We create and init a pid controller for special use
void ctrl_init()
{
  // ... 
	speed_ctrl = create_pid_regu();
	init_pid_regu(speed_ctrl, 1.1, 1, 0, 1, -1, 100, -100); 
  // ...
}

// for most, we update the control output in control period interrupt
void ctrl_int_callback() 
{
  //...
  calc_pid_output(speed_ctrl, speed_target, current_speed);
  //...
}
```

### Go further

We derive the increment type PID controller and implement it in  C language. However, many real controllers are based on PID control, but more complicated than simple PID control. There may be more limitations or requirements in different applications. So we can inherit PID controller module to other controllers where the PID controller is the base class of a real controller. Although inheritance can be implemented in C, it's more simple and natural to use it in C++. So we can implement the PID controller as a base class of real controllers in c++.

  

