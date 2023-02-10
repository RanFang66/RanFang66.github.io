---
title: keras模型转换到TFLITE格式并部署到RK3588
date: 2023-02-10 16:29:45
categories: Notes
tags: 
  - MachineLearning
  - embeded
---

## keras模型的保存和加载

keras模型主要有以下几种保存的形式：

### save/load weights

```python
## 模型保存： 只保存权重参数，不保存模型网络结构
model.save_weights(‘./checkpoins/my_checkpoint’)

## 加载模型
model = create_model()
model.load_weights(‘./checkpoints/my_checkpoint’)
```

### save/load entire model

```python
# 保存整个模型
model.save('my_model.h5')

# 加载模型
model = tf.keras.models.load_model('my_model.h5')
```

### save_model

```python
# 保存模型到文件夹
tf.saved_model.save(model, './my_model/')

# 导出模型
model = tf.saved_model.load('./my_model/')
f = model.signatures['serving_default']
```

<!--more-->

## keras模型到TFLITE模型的转换

### h5格式文件的转换

这里要注意：

1. 转成tflite的keras模型是要通过save保存的完整模型，而不是通过save_weights保存的权重。
2. 先从保存的h5文件加载模型，然后在利用tf提供的TFLiteConverter工具转换成TFLite模型。
3. tf1和tf2提供的api接口是不一样的，tf1是from_keras_model_file('/path')， tf2是from_keras_mode(model).

```python
# tf2 将keras的h5格式模型文件转换为tflit格式

# 加载keras模型
model = tf.keras.models.load_model('my_model.h5')

# 定义converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 完成转换得到tflite模型
tflite_model = converter.convert()

# 保存tflite模型
open("my_tflite_model.tflite", "wb").write(tflite_model)
```

### savedModel格式的转换

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("my_tflite_model.tflite", "wb").write(tflite_model)
```



## TFLITE模型在RK3588上的部署

### rknn-toolkits2 开发套件

rockchip公司提供了rknn-toolkits2开发套件为用户提供在PC平台上进行模型转换、推理和性能评估的功能，用户通过该工具提供的python接口可以便捷的完成以下功能：

- 模型转换：支持Caffee, TensorFlow, TensorFlow Lite, ONNX, PyTorch, DarkNet等模型转换为RKNN模型，RKNN模型能够在RockchipNPU平台上加载使用。
- 量化功能：支持将浮点模型量化为定点模型。
- 模型推理：支持PC上模拟NPU环境进行推理或者直接通过USB通信在实际的NPU上进行推理。
- 性能和内存评估：评估模型的性能指标。
- 量化精度分析：给出量化前后的的推理结果差距，方便提高量化精度。
- 模型加密：使用指定加密等级对RKNN模型进行整体加密。

### rknn-toolkit2的安装

1. 通过pip install 命令安装
   1. 安转相关的依赖包(具体查看帮助文档)
   2. 通过rknn_toolkit2*.whl安转相应版本的rknn_toolkit2
2. 直接通过Docker镜像安装



### rknn-toolkit2的使用

通过rknn-toolkit2将其他格式的模型转换为rknn格式并调用的流程如下图所示：

![image-20230207181258720](/home/ran/gitRepos/RanFang66.github.io/source/_posts/keras模型转换到TFLITE格式并部署到RK3588/image-20230207181258720.png)

具体到tflite格式的模型的转换API就是load_tflite，该接口只有一个参数，即tflite格式模型文件的路径。

```python
import numpy as np
from rknn.api import RKNN

if __name__ == '__main__':
    # Create RKNN object
    rknn = RKNN(verbose=True)
    
    # Pre-process config
    print('-->Config model')
    rknn.config(mean_values=[128, 128, 128], std_values=[128, 128, 128])
    print('done')
    
    # Load model
    print('--> Loading model')
    ret = rknn.load_tflite(model = 'my_tflite_model.tflite')
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')
    
    # build model
    print('--> Buliding model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')
    
    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    
    # Inference
    print('-->Running model')
    outputs = rknn.inference(inputs=[img])
    print('done')
    
    # evaluate
    print('-->Evaluate model performance')
    rknn.eval_perf(inputs=[img], is_print=True)
    print('done')
    
```

