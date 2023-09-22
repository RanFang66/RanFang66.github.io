---
title: Keras入门学习
date: 2023-09-23 01:10:23
categories: Learning
tags: 
  - Keras
  - MachineLearning
---
# Keras快速入门

## 深度学习对于数据预处理的要求

神经网络不能直接处理原始数据，比如原始的文本，图像，表格等原始数据，需要将原始数据导入并转换为向量化(vectorized)和标准化(standardized)的表示。

- 文本文件需要被读取成字符串张量，然后再分割成单词，最终这些单词还需要被索引化并转换为整数张量。
- 图像需要被读取并解码成为整形张量，然后转换为浮点型并正则化到较小范围的数值（通常使用0到1之间的小数）
- csv类型的数据(特征数据)在导入时需要被解析，对于数值型特征转化为浮点型张量，分类特征索引化为整数张量。并且每一个特征都需要正则化(zero-mean & unit-variance)

## Keras 数据导入

keras模型支持三种类型的输入：

- numpy 数组
- tensorflow dataset 对象
- Python generators

如果训练数据很大，需要使用gpu，优先考虑使用dataset对象，因为这样能够在性能上得到优化。

从文件夹导入图片数据：

```python
dataset = keras.utils.image_dataset_from_directory(
	'path/to/main_directory', batch_size = 64, image_size(200, 200))
```

从文件夹导入文本数据:

```python
dataset = keras.utils.text_dataset_from_directory(
  'path/to/main_directory', batch_size = 64)
```
<!--more-->
## Keras数据预处理

将数据导入之后，需要进行预处理，一般包括以下的流程：

- 数据标签化和索引化
- 特征正则化(normalization)

### 理想化的机器学习模型应该是端到端(end-to-end)的

在搭建机器学习模型时，应当尽可能的将数据预处理过程整合到模型到中，而不是作为模型之外的额外处理流程。因为如果数据的预处理过程不包含在模型中，在进行模型的移植时就会很麻烦，不仅需要重新实现预处理流程，还容易出错或降低模型性能。

因此理想的机器学习模型的输入就是接近原始数据例如对于文本处理的模型，它的输入是utf-8编码的字符串，对于图像处理的模型，它的输入是8位3通道的rgb像素值等等。

### keras中的数据预处理层

在keras中，可以通过keras提供的预处理层进行模型内的数据预处理，包括:

- 通过TextVectorization层向量化文本字符串
- 通过Normalization层实现特征的归一化
- 实现图像数据的裁剪，改变大小或者图像增强

使用keras的预处理层的最大好处是这些层可以直接被包括到机器学习模型当中。

一些预处理层包含着状态信息，例如：

- TextVectorization 层维护了一个索引表
- Normalization层 维护了特征的平均值和方差

通过layer.adapt(data)可以获取预处理层在数据data上的状态信息。

``` python
import numpy as np
from tensorflow import keras
from keras.layers import TextVectorization
training_data = np.array([["This is the 1st sample"], ["Here is the 2nd sample"]])
vectorizer = TextVectorization(output_mode="int")
vectorizer.adapt(training_data)
integer_data = vectorizer(training_data)
print(integer_data)
```

## Keras 构建模型(Functional API)

一个"层"(layer)是简单的输入-输出变换，例如下面是一个将输入线性映射到16特征空间输出的层:

``` python
dense = keras.layers.Dense(units = 16)
```

一个"模型"(model)是层的有向无环图(directed acyclic graph)，也可以看成是由很多子层组成的更大的层。

最常见的构建keras模型的方式是通过调用Functional API。开始时，需要指定模型的输入形状(开可以指定数据类型)，如果输入形状中的某一个维度是可变的，用None指定即可。例如指定模型的输入为任意大小的3通道rgb图像：

``` python
inputs = keras.Input(shape=(None, None, 3), dtype = np.uint8)
```

在定义好模型的输入之后，就可以从输入层之后一层层的添加需要的层，直到最后的输出层:

```python
# preprocessing layers 
x = layers.CenterCrop(height = 150, width = 150)(inputs)
x = layers.Rescaling(scale = 1.0 / 255)(x)

# convolution and pooling layers
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation = "relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation = "relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation = "relu")(x)

# global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()(x)

# add a dense classifier on top
num_classes = 10
outputs = layers.Dense(num_classes, activation="softmax")(x)
```

在完成从输入到输出的所有层的定义后，就可以通过定义的输入和输出去实例化一个keras的Model对象:

``` python
model = keras.Model(inputs = inputs, outputs = outputs)
```

这个实例化的模型可以看成是一个更大的层，可以直接用它调用数据进行处理:

```pyt
data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")
processed_data = model(data)
print(processed_data)
model.summary()
```

通过调用model.summary()函数可以输出模型的整体信息。

## 模型训练

### 为模型指定优化器和损失函数

keras的Model类提供内置的训练接口fit()方法，fit()方法接受Dataset对象，python generators或者numpy数组作为输入进行模型的训练。

在使用fit()训练之前，需要制定optimizer和损失函数，这一步是通过compile接口完成的：

```python
model.complie(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3), loss=keras.losses.CategoricalCrossentropy())
```

模型的优化器和损失函数还可以通过字符串名称去指定，这种情况下，模型的优化器和损失函数在构造时使用的都是默认参数

```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
```

### 训练模型

在通过compile指定优化器和损失函数之后，就可以使用fit()进行模型的训练，在使用fit进行训练时，除了指定训练的输入数据(输入和输出)之外，一般还会指定训练的epochs和bacth_size。

```python
# training model with numpy array data
model.fit(numpy_array_of_samples, numpy_array_of_labels, batch_size = 32, epochs = 10)

# training model with dataset object, since data yielded by a dataset is expected to be already batched, you don't need to specify the batch size here
model.fit(dataset_of_samples_and_labels, epochs = 10)
```

fit()函数会返回一个history对象，history对象记录了在模型训练期间发生的事件。history.history维护了一个字典，这个字典里记录了每一个epoch的时间序列和训练指标值(默认包含loss)。

### 监控模型训练时的性能指标

在训练模型时，我们通常希望能够持续监控模型的各项性能指标，比如分类的准确性，召回率，精度等等。而且希望不仅能够监控模型在训练集数据上的性能，还能监控模型在验证集数据上的性能。在调用模型的compile()借口时，可以通过传入metrics来指定想要监控的性能指标。例如：

```python
model.compile(optimizer = 'adam',
             loss='sparse_categorical_crossentropy',
             metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')])
history = model.fit(dataset, epochs = 10)
```

通过将验证集数据传入到fit()中，还可以监控模型在验证集上的损失和其他性能指标。验证集上的性能指标会在每一个epoch完成时报告。

```python
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
history = model.fit(dataset, epochs=10, validation_data=val_dataset)
```

### 定期保存模型训练的状态

模型的训练通常比较耗费时间，训练过程中也有可能因为各种问题中断。定期保存模型的训练状态，并且能够从某一个状态重新开始训练模型在实际训练时非常有用。

Keras提供了一个非常有用的特性callbacks，通过fit()函数中的callbacks参数可以设定模型训练时的callbacks。callbacks是在模型训练过程中的指定训练节点会调用的对象，通常可以选择调用的节点包括

- 每一个batch开始和结束的时候
- 每一个epoch开始和结束的时候

通过使用callbacks就可以周期性的保存模型，例如在下面的例子中， 一个modelCheckpoint callback就配置了模型在每一个epoch接受的时候保存模型，并且以当前的epoch去命名保存的模型文件。

```python
callbacks = [ keras.callbacks.ModelCheckpoint(filepath='path/to/my/model_{epoch}', save_freq = 'epoch')
]
model.fit(dataset, epochs=10, callbacks=callbacks)
```

除了定期保存模型外，还可以使用callbacks在模型训练过程中进行一些周期性事务的处理，比如改变优化器的学习率，输出性能指标到指定的文件，以及在模型训练结束时发送邮件通知等。

### 使用tensorboard监控模型训练过程

tensorboard是一个可以实时显示模型性能指标图表的网页应用，使用tensorboard可以更加高效的监控模型的训练过程。

通过在调用fit()时传入一个keras.callbacks.TensorBoard对象并指定tensorboard日志的存储目录，就可以使用tensorboard了。

```python
callbacks = [
  keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(dataset, epochs = 2, callbacks = callbacks)
```

通过如下的命令，就可以在浏览器中启动一个tensorboard实例用于监控所训练的模型，不仅如此，还可以直接在jupyter/colab中使用tensorboard。

```shell
tensorboard --logdir=./logs
```

## 评估和使用模型

在模型训练完成后，可以使用新的数据来评估模型的性能，在keras中使用evaluate()进行模型性能的评估，它会返回模型在指定输入数据上的损失和精度指标。

```python
loss, acc = model.evaluate(val_dataset)
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)
```

通过调用predict()可以使用训练好的模型去对新的数据进行预测，它会输出输入数据经过模型预测后的结果，也就是模型的output层的输出。

```python
predictions = model.predict(val_dataset)
print(predictions.shape)
```

