# Forward Network

![BP神经网络](https://camo.githubusercontent.com/d4ab04e0d1f25654a3a7ced664613432496a02a7cc748b07879879725bf2ca6e/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746875622f42504e4e3f69636f6e266c6162656c3d476974487562)![C++](https://camo.githubusercontent.com/0b73e94abcf56828e223f41e42c2198aa8414582a80e229307cf7502ca5848a6/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f737570706f72742d4325324225324231312532306f722532306c617465722d626c75653f7374796c653d666c6174266c6f676f3d63706c7573706c7573)

本项目有一定程度的代码结构借鉴！

本项目将于明年重构

如需测试，请自行下载**<u>[MNIST](http://yann.lecun.com/exdb/mnist/)</u>**数据集！并放在main.cpp同一个目录下

## sln解决方案结构描述

```c++

├── slnBackNN                 
├── 外部依赖项                   
├── 头文件                      // 声明
│   ├── funct.h
│   ├── layer.h                
│   └── process.h       
├── 源文件                      // 定义
│   ├── funct.cpp
│   ├── layer.cpp               
│   ├── main.cpp               // 主函数
│   └── process.cpp  
├── 资源文件                    // MNIST数据集
│   ├── t10k-images.idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
└── ..
```

## 基本描述

### 全连接神经网络在MNIST上的实现

#### 文件功能介绍

funct.h:声明了一些辅助函数,如激活函数等
layer.h:声明了核心的神经网络层Layer类
process.h:声明了读取MNIST数据集的函数
funct.cpp:实现了辅助函数
layer.cpp:实现了Layer类,包含网络的训练和预测函数
main.cpp:程序入口,组织网络的训练和预测
process.cpp:实现了读取MNIST数据集的函数

#### 程序设计

本程序通过Layer类实现了一个包含输入层、隐层和输出层的全连接神经网络。

network的训练和预测在Layer类的train()和predict()函数中实现。

辅助函数如激活函数在funct.cpp中实现。

MNIST数据集的读取和预处理在process.cpp中实现。

main.cpp作为入口,组织调用过程。

#### 运行方法

直接编译main.cpp和其他文件,然后运行可执行文件,程序将自动开始训练和测试过程。

训练完成后,会输出在测试集上的损失。

#### 结果示例

训练迭代300个样本,损失下降到0.081
在测试集上平均损失为0.097

#### 总结

该项目实现了一个特别简单的（300batch-1epoch），可以在MNIST数据集上训练和测试的全连接神经网络模型。

### 结语（有感而发）

本项目于11月1日正式开始，11月6号完成搭建；为什么想要搭建MNIST手写数字数据集识别？因为这任务被称为深度学习的Hello World。

由于C++基础知识遗忘完了以及期中考试临近，所以被迫简化了原本打算实现的mini-batch学习，最终简化成了这种超级半吊子版本。离开了numpy真就什么都不行了。。。

最初打算通过Cmake搭建项目，捣鼓了一天还捣鼓不明白CmakeList.txt。。。

我有意保留了代码中的部分不成熟片段。算是人生第一个小项目了，代码并不是完全原创。踩了很多的雷，还记得当年第一次跑同MNIST数据集读取时，我是多么激动（好吧我承认，这个读取也有一部分的借鉴）也获得了很多经验。也感叹如今LLM的发达程度，起码Chatgpt3.5陪我一起面临数十项的报错。

路漫漫其修远兮。