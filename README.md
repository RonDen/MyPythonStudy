# MyPythonStudy
This is my Python study files. Something about the codes, PDF files or other test files


## Welcome to my github.

This is my first use of github, and I use it for my python study because I am learning Python and I like it very much.
It will be fine if you study with me togecher.

### 用numpy实现全连接网络并且在MNIST数据集上做测试

MNIST.py用于数据加载，实现了Loader, ImageLoader等类，main()中测试能够将手写数字数据集加载并且打印出来

DNN.py用于实现神经网络，分别实现了Sigmoid激活函数类，单层全连接层FullyConnectedLayer，以及神经网络类Network

main.py用于测试手写数字识别效果。读取6000笔数据进行训练，1000笔数据进行测试，在我的电脑上（i5，内存4G）平均每个epoch跑20秒，20次epoch就可以结束（用的是while循环，当错误率降低至0.1时停止循环）。
