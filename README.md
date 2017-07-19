#Digit-Recognizer

- 安装dependencies

``` bash
# install dependencies
pip install -r requirements.txt
```

- 数据集简介：
><span style="font-size: 14px;">&#160; &#160; &#160; &#160;train.csv是训练集，共有42000行，748列。每一行对应一个手写的数字图片。图片以矩阵的方式存储，pixel为其像素。
此数据集中有42000个数字，第一列为其结果，后面都是其像素表达。</span>

&emsp;
><span style="font-size: 14px;">&#160; &#160; &#160; &#160;test.csv是测试集，为了检验结果与验证模型的准确率，模型必须在test上再跑一次。同时也是为了防止过拟合现象发生。</span>

- 框架的使用问题

&#160; &#160; &#160; &#160;本项目使用开源框架<font color=#0099ff>Tensorflow</font>
