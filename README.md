# face_detect
本项目旨在开发一个人脸识别系统，利用深度学习模型对输入的人脸图像进行分类和识别。模型将输入的人脸图像作为输入，并通过多个卷积层和池化层提取图像特征，最终通过全连接层进行分类，识别并标识出输入图像中的人物身份。

## Prerequisites

You need to install:
- [Python3 >= 3.6](https://www.python.org/downloads/)
- Use `requirements.txt` to install required python dependencies

    ```Shell
    # Python >= 3.6 is needed
    pip3 install -r requirements.txt
    ```

    
## Quick-start
1. Install python packages: 
   ```Shell
    pip3 install -r requirements.txt
    ```
2. Detection 
  ```Shell
    # inference using best model
    python inference.py
  ```

## Training
1. prepare data
    * 将数据集放在`dataset/images`文件夹中
    * 更改标签文件：`dataset/name.txt`
    * 准备训练标签：`dataset/train.txt`
    
2. run following command
    ```Shell
    python train.py
    ```
