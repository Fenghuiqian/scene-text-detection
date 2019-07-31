# Scene-Text-Detection
阿里云-网络图像的文本检测

https://tianchi.aliyun.com/competition/entrance/231685/introduction?spm=5176.12281949.1003.1.493e28ee8FHyCF
## 环境
1. Ubuntu18.04, GPU Driver Version: 390.116
2. tensorflow-gpu=1.9.0
3. tensorflow object-detection API, https://github.com/tensorflow/models.
4. seglink安装， https://github.com/dengdan/seglink.git
5. 安装pylib包， https://github.com/dengdan/pylib， 包的路径放入PYTHONPATH
## 使用
1. 运行convert_to_tfrecords.py,将数据集转换成tf seglink版本所需格式。
2. 网盘下载ICDAR2015预训练的模型，用于finetune, https://pan.baidu.com/s/1slqaYux#list/path=%2F.
3. 模型训练参数，seg_conf_threshold=0.5, link_conf_threshold=0.5, batch_size=4(8G GPU), learning_rate=10e-4, image_size=512,512
4. finetune过程可通过tensorboard的loss曲线与不同阶段step的checkpoint预测的结果，直接提交阿里云竞赛做验证，以此判断finetune的效果。此seglink pkg也有验证脚本，也可从训练集中抽出部分用作验证集自行验证。
5. 测试参数，seg_conf_threshold=0.8, link_conf_threshold=0.5
## 结果
Leaderboard top12%
## 示例
![img_1](https://github.com/Fenghuiqian/scene-text-detection/blob/master/test_examples/1.png)
![img_2](https://github.com/Fenghuiqian/scene-text-detection/blob/master/test_examples/2.png)
![img_3](https://github.com/Fenghuiqian/scene-text-detection/blob/master/test_examples/3.png)
![img_4](https://github.com/Fenghuiqian/scene-text-detection/blob/master/test_examples/4.png)
loss曲线训练约6个epoch(训练集10000imgs)后中止，继续训练可能会有更好的收敛。
