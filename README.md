# RKNN_learning_notes

本项目用于记录学习如何使用RKNN-tool2

## 快速开始

### 环境搭建

实测使用WSL2(Ubuntu-22.04)和鲁班猫5可以运行本项目

1. 新建WSL2环境之后，配置Python环境

```bash
# 在WSL2终端下
# 1. 安装RKNN依赖
pip install rknn-toolkit2 -i https://pypi.org/simple
# 2. 补充OpenCV库
pip install opencv-python  
# 3. 安装ADB工具用于连板
apt install adb
```

2. 配置鲁班猫环境

请先用网线连接鲁班猫和电脑，然后在电脑设置中，设置鲁班猫的网络，设置其IPv4地址为静态的 `192.168.137.12` 子网掩码为 `255.255.255.0` 并将电脑的可用网络设置共享到鲁班猫上

```bash
# 1. 安装ADB工具用于连板
apt install adb
# 2. 开启ADB服务并在后台运行
adbd &
```

然后可以在window终端运行如下命令测试网络是否连通
```bash
ping 192.168.137.12
```

3. 配置WSL2的网络并重启连接
在win系统的用户根目录下创建 `.wslconfig` 文本文件并写入如下内容，将WSL2配置为镜像网络模式

```
[wsl2]
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true
```

然后关机重启WSL2并使用ADB连接鲁班猫

```bash
# 1. 关闭wsl
wsl --shutdown
# 2. 手动重启wsl
wsl
# 3. 使用ADB连板 (ADB默认端口为5555)
adb connect 192.168.137.12:5555 
```

### 运行项目

1. 克隆本仓库
```bash
git clone https://github.com/Lin-Ruipeng/RKNN_learning_notes.git
```

2. 下载原始模型文件
```bash
cd RKNN_learning_notes/model
chomd +x download_model.sh
./download_model.sh
```

3. 运行源码

```bash
cd ../src
python demo.py
# 如果想保存日志内容请运行
python demo.py &> log.txt
```

## 日志内容分析

待补充
