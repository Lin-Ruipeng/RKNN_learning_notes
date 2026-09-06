import sys
import cv2
from scipy.special import softmax
import numpy as np
from rknn.api import RKNN

# 模型\数据集\标签文件
DATA_PATH = "../data/imagenet/ILSVRC2012_img_val_samples/dataset_20.txt"
MODEL_DIR = "../model/"
MODEL_PATH = MODEL_DIR + "mobilenetv2-12.onnx"
OUT_RKNN_PATH = MODEL_DIR + "mobilenet_v2.rknn"
CLASS_LABEL_PATH = MODEL_DIR + "synset.txt"


def main():

    print("step 1. 创建RKNN实例")
    rknn = RKNN(verbose=True)  # 打开日志输出

    print("step 2. 配置模型 (归一化参数和模型文件有关)")
    rknn.config(
        mean_values=[[255 * 0.485, 255 * 0.456, 255 * 0.406]],
        std_values=[[255 * 0.229, 255 * 0.224, 255 * 0.225]],
        target_platform="rk3588",
    )

    print("step 3. 载入模型文件 (需要设置输入形状)")
    ret = rknn.load_onnx(
        model=MODEL_PATH,
        inputs=["input"],
        input_size_list=[[1, 3, 224, 224]],
    )
    if ret != 0:
        print("载入模型文件失败! 请检查模型文件!")
        sys.exit(ret)

    print("step 4. 构建模型 (可以设置是否进行INT8量化)")
    do_quant = True  # 开启 / 关闭 量化
    ret = rknn.build(
        do_quantization=do_quant,  # 是否量化
        dataset=DATA_PATH,  # 量化参考数据集
    )
    if ret != 0:
        print("模型构建失败!")
        sys.exit(ret)

    print("step 5. 导出rknn模型文件")
    ret = rknn.export_rknn(OUT_RKNN_PATH)
    if ret != 0:
        print("模型文件导出失败!")
        sys.exit(ret)

    print("读取一个输入图片")  # 读图并且改尺寸和加维度
    img = cv2.imread("../model/bell.jpg")
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, 0)

    print("step 6. 初始化运行时环境")
    # 如果不传入目标target参数就是用仿真器跑: rknn.init_runtime()
    ret = rknn.init_runtime(
        target="rk3588",
        perf_debug=True,  # 性能调试 打开
        eval_mem=True,  # 内存占用评估 打开
    )
    if ret != 0:
        print("运行时环境启动失败! 请检查ADB连接!")
        sys.exit(ret)

    print("step 6.2: 性能评估")
    rknn.eval_perf()

    print("step 6.3: 内存评估")
    rknn.eval_memory()

    print("step 7. 运行推理")
    outputs = rknn.inference(inputs=[img])

    print("step 8. 后处理")
    # 取出数字对应的标签值
    with open(CLASS_LABEL_PATH, "r") as f:
        labels = [l.rstrip() for l in f]
    # softmax计算概率
    scores = softmax(outputs[0])
    # 只打印TOP5
    print("推理结果的 Top5 :")
    scores = np.squeeze(scores)
    a = np.argsort(scores)[::-1]
    for i in a[0:5]:
        print('[%d] score=%.2f class="%s"' % (i, scores[i], labels[i]))
    print("推理结果如上")

    print("step 9. 精度评估")
    ret = rknn.accuracy_analysis(
        inputs=["../model/bell.jpg"],
        # target="rk3588", # 不知道为什么精度评估连板会失败,必须注释
    )
    if ret != 0:
        print("精度评估失败!")
        sys.exit(ret)

    print("step 10. 释放模型")
    rknn.release()


if __name__ == "__main__":
    main()
