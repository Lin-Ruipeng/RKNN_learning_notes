# 日志分析

## Step3 载入模型文件部分

### 1. base_optimize + fold_constant ✅
完全正确。`fold_constant` 就是**常量折叠（Constant Folding）**，把能在编译期算出结果的子图直接替换为常量张量。日志里移除的 `Shape → Gather → Unsqueeze → Concat` 这条链是 ONNX 导出时常见的"动态 shape 推导残留"，因为输入 shape 在 RKNN 里已经固定为 `[1,3,224,224]`，所以整条链都可以折叠掉。这属于标准的**图优化**。

### 2. Fixed the shape information ✅
对。ONNX 模型中很多中间 tensor 的 shape 是符号化的（比如用 `batch_size`、`-1` 表示），NPU 需要**完全静态的 shape** 才能分配内存和生成指令。这一步就是把所有符号 shape 解析/推断为具体数值。如果某层 shape 推不出来，这里就会报错。

### 3. correct_ops 🔍
不完全是"检查支持性"（那个在更早的阶段就做过了）。`correct_ops` 的作用是**语义修正/规范化**，比如：

+ 把某些等价但 NPU 不友好的算子写法替换为标准写法
+ 修正属性值（如 padding、axis 等）使其符合 RKNN 内部 IR 的约定
+ 处理 ONNX opset 版本差异导致的语义歧义

可以理解为：**算子已经确认支持了，但这一步确保它们的"写法"是 NPU 编译器最喜欢的形式**。如果有真正不支持的算子，早在 `load_onnx` 阶段就会报 `Unsupported operator` 错误。

### 4. fuse_ops ✅
你的理解正确。注意日志里的融合策略很有信息量：

+ `convert_global_avgpool_to_conv`：GAP → Conv，因为 NPU 上 GAP 没有独立硬件单元，转成 1x1 Conv 更高效
+ `convert_gemm_by_exmatmul → convert_exmatmul_to_conv`：FC 层（Gemm）→ MatMul → Conv，这是 RKNN 的经典做法，**把全连接层转为卷积**以便利用 NPU 的卷积加速单元
+ `bypass_two_reshape / fuse_transpose_reshape`：消除冗余的 reshape/transpose，减少数据搬运

这些融合结果直接影响最终 NPU 上的执行效率，后面看 `eval_perf` 时可以验证。

### 5. outlier value ⚠️ （重点）
这是**量化前的权重/激活值统计分析**。工具链会扫描每一层的参数分布，当某个值偏离均值超过一定阈值（通常是 `mean ± k*std`）时标记为 outlier。

你日志里的 4 个异常值：

| const | abs_mean | abs_std | outlier | 偏离倍数 |
| --- | --- | --- | --- | --- |
| 478 | 0.89 | 1.59 | -15.07 | ~9.5σ |
| 550 | 0.61 | 0.68 | 11.30 | ~16.6σ |
| 577 | 0.64 | 0.65 | -9.88 | ~15.2σ |
| 604 | 0.60 | 0.55 | -9.97 | ~18.1σ |


这些极值会**严重压缩 INT8 量化区间**（因为 INT8 只有 -128~127，量化范围要被极值撑开，导致正常值的分辨率下降）。这就是为什么日志说 _"this may affect quantization accuracy"_。

💡 **建议**：后面看 `accuracy_analysis` 结果时重点关注这几层对应的精度损失。如果 Top1 掉点明显，可以考虑：

+ 关闭量化（`do_quant=False`）做 baseline 对比
+ 使用混合精度量化（对这几层保持 FP16）
+ 增加校准图片数量（当前只有 20 张，偏少）

### 6. sparse_weight ✅
对，这是**权重稀疏化/剪枝**。如果权重矩阵中有大量接近零的值，工具链会将其置零并用稀疏存储格式压缩，减小模型体积和内存占用。这一步通常是无损或近无损的。

---

## Step4 构建模型部分
你的观察非常细致，逐条回应并补充关键信息：

### 1. GraphPreparing + Quantizating ✅
正确。`GraphPreparing` 是为量化做图准备（插入 fake-quant 节点、收集激活值分布等），`Quantizating` 是实际计算量化参数（scale/zp）的过程。

### 2. adjust_relu 的 Clip 优化 🔍（重点澄清）
**不是调整 ReLU 的分界点**。这里的本质是：

ONNX 中的 ReLU6 / HardSwish / Clip 等激活函数，在 INT8 量化后，其 clip 范围可能**不再对齐到 INT8 的量化网格上**。

举个例子：假设某层输出量化参数为 `scale=0.05, zp=-128`，那么 INT8 值 `-128` 对应浮点 `0.0`，`-1` 对应 `6.35`。如果原始 Clip 的上限是 `6.0`，它并不精确落在某个 INT8 值上。`adjust_relu` 就是把这些 Clip 的上下限**微调对齐到最近的 INT8 可表示值**，避免量化后的激活函数产生额外的截断误差。

所以日志里列出的 35 个 Clip 节点，是工具链在告诉你："这些激活函数的边界被我调整过了，以适配 INT8 量化网格"。这是**量化感知的激活函数优化**，不是重新测量临界值。

### 3. dtype changed from float32 to int8 ⚠️
**不是提醒你量化误差大**，而是提醒你：

量化后模型的**输入/输出 tensor 的数据类型从 float32 变成了 int8**。如果你在板端用 C/C++ Runtime API 部署，喂数据和取结果时必须传 `INT8` 类型的 buffer，不能再传 float32，否则数据会被错误解释。

对于你用 Python toolkit 做推理来说，toolkit 内部会自动处理这个转换，所以这条警告对你当前代码**没有实际影响**。但如果你后续要写 C++ 部署代码，就必须注意这一点。

### 4. conv_eltwise_activation_fuse = 1 ✅
完全正确。这是 RKNN 编译器的一组优化开关：

| 参数 | 含义 |
| --- | --- |
| `conv_eltwise_activation_fuse = 1` | Conv + ElementWise(Add/Mul) + Activation 三合一融合 |
| `global_fuse = 1` | 全局算子融合（跨多层的激进融合） |
| `multi-core-model-mode = 7` | RK3588 三核 NPU 并行调度模式 |
| `output_optimize = 1` | 输出层特殊优化 |
| `layout_match = 1` | 自动匹配最优内存布局（NC1HWC2） |


你在 Network Layer Information Table 里看到的 `ConvClip`、`ConvAdd` 就是这些融合的直接结果。

### 5. RKNN Pass 日志 🔍
**不是逐层量化**。量化在之前的 `Quantizating` 阶段已经完成了。这一系列 Pass 是 **NPU 编译后端优化流水线**，每个 Pass 负责一个特定的硬件适配任务：

| Pass | 作用 |
| --- | --- |
| `RKNNBindNorm` | 将归一化参数绑定到首层 Conv |
| `RKNNEliminateQATDataConvert` | 消除 QAT 模型中多余的量化/反量化节点 |
| `RKNNConvStrideFixPass` | 修复 NPU 不支持的 stride 组合 |
| `RKNNTileGroupConv` | 将大通道卷积切分为 NPU 友好的 tile |
| `RKNNBnQuant` | BN 层参数折叠进 Conv 的 weight/bias |
| `RKNNFuseOptimizerPass` | NPU 层面的算子融合 |
| `RKNNTilingPass` | 特征图分块策略（适配 SRAM 大小） |
| `RKNNLayoutMatchPass` | 确定每层的 NC1HWC2 具体排列 |
| `RKNNWeightTransposePass` | 权重重排为 NPU 矩阵乘法单元要求的格式 |
| `RKNNModelBuildPass` | 生成最终的 NPU 指令序列 |
| `RKNNSubGraphMemoryPlanPass` | 内存复用规划 |


注意其中有一条异常日志：

```plain
D RKNN: [14:12:05.670] Unkown op target: 0
```

这出现在 `RKNNCPUWeightTransposePass` 期间，说明有一个算子的目标设备标记为 0（未指定），被当作 CPU 算子处理了。**通常无害**，但如果后续精度有问题可以关注。

### 6. Network Layer Information Table ✅
正确。补充几个关键字段的解读：

+ **Cycles(DDR/NPU/Total)**：`23540/112896/112896` 表示 DDR 访问耗时 23540 cycles，NPU 计算耗时 112896 cycles，总耗时取两者最大值（因为并行）。当 DDR > NPU 时说明该层是**访存瓶颈**。
+ **RW(KB)**：该层的读写内存量。Layer 5 (`Conv_7`) 的 RW=1177KB 是全网络最大的，因为它是第一个下采样层，feature map 还很大。
+ **Target=NPU/CPU**：所有计算层都在 NPU 上，只有 Input/Output/Reshape 在 CPU 上，这是理想的。

### 7. Export to /tmp/...check.rknn 🔍
这是**中间校验文件**，不是最终导出。工具链先导出到临时目录做完整性检查，确认无误后才会在 step 5 写入你指定的 `OUT_RKNN_PATH`。

### 8. Feature Tensor Information Table 🔍
**不是算子融合详情**，而是**运行时特征图（activation）的内存分配表**。关键信息：

+ **NativeShape**: `(1,2,112,112,16)` 就是 NC1HWC2 格式，C=32 被拆成 C1=2 组 × C2=16 通道
+ **[Start End) Size**: 确实是内存地址范围和大小
+ **带 **`*****`** 的地址**（如 `0x00062000*0x00188000`）：表示这块内存与其他 tensor **共享/复用**，这是 `SubGraphMemoryPlanPass` 做内存复用的结果
+ 整个 feature map 峰值内存 ≈ 0x00188000 = **1.56MB**，与后面的 `Total Internal Memory Size: 1568KB` 吻合

### 9. Const Tensor Information Table ✅
正确。这就是**权重和 bias 的内存布局表**。补充：

+ 权重全部是 INT8，bias 全部是 INT32（这是 INT8 量化的标准做法，INT32 bias 避免累加溢出）
+ 最大的一块权重是 `classifier.1.weight` (1000×1280×1×1) = **1.22MB**，占权重总内存的 ~34%
+ 权重总大小 3610.81KB ≈ **3.53MB**，加上 feature map 1.56MB，模型运行时总内存约 **5.1MB**

### 10. ADB 连接 ✅
注意日志开头有一行：

```plain
adb: unable to connect for root: closed
```

但最终 `Connect to Device success!`，说明首次 root 连接失败后回退到了普通 adb 连接，**不影响使用**。

### 11. 内存统计 ✅
补充解读：

+ **Internal Memory 1568KB** = NPU SRAM 中的 feature map 峰值占用（不含权重缓存）
+ **Weight Memory 3610.81KB** = 所有权重+bias 的总大小
+ RK3588 NPU SRAM 总共约 2MB，你的模型只用了 1.56MB，**还有余量**，说明 tiling 策略比较保守，没有充分利用 SRAM

---

### 📌 关键发现汇总
| 项目 | 状态 | 备注 |
| --- | --- | --- |
| 量化流程 | ✅ 正常 | 35 个 Clip 已对齐量化网格 |
| 算子融合 | ✅ 充分 | ConvClip/ConvAdd 融合完整 |
| NPU 利用率 | ✅ 良好 | 仅 I/O 和 Reshape 在 CPU |
| 内存占用 | ✅ 健康 | SRAM 余量充足 |
| Unknown op target | ⚠️ 轻微 | 需关注但不阻塞 |
| 输入输出变 INT8 | ℹ️ 提醒 | Python 推理无影响，C++ 部署需注意 |


## Step6 性能评估
### 1. 频率打印 ✅
正确。这是性能评估的**环境基线记录**。注意你的 NPU 频率是 `1000000000`（1GHz），这是 RK3588 NPU 的标准频率。如果后续发现性能不达预期，可以先确认板端是否被降频了。

### 2. Debug 模式 vs 实际性能 🔍（重要澄清）
**不是"不开 debug 就更快"**，而是：

`perf_debug=True` 会在每层之间**插入硬件计数器读取和同步屏障**，导致测出来的时间是**逐层串行耗时之和**，而实际推理时 NPU 支持层间流水线并行（pipeline overlap），所以：

+ **eval_perf 报告的 Total Time (5529μs) ≥ 实际端到端推理时间**
+ 日志也明确说了 _"The performance result is just for debugging, may worse than actual performance!"_

要获取真实推理延迟，应该用 `rknn.inference()` 配合 Python 的 `time.time()` 或 C++ 的计时器来测。**eval_perf 的价值在于定位瓶颈层，而不是给出绝对性能数字。**

### 3. Network Layer Information Table 详解 🔍（核心问题）
#### 各列含义补充
| 列 | 含义 |
| --- | --- |
| `Cycles(DDR/NPU/Total)` | 该层的 DDR 访问 cycles / NPU 计算 cycles / max(DDR,NPU)（因为并行执行，总耗时取瓶颈） |
| `Time(us)` | 实测耗时（含 debug 开销），≈ Total Cycles / NPU频率 |
| `MacUsage(%)` | NPU MAC 单元利用率，三列对应三个 NPU 核心 |
| `WorkLoad(0/1/2)` | 三个 NPU 核心的负载分配比例 |
| `RW(KB)` | 该层读写 DDR 的数据量 |


#### 关于 "21211/4992/21211" 的理解 ⚠️ 需要纠正
你说 _"如果是0拷贝就可以更快"_ —— **这个理解有误**。

这里的 DDR cycles **不是数据搬运/拷贝的时间**，而是 **NPU 通过 DMA 从 DDR 读取权重和特征图的访存等待时间**。RK3588 的 NPU 架构中：

+ NPU 计算单元和 DMA 引擎是**并行工作**的
+ `Total = max(DDR, NPU)` 正是因为两者重叠执行
+ 当 DDR > NPU 时（如 Layer 25: 21211 > 4992），说明该层是**访存受限（memory-bound）**，NPU 在等数据
+ 当 NPU > DDR 时（如 Layer 34: 7488 < 22272），说明该层是**计算受限（compute-bound）**

**即使使用零拷贝（zero-copy），DDR cycles 也不会变成 0**，因为权重和中间结果仍然存储在 DDR 中，NPU 仍然需要通过总线去读。零拷贝省掉的是 **CPU****↔****NPU 之间的额外 memcpy**，而不是 NPU↔DDR 的访存。

#### MacUsage 极低的问题 ⚠️（关键发现）
你所有层的 MacUsage 都在 **0.09% ~ 18.49%** 之间，三核中只有 Core0 有负载，Core1/Core2 全是 0%。这说明：

**当前模型只使用了 RK3588 三核 NPU 中的 1 个核心！**

结合 `multi-core-model-mode = 7`（编译时设置了三核模式），但实际运行时 `WorkLoad` 全是 `100%/0%/0%`，可能原因：

1. **模型太小**：MobileNetV2 的计算量对 RK3588 来说太轻，编译器判断单核就够了
2. **perf_debug 模式下强制单核串行**：debug 模式可能禁用了多核并行以方便逐层 profiling
3. **Runtime 版本与 Toolkit 版本不完全匹配**

💡 **建议**：关闭 `perf_debug` 后用 `rknn.inference()` 实测延迟，再对比 eval_perf 的数据。如果实测延迟远小于 5529μs，说明确实是多核并行生效了；如果差不多，可能需要检查多核配置。

#### 关于 Python API 零拷贝
Python toolkit 的 `rknn.inference(inputs=[img])` 确实会做隐式拷贝。如果你追求极致性能，可以用：

```python
# 预分配 NPU 内存，避免每次推理的 malloc+memcpy
rknn.init_runtime(target="rk3588")
input_attrs = rknn.get_input_attrs()
output_attrs = rknn.get_output_attrs()
# 使用 pass_through 模式直接传 INT8 数据
outputs = rknn.inference(
    inputs=[img_int8], data_format="nhwc", inputs_pass_through=[True]
)
```

但这只是省掉了 CPU 侧的预处理拷贝，**不影响上面表格中的 DDR cycles**。

### 4. Operator Time Consuming Ranking Table ✅
正确。前一个表是**逐层视角**（Layer-level），这个是**算子类型聚合视角**（OpType-level）。关键信息：

+ ConvClip 占了 70.28%，Conv 占 15.34%，ConvAdd 占 13.08% → **98.7% 的时间都在卷积类算子上**，融合效果很好
+ CPU 时间仅 72μs（Reshape + I/O），占比 1.3%，**几乎没有 CPU 瓶颈**

### 5. Memory Profile Info Dump ✅
正确。补充解读：

| 项目 | 大小 | 说明 |
| --- | --- | --- |
| Weight Memory | 3.53 MiB | 权重+bias，与 build 阶段的 3610.81KB 一致 |
| Internal Tensor Memory | 1.53 MiB | 运行时 feature map 峰值，与 build 阶段的 1568KB 一致 |
| Other Memory | 373.19 KiB | NPU 指令、描述符表、对齐填充等开销 |
| **Total Memory** | **5.42 MiB** | 模型运行时 NPU 侧总内存占用 |
| Model Size | 3.97 MiB | .rknn 文件大小（含权重+指令+元数据） |


💡 注意 Total Memory (5.42MiB) > Model Size (3.97MiB)，差值就是运行时动态分配的 feature map 内存。部署时需要的内存预算应以 **5.42 MiB** 为准。

---

### 📌 Step 6 关键发现汇总
| 项目 | 状态 | 行动建议 |
| --- | --- | --- |
| eval_perf 时间偏高 | ℹ️ 预期行为 | 用 `inference()` 实测真实延迟 |
| MacUsage 极低 + 单核运行 | ⚠️ 需验证 | 关闭 perf_debug 后实测，确认多核是否生效 |
| 大部分层 memory-bound | ℹ️ MobileNetV2 特性 | 轻量模型正常现象，无需优化 |
| 内存占用 5.42 MiB | ✅ 健康 | RK3588 完全无压力 |
| CPU 开销 1.3% | ✅ 优秀 | 无需优化 |


## Step9 精度评估
### 1. 两个指标的含义补充
| 指标 | 含义 | 判断标准 |
| --- | --- | --- |
| **cos (余弦相似度)** | 衡量输出向量的**方向一致性**，对整体趋势敏感 | ≥0.995 优秀，≥0.990 良好，<0.980 需关注 |
| **euc (欧氏距离)** | 衡量输出向量的**绝对偏差大小**，对幅值误差敏感 | 越小越好，但需结合该层激活值的量级来看 |


💡 **关键区别**：`entire` 是从输入到当前层的**累积误差**（前面所有层的量化误差叠加），`single` 是**仅当前层**引入的误差。如果 `entire` 远差于 `single`，说明误差在逐层累积；如果两者接近，说明当前层本身就是主要误差源。

### 2. 全网络精度健康度评估
#### ✅ 整体结论：**量化质量良好**
+ 最终输出层 `[exDataConvert] output` 的 entire cos = **0.99331**，对于 INT8 量化的 MobileNetV2 来说这是**正常且可接受**的水平
+ ImageNet Top1 准确率损失通常在 1~3% 以内

#### ⚠️ 需要关注的异常层（日志中已用黄色高亮）
按严重程度排序：

| 层 | entire cos | single cos | 诊断 |
| --- | --- | --- | --- |
| **Add_452** | **0.96674** | **0.98270** | 🔴 **全网络最差**。残差相加层，两条分支的量化误差在此叠加。single cos 也低，说明该层本身量化困难 |
| **Conv_624** | 0.97062 | 0.99910 | 🟡 entire 差但 single 极好 → 纯粹的**累积误差受害者**，不是自身问题 |
| **Conv_627 + Clip_463** | 0.97429 | 0.99990 | 🟡 同上，GAP 转 Conv 层，累积误差传导至此 |
| **Conv_615** | 0.97768 | 0.99873 | 🟡 轻微累积 |
| **Add_443** | 0.98057 | **0.99093** | 🟠 第二个残差加法的瓶颈，single cos=0.991 说明该 Add 本身也有量化损失 |
| **Conv_597** | 0.98648 | **0.99501** | 🟠 single cos 开始下降，对应之前 outlier 警告中的 const 604 附近 |


### 3. 误差传播模式分析 🔍（核心洞察）
从表格中可以清晰看到 **MobileNetV2 的 Inverted Residual Block 误差传播规律**：

```plain
[Conv expand] → cos ≈ 0.999x  (扩展层，误差小)
[DWConv]      → cos ≈ 0.999x  (深度卷积，误差小)  
[Conv project]→ cos ≈ 0.999x  (投影层，误差小)
[Add residual]→ cos ↓↓↓       (⚠️ 残差连接处误差突增!)
```

**这是 MobileNetV2 INT8 量化的经典痛点**：残差加法要求两个分支的量化参数（scale/zp）对齐，但两条路径经过不同的卷积链后，激活值分布差异大，强制对齐会引入额外截断误差。

具体到你的模型：

+ **Add_339**: entire 0.99440 → 第一个 bottleneck，误差刚开始累积
+ **Add_356**: entire 0.99335 → 逐步下降
+ **Add_365**: entire 0.99230
+ **Add_400**: entire 0.99077
+ **Add_417**: entire 0.99071
+ **Add_426**: entire 0.98957
+ **Add_443**: entire 0.98057 ← 加速恶化
+ **Add_452**: entire **0.96674** ← 谷底

**从 Add_426 到 Add_452，cos 从 0.9896 骤降到 0.9667，掉了 2.3 个百分点**。这恰好对应网络后半段通道数从 576→960 的区域，也就是之前 build 阶段报 outlier 的那几层（const 577, 604）所在位置。

### 4. 与 Build 阶段 Outlier 的关联验证
| Build 阶段 Outlier | 对应精度分析位置 | 影响 |
| --- | --- | --- |
| const 478 (-15.07) | Conv_477/Clip_320 附近 | entire cos 0.99951，影响轻微 |
| const 550 (11.30) | Conv_549/Clip_388 附近 | entire cos 0.99486，可控 |
| **const 577 (-9.88)** | **Conv_576/Clip_414 → Add_417** | **entire cos 从 0.9929 降至 0.9907，开始加速** |
| **const 604 (-9.97)** | **Conv_603/Clip_440 → Add_443** | **entire cos 骤降至 0.9806，single cos=0.991** |


✅ **验证了之前的预判**：outlier 确实导致了实质性的精度损失，且主要集中在网络后半段的残差连接处。

### 5. 优化建议（按优先级排序）
| 优先级 | 措施 | 预期收益 | 成本 |
| --- | --- | --- | --- |
| 🥇 | **增加校准图片数量**：从 20 张增加到 100~200 张 | 改善后半段量化参数估计，可能提升 0.5~1% cos | 低 |
| 🥈 | **混合精度量化**：对 Add_443、Add_452 及其前驱层设为 FP16 | 直接消除瓶颈层量化误差 | 中（需手动指定层） |
| 🥉 | **关闭量化做 baseline**：`do_quant=False`<br/> 对比 Top5 | 确认量化损失的具体幅度 | 低 |
| 4 | 检查 BGR/RGB 问题（之前提到的） | 排除非量化因素的干扰 | 极低 |


### 6. 关于你注释掉的 `target="rk3588"`
你在代码中提到连板精度评估失败。当前日志显示的是**模拟器（simulator）** 的精度分析结果：

+ 模拟器精度 ≠ 板端真实精度，但通常**高度相关**
+ 如果模拟器 cos=0.993，板端一般也在 0.990~0.995 范围内
+ 连板失败常见原因：toolkit 2.3.2 与板端 runtime 版本不完全匹配、或 `accuracy_analysis` 在连板模式下内存需求更大导致 OOM

---

### 📌 最终总结
| 维度 | 评估 |
| --- | --- |
| 量化整体质量 | ✅ 良好（output cos=0.993） |
| 主要瓶颈 | ⚠️ 后半段残差加法层（Add_443, Add_452） |
| Outlier 影响 | ✅ 已验证有实质影响 |
| 部署可行性 | ✅ 可以部署，Top1 损失预计 1~3% |
| 进一步优化空间 | 有，增加校准图或混合精度可再提升 |

