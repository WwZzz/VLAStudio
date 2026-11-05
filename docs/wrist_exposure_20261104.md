# 三路相机曝光检查（2026-11-04）

通过 `scripts/bi_alicia_d_vr_tele.sh --exposure-sweep-ms ...` 测试，使用与遥操作相同的
OpenCVCamera 子进程、get_data、共享内存写入和共享内存读取路径。测试仅实例化三路
相机，未连接机械臂或 Quest。640×480、MJPG、30fps，海康参数未改变。
重复读取同一时间戳不计数；另检查图像 CRC 是否相邻完全重复。

两腕同时设为手动曝光，其他原始控制不变：

| 曝光设置 | 海康 | 左腕 | 右腕 | 每档时长 |
|---|---:|---:|---:|---:|
| 10/15/20/25/30/31/31.2 ms | 约30fps | 约30fps | 约30fps | 8秒 |
| 31.5/31.8/32/32.5/32.9/33/40/50 ms | 约30fps | 约25fps | 约25fps | 8秒 |
| 66/100 ms | 约30fps | 约20fps | 约20fps | 8秒 |
| 30 ms | 1802张 | 1800张 | 1800张 | 60秒 |

设定曝光以 V4L2 100us 为单位并读回确认，但读回值不是对真实积分时间的测量。
100ms 设置仍输出20fps，说明不能把设置值直接当作有效积分时间。
这轮测试可确认30ms能保持三路30fps，掉帧率边界在31.2–31.5ms之间。
表现更支持相机自身曝光/帧周期限制，未做单相机对照，不能声称已完全排除总线因素。
测试没有硬件触发或曝光同步，不是三路硬同步证明。

30ms时两腕灰度均值仍仅约3.4/2.7（范围0–255）。增益200时约9.7/8.0；
增益255、Gamma120时约16.8/12.5，两组短测仍各自30fps。
图像中腕部主要为桌下、支架和阴影，与海康所拍亮桌面的光照不同。
不能仅据此认定硬件故障；可检查相机朝向/遮挡、增益、Gamma和照明。
原始左/右增益105/108，Gamma120/85；两相机均手动10ms。

复现（均自动恢复原始控制值）：
```bash
cd /home/jetson/Project/VLAStudio
bash scripts/bi_alicia_d_vr_tele.sh --exposure-sweep-ms 10 20 30 31.2 31.5 40 --measure-seconds 8
bash scripts/bi_alicia_d_vr_tele.sh --exposure-sweep-ms 30 --measure-seconds 60
bash scripts/bi_alicia_d_vr_tele.sh --exposure-sweep-ms 30 --wrist-gain 200 --wrist-gamma 200 --measure-seconds 8
```

原始结果在仓库 `logs/wrist_exposure_*_20261104/` 和 `logs/wrist_gain*_20261104/`，
含控制参数快照、每帧时间戳、逐档统计和带标签的相机对照图。

## 自动曝光补充测试

`bash scripts/bi_alicia_d_vr_tele.sh --auto-exposure-test --measure-seconds 30 --exposure-output logs/wrist_auto_exposure_20261104`

两路腕部设置 Auto Exposure=3，预热10秒再统计30秒。海康901张（30.03fps），
左腕693张（23.10fps），右腕691张（23.03fps）。腕部主要在20/25fps间变化，
不能称为稳定30fps。灰度均值左/右约108/112，画面比手动30ms明显明亮。
曝光绝对值读回仍为100（10ms），显然不能作为自动模式真实积分时间的测量。
测试后已读回确认恢复手动10ms、增益105/108、Gamma120/85。

## 自动模式与 Windows 表现差异排查

用户后续确认此前自动曝光30Hz发生在 Windows 或其他机器上。当前Jetson对照：

- 三路一起自动曝光，50Hz防频闪：左腕25fps，右腕20fps，海康30fps。
- 左腕单独自动曝光：25fps，与三路时相同，未显示海康争用导致其降帧。
- 三路自动曝光，60Hz防频闪或关闭防频闪：两腕均约20fps，海康30fps。
- 三路自动曝光并将增益提高到255：左25、右20fps；画面已过亮，未恢复30fps。

设备在当前Linux标准控制接口仅提供Manual(1)和Aperture Priority(3)，无
Exposure Auto Priority或自动曝光上限控制。不能仅凭请求fps=30或曝光绝对值读回30ms
认定自动模式真实30fps或实际积分30ms。当前证据支持自动曝光/设备配置链路差异，
但尚未拿到Windows当时采集格式、实际帧率证据和驱动参数，无法确认跨平台差异的具体根因。
不能据此判定硬件损坏，也不能归因于三路USB带宽不足。
本轮结束已恢复用户要求的手动30ms、增益105/108、Gamma120/85、防频闪50Hz。
结果位于 `logs/auto_diagnosis_20261104/`。

## 当前开灯环境下可保持30Hz的提亮组合

- 自动曝光+Backlight Compensation最低36：两腕约25fps；未恢复30。
- 左腕单独YUYV自动曝光：仍25fps；不能归因于MJPG格式。
- 手动30ms、增益255、Gamma200：三路15秒各450张（30fps），腕部灰度均值54/59，
  图像中可辨认夹爪和桌面，较原参数13/16明显提亮；有可见噪声。
- 手动30ms、增益200、Gamma200：三路仍各30fps，灰度均值39/43，亮度较低。

本轮均为临时参数测试，结束恢复原手动30ms、增益105/108、Gamma120/85。
结果/对照图位于 `logs/auto_parameters_20261104/`。
