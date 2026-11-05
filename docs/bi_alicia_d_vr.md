# Jetson 双 Alicia-D / Quest VR

分支：`dev_update`。左臂机身号 `ADFX260423AFNTM0`，右臂 `ADFX2604211G8E8N`。
每次连接都会向设备读取机身序列号；启动不归零、不打开夹爪。

```bash
cd /home/jetson/Project/VLAStudio
bash scripts/bi_alicia_d_vr_tele.sh --check  # 只读机械臂、计算 IK、三路出图测试
bash scripts/bi_alicia_d_vr_tele.sh          # 遥操作 + 三路相机共享内存
bash scripts/bi_alicia_d_vr_collect.sh -o /home/jetson/Project/VLAStudio/data/alicia_d_25hz
# 25Hz 遥操作录制，默认HDF5；按回车开始/停止，再按提示保存
```

Quest 与 Jetson 接入相同局域网，用 Quest 浏览器访问 `https://192.168.71.118:8443`。
先允许自签证书；如 WebSocket 连接报证书错误，先访问 `https://192.168.71.118:8442`
允许该端口证书后回到 8443。点击网页进入 VR。IP 改变时使用 Jetson 当前 IP。
左控制器对应左臂，右控制器对应右臂。按住各自侧握键建立独立锚点并控制相对位姿，
松开侧握键停止发送该臂的新目标。食指扳机连续控制对应夹爪；映射沿用单臂 Alicia-D。
B/Y 回到本次启动时读取的两臂姿态（3 秒插值）；不是全关节零位。
回位期间持续查询两臂新状态并发布真实反馈，避免阻塞常规循环后触发0.5秒状态过期保护。
任一臂反馈超时、目标发送失败或保护生效时停止继续推进；末端目标发完后最多等待2秒，
以实际关节误差≤1度、夹爪误差≤15/1000确认回位，未确认时会报告失败。
Ctrl+C 退出，关闭连接；不发送卸力指令，不保证机械臂断力。

在 Jetson 图形桌面的终端可加 `--visualize` 显示相机窗口；纯 SSH 不加该选项。
不使用相机时加 `--no-cameras`。`--robot`、`--teleop` 可以覆盖配置路径。

机器人配置：`src/vlastudio/configs/robot/bi_alicia_d_rel_ee.yaml`。
遥操作配置：`src/vlastudio/configs/teleop/quest3_bi_alicia_rel_ee.yaml`。
海康作为 `head_camera`；两个普通相机暂定为 `left_wrist_camera`、`right_wrist_camera`。
`--check` 将并排比较图写到 `logs/bi_alicia_camera_check.jpg`。
左右装反时只交换配置中两个腕部的 `index_or_path`；两个普通相机序列号相同，
因此使用 USB 插口路径，重新换插口后须更新路径。三路均 640×480 / MJPG / 25 fps。
相机沿用已设置的设备颜色、曝光参数，不额外做镜像。
两腕启动及断流重开时会应用配置中的 `v4l2_controls`：自动曝光、背光补偿36、
50Hz防频闪，增益105/108、Gamma120/85。头部不覆盖这些控制。
机械臂控制循环上限100Hz，Quest输入配置60Hz；录制明确设为25Hz，排除稀疏VR输入对录制帧率的限制。
三台相机独立曝光，按主机时间戳对齐，不是硬件同步。

控制格式为 `[left_xyz,rpy,grip,right_xyz,rpy,grip]`，每臂 7 维。
两套单臂 Alicia-D Piper/Pinocchio+SciPy IK 和 EMA 独立维护，alpha=0.5，
沿用 stop-hold、软制动和 IK 求解时间预算。双 IK 顺序运行，每臂预算 22ms；
100Hz 是循环上限，不承诺实际双臂 IK 达到 100Hz。VR 输入配置为 60Hz。
默认关节速度上限 80deg/s，可在配置调节。锚点读取真实关节反馈。
这份配置沿用单臂的 link6 末端参考点；左右臂基座各自坐标独立。
尚未加入双臂间的碰撞规划或标定基座间外参。

E1 仅告警，不暂停、不锁定新目标，也不产生持久故障记录。
E2 保留连续 10 个有效新状态帧才锁定新目标的策略；不同状态会清零 E2 连续计数。
持久 E2/未知故障不会被 E1 清除。故障记录沿用 `local/alicia_serial/thermal_faults`。
驱动、配置、控制脚本均在分支源码中，不依赖旧 `local/` Python 模块。
脚本指定仓库 `src` 为导入路径，使用 `/home/jetson/miniforge3/envs/vlastudio/bin/python`。
已补充依赖 scipy 1.14.1、pin 2.7.0、websockets 13.1，保持 NumPy 1.26.4。
HDF5录制另已补充 h5py 3.12.1。

本地回归测试（不接硬件）：
```bash
PYTHONPATH=src /home/jetson/miniforge3/envs/vlastudio/bin/python -m unittest discover -s tests -p test_bi_alicia_d.py -v
```

验证记录：两台序列号与反馈读取通过，左右 FK/IK 自检通过；三路同时采集短测
约 31fps。23 项路由和告警策略测试通过。完整服务启动检查已验证 HTTPS 8443、WSS 8442、
双臂观察循环与三路相机子进程能同时运行，检查时驱动仅允许查询报文，未发运动。
启动检查命令为 `bash scripts/bi_alicia_d_vr_tele.sh --smoke-seconds 15`。
尚需现场佩戴 Quest 完成真实手柄输入到双臂运动的体验验证。
此次相机对照图中两路腕部画面很暗，出图正常但尚不能据图确认物理左右。

25Hz正式配置复测：三路同时连续60秒，海康1502张(25.03fps)，两腕各1500张
(25.00fps)，相邻图像无完全重复。腕部自动曝光下灰度均值约110/96，画面正常。
结果与对照图：`logs/three_25fps_auto_final_20261104/`。

B键回位修复：事发日志 `alicia_d_sdk_20261104_224801.log` 有172条
`No fresh joint-status frame`，无E2持久故障记录。原回位阻塞3秒且没有查询状态，
造成0.5秒后新目标被拦截。新增6项不接硬件的回归测试覆盖全程回位、持续反馈、
状态超时、发送失败、实际不到位、E1兼容和E2保护；与路由/告警测试共30项通过。
修复验证没有发送实物运动指令，需重启遥操作加载新代码。

2026-11-05：瞬态E1/E2静默计数，不再在串口接收线程持有通信锁时同步打印
`E2 thermal status: 1/10`。单帧E2依然不暂停，连续10帧E2仍拦截新目标。
已验证200帧交替E2/正常不调用日志输出且不阻止目标，第10帧连续E2仍产生持久保护。
全套33项回归通过。不能仅凭告警与卡顿同时出现认定其为全部原因，仍需实机遥操作复核。

2026-11-05 跟随延迟修复：VR 发布上限30→60Hz，关节EMA alpha 0.15→0.5。
静止判定改为10帧，维持约原来的时间窗口。共用AliciaD逻辑现在要求EMA已追上
目标后才进入静止保持，并按相对保持位置的累计位移恢复，避免缓慢手势永远无法
跨过单帧阈值。速度80deg/s、IK时间预算、相机/录制25Hz、热状态保护未更改。
可达目标的无运动测试中，两次IK合计中位数约11.7ms；这不是物理端到端延迟。
