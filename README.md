# 基于 RV1126B NPU 加速的智能宿舍物资借还终端

<p align="left">
  <img src="https://img.shields.io/badge/Platform-RV1126B-blue" alt="Platform">
  <img src="https://img.shields.io/badge/NPU%20Acceleration-RKNN-orange" alt="NPU">
  <img src="https://img.shields.io/badge/Algorithms-YOLOv8s%20%7C%20RetinaFace%20%7C%20MobileFaceNet-green" alt="Algorithms">
  <img src="https://img.shields.io/badge/Framework-Flask%20%7C%20SQLite3-lightgrey" alt="Framework">
  <img src="https://img.shields.io/badge/License-MIT-brightgreen" alt="License">
</p>

## 📌 项目简介

本项目是一款面向高校宿舍楼层、班级活动室等公共场景的**智能借还管理终端**。系统以瑞芯微 **RV1126B** 高性能 AI 处理器为核心，充分利用其 2.0 TOPS 的片上 NPU 算力，实现了人脸识别身份鉴权、多工具多目标自动清点、以及业务逻辑处理的本地化高效闭环，有效解决了公共场景下工具借还登记繁琐、流转不透明等痛点。

---

## 🌟 主要特性

*   **双路异构加速**：人脸识别与物品检测双路深度学习流水线完全卸载至硬件 NPU，CPU 仅进行业务调度（算控解耦）。
*   **级联式人脸鉴权**：级联运行 **RetinaFace**（检测/五点对齐）与 **MobileFaceNet**（特征提取），支持高精度无感身份验证，设计滑动窗口消除状态抖动。
*   **全链模型编译与留存**：项目包含原始的 `ONNX` 格式浮点模型及经过量化转换后的 `RKNN` 格式端侧硬加速模型，具备完整的模型部署参考价值。
*   **有限状态机控制（FSM）**：业务层对“陌生人 → 已识别 → 借用中 → 归还/报损”借还全流程进行状态机建模，保障多线程环境下的数据强一致性。
*   **全触控交互体验**：前端深度适配一体化屏幕形态，集成虚拟键盘（Simple-Keyboard），支持借出清单的人工二次微调确认。

---

## 📂 工程目录结构

```text
SmartStorage_RV1126B/
├── main_app.py                           # 主程序 (整合 Flask 后端、人脸识别与 USB 视频流)
├── database.py                           # 数据库表结构初始化脚本
├── smart_storage.db                      # 运行时生成的 SQLite 数据库 (存储用户与借还记录)
├── README.md                             # 项目开源说明文档
├── func/                                 # 物品识别算法工具包
│   └── func_yolov8_optimize.py           # YOLOv8 算子优化与推理前/后处理函数
├── rknnModel/                            # 物品检测专用模型文件夹
│   └── best.rknn                         # NPU YOLOv8 物品识别模型 (RKNN格式)
├── rknnpool/                             # NPU 异步多线程推理池
│   └── rknnpool_ld.py                    # 硬件级 RKNN 线程池创建与并发调度管理脚本
├── models/                               # 人脸识别及模型备份文件夹
│   ├── RetinaFace.rknn                   # NPU 级联人脸检测模型 (RKNN格式)
│   ├── MobileFaceNet.rknn                # NPU 人脸特征提取模型 (RKNN格式)
│   ├── RetinaFace_mobile320.onnx         # 原始人脸检测模型 (ONNX格式)
│   ├── mobilefacenet.onnx                # 原始人脸特征提取模型 (ONNX格式)
│   └── best.rknn                         # YOLOv8 物品识别模型备份 (RKNN格式)
├── templates/                            # Flask 网页模板文件夹 (HTML)
│   ├── index.html                        # 智能仓库主页 (人脸识别与状态自锁控制面)
│   ├── register.html                     # 身份录入页面 (适配虚拟触控键盘)
│   └── dashboard.html                    # 业务控制台 (包含左右分栏与借用确认弹窗)
└── static/                               # 网页静态资源文件夹 (CSS/JS)
    ├── index.css                         # 前端全局样式表
    └── index.js                          # 虚拟键盘事件绑定与网络请求逻辑
```

---

## 🚀 快速运行指南

### 1. 开发板配网与系统时间校准
开发板每次断电重启后，在 `adb shell` 或串口调试终端依次执行配网及 RTC 芯片对齐指令：
```bash
# 配置IP并启用网卡 (确保宿主机网卡共享网络至192.168.137.1)
ip addr add 192.168.137.100/24 dev eth0
ip link set eth0 up
ip route add default via 192.168.137.1

# 配置DNS并测试网络
echo "nameserver 114.114.114.114" > /etc/resolv.conf
ping -c 3 www.baidu.com

# 强制校准北京时间并同步到硬件时钟
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
date -s "2026-07-08 21:30:00"  # 请替换为当前准确时间
hwclock -w
```

### 2. 残留进程清理
在重新推送运行前，强烈建议清理占用 `5000` 端口的残留进程：
```bash
fuser -k 5000/tcp
pkill -9 chromium
pkill -9 python3
```

### 3. 后台启动后端业务程序
```bash
cd /userdata/SmartStorage_RV1126B/
# 后台运行 Flask 应用
nohup python3 main_app.py > /userdata/app_log.txt 2>&1 &
```

### 4. 强制推送本地 Kiosk UI 显示
```bash
export DISPLAY=:0
chromium --no-sandbox --disable-gpu --disable-software-rasterizer \
         --user-data-dir=/userdata/chrome_temp --kiosk http://127.0.0.1:5000 &
```

---

## 📈 实测性能指标

| 性能/设计指标 | 设计目标 | 实测结果 | 备注与测试条件 |
| :--- | :---: | :---: | :--- |
| **人脸识别准确率** | $\ge 90.0\%$ | **$> 95.0\%$** | 基于自建亚洲人脸微调；余弦相似度阈值 $0.90$ |
| **物品识别整体精度 (mAP50)** | $\ge 95.0\%$ | **$97.8\%$** | 基于 YOLOv8s-INT8 模型，1000+ 自建数据集训练 |
| **物品识别单帧推理延迟** | $\le 50\text{ ms}$ | **$< 15\text{ ms}$** | $640 \times 640$ 输入，4 线程 RKNN 推理池并行加速 |
| **系统端到端推流延迟** | $\le 100\text{ ms}$ | **$< 80\text{ ms}$** | Web M-JPEG 双路推流，动态帧率限制在 $30\text{ FPS}$ |
| **数据一致性强保护** | 杜绝超借扣负 | **通过验证** | 事务执行 $\min(\text{识别数}, \text{当前库存})$ 自锁校验 |

---

## 📜 开源协议

本项目基于 **[MIT License](LICENSE)** 协议开源。
