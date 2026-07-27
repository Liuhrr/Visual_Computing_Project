# POSE OR DIE · 姿势生存

这是一个基于摄像头人体姿态识别的简易生存游戏。游戏会在纯黑背景中显示一个随机白色人形动作；玩家必须在倒计时结束前模仿动作并保持约半秒，否则本局立即结束。

## 已实现

- 7 种随机动作，连续回合不会重复同一个动作
- 5.5 秒起步的倒计时，回合越高速度越快，最低 3 秒
- YOLOv8 Pose 实时识别 17 个 COCO 人体关键点
- 摄像头画面常驻显示，并叠加绿色骨架
- 摄像头预览固定为 480×360，窗口变化时不缩放、不裁切
- 以固定 640×480 采集并持续请求摄像头最广视角
- 提供“关闭自动取景”按钮，处理 Windows Studio Effects 人物跟踪
- 自动兼容镜像动作，不要求玩家判断左右方向
- 匹配度达到 70% 并保持约 0.55 秒即过关
- 超时显示 `YOU DIED`，支持一键或空格键重试
- 黑白极简 UI、当前回合、分数、倒计时、匹配度与状态提示

## 文件结构

```text
Silhouette_Survival_Game/
├─ silhouette_game.py       # 游戏入口与 UI
├─ models/
│  └─ yolov8n-pose.pt       # 项目提供的姿态模型
├─ utils/
│  ├─ pose_utils.py         # 姿态提取、平滑、镜像、绘制
│  └─ scoring.py            # 归一化姿态评分
├─ tests/
└─ requirements.txt
```

## 安装与运行

首次下载后，先创建隔离环境并安装依赖：

```powershell
cd "C:\Users\heh\Desktop\NUS project\Silhouette_Survival_Game"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

安装完成后，可以双击 `start_game.bat`，或在 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe silhouette_game.py
```

建议使用 Python 3.11 或更高版本。首次启动可能需要等待姿态模型加载。Windows 弹出摄像头权限提示时请选择允许。

### 摄像头仍然跟随人物放大

这通常不是游戏 UI 在缩放，而是 Windows Studio Effects 或摄像头厂商驱动的“自动取景/人物跟踪”。点击游戏摄像头区域上方的“关闭自动取景”，选择正在使用的摄像头，然后关闭：

- 自动取景 / Automatic framing
- 人物跟踪 / Auto framing（厂商名称可能不同）

关闭后重新启动游戏。游戏也会在摄像头支持标准缩放控制时，持续请求最小缩放和固定广角。

## 操作

- 点击“开始游戏”或按空格键开始
- 站远一些，确保肩膀、手腕、髋部、膝盖和脚踝进入画面
- 在倒计时结束前模仿左侧白色人形
- 按 `Esc` 退出

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

姿态识别会在实际运行时加载模型；单元测试不需要打开摄像头。
