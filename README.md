# AutoButter

基于 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 的安卓小游戏资源自动获取工具。
Windows 端用 MFAAvalonia 做界面，业务逻辑在 Python Agent 里。

手机连上后自动完成：进秘境 → 按优先级选资源格 → 派鸡妖采集 → 探尋 → 换下一个秘境，
循环到没有可采的资源为止。

> 当前版本 **v0.1.0**，主任务链路已实机跑通，仍在按实测日志持续调优。
> 每次改动的背景、边界与验证记录都在 [HANDOVER.md](HANDOVER.md)。

## 功能

| 能力 | 说明 |
|---|---|
| 主任务 | 进秘境 → 选格 → 派遣 → 探尋 → 别人秘境，一条流水线跑完 |
| 优先级选格 | 7 种资源各自 0~7 分，只采分数大于 0 的格 |
| 鸡妖派遣 | 读当前数量 → 连点加号 → 回读复核 → 补差额，点击被吃不会误记配额 |
| 弹窗清理 | 「恭喜獲得」「提示」等弹窗统一交给 `popup_guard` |
| 列表循环 | 翻页 / 刷新 / 冷却 / 重进各有独立闸门，划不动会自动回主界面重来 |
| 异常恢复 | 界面漂移或网络异常时 `RecoverNet` 尝试回冒险主页面，多轮失败才 `AbortTask` |

## 快速开始（直接用）

1. 拿到发布包并解压到**英文路径**（例如 `D:\AutoButter`）。
   仓库不存二进制，自己构建即可：`python tools\build_release.py --clean --zip`，
   产物是 `release/AutoButter-0.1.0.zip` 和同内容的 `release/AutoButter/` 目录。
2. 双击 `一键安装运行环境.bat`，只装缺的：.NET 10 运行时、VC++ 2015-2022 x64、
   Python 3.10+、`MaaFw==5.8.0`、`opencv-python-headless`、adb。
   缺系统组件时会弹 UAC；国内网络可加 `-PreferMirror` 直接走镜像源。
3. 双击 `启动AutoButter.bat`（自检 + 启动），或直接运行 `MFAAvalonia.exe`。
4. 「设置 → 设备」选中手机，点截图确认画面正常（全黑 = 应用禁止截屏）。
5. 任务队列只勾「主任务」，**手机停在冒险主页面**（或秘境列表 / 自家秘境面板）再点开始。

`config/` 和 `logs/` 首次运行时自动生成，不随包发布。

## 任务与选项

| 任务 | 作用 |
|---|---|
| 主任务（进秘境→选格→派遣→探尋→别人秘境） | 完整链路，日常用这个 |
| 进入淫乱秘境 | 从主页面进自家秘境 |
| 领取恭喜獲得弹窗 | 清「恭喜獲得」弹窗 |
| 选格并点击（调试用） | 单节点：按优先级选资源格 |
| 探寻列表选行（调试用） | 单节点：选他人秘境行 |
| 派遣并掠奪（调试用） | 单节点：加鸡妖 + 掠奪 |

**物品收集优先级**（`item_priority`）：云英、卷轴、玉璧、蜂蜜、饷银、枸杞、酒米各 0~7 分，
0 = 不采。默认只采酒米（7）。

**派遣模式**（`dispatch_mode`）：

- `完整队伍优先`（默认）：凑齐目标数才派，凑不齐的资源先延后；鸡妖总数只有 1~2 时目标数自动下调。
- `有几只派几只`：不让鸡妖闲置，有多少派多少，可能少于目标数。

## 目录结构

```tree
interface.json            # Maa 项目入口：任务、控制器、选项、agent 启动参数
resource/
├── pipeline/main.json    # 流水线，63 个节点
├── image/                # 模板素材（720 短边画布裁切）
└── model/ocr/            # OCR 模型，不入库，需自行下载（见下）
agent/                    # 自定义识别 / 动作（Maa Agent 子进程）
assets/
├── MaaCommonAssets/      # 上游 OCR 素材（git submodule）
└── image_stale/          # 已停用模板，仅存档
tools/                    # 校验 / 同步 / 构建脚本
tools/setup/              # 发布包运行环境检测与安装
docs/zh_cn/               # Maa 模板自带的中文开发文档
deps/tools/               # 从 MaaFramework 同步的 JSON Schema
```

工作区根目录就是 Maa 项目根：`interface.json` + `resource/` + `agent/` 同级，
与 `install/`、`release/AutoButter/` 的部署布局完全一致。

`install/`（本机调试运行时）、`release/`（构建产物）、`config/`、`logs/`、
`screenshots/` 都不入库。

## 开发

### 环境准备

1. `git clone --recurse-submodules`（`assets/MaaCommonAssets` 是 submodule）。
2. OCR 模型不入库，下载
   [ppocr_v6-small.zip](https://download.maafw.xyz/MaaCommonAssets/OCR/ppocr_v6/ppocr_v6-small.zip)
   解压到 `resource/model/ocr/`，确保有 `det.onnx`、`rec.onnx`、`keys.txt`。
3. Python 3.10+（命令名 `python`）与 `python -m pip install "MaaFw==5.8.0"`。
   **MaaFw 必须与 `install/runtimes/win-x64/native` 里的 MaaFramework 同版本**，
   否则 GUI 报 `Protocol version mismatch`。
4. Node.js 22+，用于跑 `maa-tools` 静态检查。
5. adb：本机统一用 QtScrcpy 自带的那份（`D:\QtScrcpy-win-x64-v4.2.1\adb.exe`）。
   别再装第二份 platform-tools，两个 adb server 会抢 5037 端口。

### 常用命令

```powershell
pnpm install                                          # 装 maa-tools
node node_modules/@nekosu/maa-tools/bin/maa-tools check   # 校验 interface / pipeline

python tools\check_prereqs.py                         # 工作区自检（模板/注册/interface/pipeline）
python tools\check_prereqs.py --install               # install 目录自检
powershell -File tools\sync.ps1                       # 同步 resource/agent 到 install
python tools\build_release.py --clean --zip           # 构建发布包
```

### 校验清单

| 命令 | 查什么 |
|---|---|
| `python tools\check_pipeline_refs.py` | 节点引用 / `on_error` 自引用死循环 / 节点名拼错 |
| `python tools\audit_timeouts.py` | 候选全失败又没写 timeout 的节点（会白等 20s） |
| `python tools\verify_pipeline_offline.py dispatch` | 离线剧本：1/3 加到 3/3 再掠奪 |
| `python tools\test_dispatch_policy.py` | 派遣配额与补点策略 |
| `python tools\test_popup_guard.py` | 弹窗识别与清理 |
| `python tools\test_wait_loop.py` | 等待 / 循环闸门 |
| `python tools\test_refresh_gate.py` | 刷新闸门 |
| `python tools\check_maa_version.py` | pip MaaFw 与内置 MaaFramework 是否同版本 |

### 改动流程

1. 手机连上后先 `adb exec-out screencap -p` 抓一张，确认不是全黑。
2. 模板只用无损原图，缩放到 720 短边后再裁，存到 `resource/image/`。
3. 改完 `resource/` 或 `agent/` 先跑 `tools\sync.ps1` 同步到 `install/`，再启动 MFAAvalonia。
4. 需要跨节点状态（计数、超时回退）时在 `agent/` 加自定义识别/动作，
   并登记到 `agent/registry.py` —— 漏登记会出现「GUI 能跑、离线剧本跑不了」。

## 实现结构

| 文件 | 职责 |
|---|---|
| `resource/pipeline/main.json` | 63 个流水线节点 |
| `agent/registry.py` | 16 个自定义识别 + 22 个动作的注册表 |
| `agent/loop_ctl.py` | 循环闸门、派遣计数、复位、恢复等状态机 |
| `agent/chicken_reco.py` / `cell_reco.py` / `row_reco.py` | 鸡妖 / 资源格 / 列表行识别 |
| `agent/popup_guard.py` | 弹窗识别与点击屏蔽 |
| `agent/priority.py` | 物品优先级打分 |
| `agent/layout.py` | 分辨率与坐标换算 |

## 素材与截图

模板素材**必须**来自无损原图（`adb exec-out screencap -p` 的 PNG），缩放到 720 短边后再裁剪。
QtScrcpy 的投屏是 H.264 有损视频流，不能用来裁模板。

## 许可

MIT，见 [LICENSE](LICENSE)。
