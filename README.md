# AutoButter

基于 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 的安卓平台小游戏资源自动获取项目。

主任务链路与自定义识别已实现，实机整链验证中。

## 目录

```tree
interface.json              # 项目接口：任务入口、控制器、选项
resource/
├── pipeline/               # 任务流水线 JSON
├── image/                  # 模板匹配素材（720p 画布裁切）
└── model/ocr/              # OCR 模型，已被 .gitignore 忽略，需本地自行下载
agent/                      # Python 自定义识别/动作（Maa Agent 子进程）
assets/
├── MaaCommonAssets/        # 上游 OCR 素材（submodule）
└── image_stale/            # 已停用模板，仅存档
install/                    # 本机调试用运行时，tools\sync.ps1 同步
release/AutoButter/         # 发布包，tools\build_release.py 生成
deps/tools/                 # 从 MaaFramework 同步的 JSON Schema
docs/                       # 模板自带的开发说明
tools/                      # 校验 / 同步 / 构建脚本
tools/setup/                # 发布包运行环境检测/安装（install_env.ps1 + 两个 .bat 入口）
```

工作区根目录就是 Maa 项目根：`interface.json`、`resource/`、`agent/` 同级，
与 `install/`、`release/AutoButter/` 的部署布局一致（11-20 从 `assets/` 迁出）。

## 环境准备

1. **ADB**：统一使用 QtScrcpy 自带的那份，路径 `D:\QtScrcpy-win-x64-v4.2.1\adb.exe`。
   不要再装第二份 platform-tools，两个 adb server 会互相抢占 5037 端口。
2. **OCR 模型**：下载 [ppocr_v6-small.zip](https://download.maafw.xyz/MaaCommonAssets/OCR/ppocr_v6/ppocr_v6-small.zip)
   解压到 `resource/model/ocr/`，确保有 `det.onnx`、`rec.onnx`、`keys.txt`。
3. **Node.js 22+**：用于跑 `maa-tools` 做静态检查。
4. **MFAAvalonia**：运行和调试用的 GUI，需 .NET 10 Desktop Runtime。
5. **终端用户**：发布包根目录的 `一键安装运行环境.bat` 会检测并补齐 .NET 10 / VC++ /
   Python / `MaaFw==5.8.0` / adb，只装缺的（实现在 `tools/setup/install_env.ps1`）。

## 常用命令

```bash
# 安装依赖
pnpm install

# 校验 interface.json 与 pipeline
node node_modules/@nekosu/maa-tools/bin/maa-tools check

# 构建发布包（→ release\AutoButter）
python tools\build_release.py --clean
```

模板文档写的是 `npx @nekosu/maa-tools check`，本机没有全局 `npx` 时用上面这条。

## 当前任务

| 任务 | 节点 | 作用 |
|---|---|---|
| 主任务 | `主任务` | 进秘境 → 选格 → 派遣 → 探尋 → 别人秘境，完整链路 |
| 进入淫乱秘境 | `进入淫乱秘境` | 从主页面进自家秘境 |
| 领取恭喜獲得弹窗 | `关闭获得弹窗` | 清「恭喜獲得」弹窗 |
| 选格并点击（调试用） | `选格并点击` | 单节点：按优先级选资源格 |
| 探寻列表选行（调试用） | `选行并点击` | 单节点：选他人秘境行 |
| 派遣并掠奪（调试用） | `派遣并掠奪` | 单节点：加鸡妖 + 掠奪 |

## 开发顺序

1. 手机连上后先 `adb exec-out screencap -p` 抓一张图，确认不是全黑（全黑 = FLAG_SECURE）。
2. 裁模板只用无损原图，缩放到 720 短边后再裁，存到 `resource/image/`。
3. 改完 `resource/` 或 `agent/` 先跑 `tools\sync.ps1` 同步到 `install/`，再启动 MFAAvalonia。
4. 需要跨节点状态（计数、超时回退）时在 `agent/` 加自定义识别/动作，并登记到 `registry.py`。

## 素材来源

模板素材**必须**来自无损原图（`adb exec-out screencap -p` 的 PNG），缩放到 720 短边后再裁剪。
QtScrcpy 的投屏是 H.264 有损视频流，不能用来裁模板。

## 许可

MIT
