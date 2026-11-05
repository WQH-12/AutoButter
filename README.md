# AutoButter

基于 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 的安卓平台小游戏资源自动获取项目。

当前处于骨架阶段：只有环境自检任务，还没有针对具体游戏的流水线。

## 目录

```tree
assets/
├── interface.json          # 项目接口：任务入口、控制器、选项
└── resource/
    ├── pipeline/           # 任务流水线 JSON
    ├── image/              # 模板匹配素材（720p 画布裁切）
    └── model/ocr/          # OCR 模型，已被 .gitignore 忽略，需本地自行下载
agent/                      # Python 自定义识别/动作（当前未启用）
deps/tools/                 # 从 MaaFramework 同步的 JSON Schema
docs/                       # 模板自带的开发说明
```

## 环境准备

1. **ADB**：统一使用 QtScrcpy 自带的那份，路径 `D:\QtScrcpy-win-x64-v4.2.1\adb.exe`。
   不要再装第二份 platform-tools，两个 adb server 会互相抢占 5037 端口。
2. **OCR 模型**：下载 [ppocr_v6-small.zip](https://download.maafw.xyz/MaaCommonAssets/OCR/ppocr_v6/ppocr_v6-small.zip)
   解压到 `assets/resource/model/ocr/`，确保有 `det.onnx`、`rec.onnx`、`keys.txt`。
3. **Node.js 22+**：用于跑 `maa-tools` 做静态检查。
4. **MFAAvalonia**：运行和调试用的 GUI，需 .NET 10 Desktop Runtime。

## 常用命令

```bash
# 安装依赖
pnpm install

# 校验 interface.json 与 pipeline
node node_modules/@nekosu/maa-tools/bin/maa-tools check
```

模板文档写的是 `npx @nekosu/maa-tools check`，本机没有全局 `npx` 时用上面这条。

## 当前任务

| 任务 | 节点 | 作用 |
|---|---|---|
| 环境自检：截图存盘 | `SelfCheck` | 截图存到 `log_dir/screencap/selfcheck.png`，用于确认不是黑屏 |
| 环境自检：点击屏幕中心 | `TapCenter` | 在 720x1280 画布中心点击，验证输入通道 |

## 开发顺序

1. 手机连上后先跑「环境自检：截图存盘」，确认截出来的图不是全黑（全黑 = FLAG_SECURE）。
2. 用 `adb exec-out screencap -p` 抓无损原图，缩放到 720 短边后再裁模板。
3. 按游戏实际界面写 pipeline，每加一个节点就用调试工具单节点验证。
4. 需要跨节点状态（计数、超时回退）时再启用 `agent/`。

## 素材来源

模板素材**必须**来自无损原图（`adb exec-out screencap -p` 的 PNG），缩放到 720 短边后再裁剪。
QtScrcpy 的投屏是 H.264 有损视频流，不能用来裁模板。

## 许可

MIT
