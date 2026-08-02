# Bilibili AI Reader

[English](README.md) | **中文** | [Deutsch](README.de.md)

一个本地优先的 Streamlit 应用：通过字幕、语音转写以及可选的视频画面分析读取
B站视频，并生成详细视频笔记和基于视频内容的问答。

## 主要功能

- 支持 B站视频链接、BV 号和多分P视频。
- 自动按照“已有字幕 → 本地 Whisper 转写 → 视频画面分析”的顺序读取。
- 同时检查文字密度和时间线覆盖率，不会把少量零散字幕误判为完整内容。
- 对无声或以画面为主的视频进行自适应抽帧，长视频最多抽取 180 个代表性画面。
- 支持 Gemini、OpenAI、DeepSeek、Anthropic Claude、Ollama，以及 LM Studio、
  LocalAI、llama.cpp、vLLM 等 OpenAI 兼容服务。
- 自动读取 Ollama 已安装模型，并在接口支持时检测模型上下文和视觉能力。
- 生成详细的时间线视频笔记和带时间戳的问答。
- 模型因输出长度上限停止时自动续写，同时检测重复内容，避免无限循环。
- 视频检测、下载、转写、画面分析、总结和问答都在后台任务中运行。
- 刷新页面、Streamlit Rerun、打开设置、切换历史或新建对话不会中断任务。
- 显示任务进度、运行时间、预计耗时，以及本地模型并发过载风险。
- 视频问答保存在对应视频对话中；修改旧问题会创建新分支，旧版本仍可查看。
- 提供类似 ChatGPT 的历史侧边栏、归档、批量管理和 15 天可恢复回收站。
- 设置保存在本机，下次启动自动恢复。
- 关闭最后一个应用网页后，本地服务器会自动退出。

## 环境要求

- Windows 10/11、macOS 或常见 Linux 发行版
- Python 3.10 或更高版本，推荐 Python 3.11+
- FFmpeg
- 首次安装依赖、访问 B站、调用云端 API 和首次下载 Whisper 模型时需要联网

Windows 安装 FFmpeg：

```powershell
winget install --id Gyan.FFmpeg
```

安装后重新打开终端并检查：

```powershell
python --version
ffmpeg -version
```

## 安装与启动

```powershell
git clone https://github.com/SchwarzeTinte/bilibili-ai-reader.git
cd bilibili-ai-reader
```

### Windows

双击 `run.bat`，或者运行：

```powershell
.\run.bat
```

启动器会创建 `.venv`、安装或更新依赖、检查 FFmpeg、复用已运行的项目实例，
然后打开：

```text
http://localhost:8501
```

只检查环境、不启动应用：

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -CheckOnly
```

立即关闭应用及其后台子进程，可以双击 `stop.bat`，或者运行：

```powershell
.\stop.bat
```

关闭最后一个连接到本应用的网页后，服务器会在约 6 秒缓冲时间后自动退出。
普通刷新、Streamlit Rerun 或仍有其他应用标签页连接时不会退出。

### macOS 与 Linux

先通过系统包管理器安装 Python 和 FFmpeg，然后运行：

```bash
bash run.sh
```

## 基本使用流程

1. 输入 B站视频链接或 BV 号；多分P视频需要选择分P。
2. 点击智能读取。程序先读取字幕，文字不足时尝试 Whisper，最后才补充画面分析。
3. 只有需要强制指定读取方式时才展开高级读取选项。
4. 打开“设置”，选择 AI 服务、模型、上下文预算、B站访问方式和 Whisper 参数。
5. 长任务开始前建议点击“测试 AI 连接”。
6. 生成详细视频笔记，或者针对当前视频提问。

## 自动读取逻辑

标准完整度设置约要求每分钟 30 个有效文字单位。对于超过 3 分钟的视频，有效文字
还应覆盖至少 35% 的时间线。如果任一条件不满足，程序会继续尝试音频或画面分析，
而不是忽略视频的大部分内容。

完整度敏感度分为三档：

- **节省费用**：减少视觉模型调用。
- **标准（推荐）**：平衡成本和时间线覆盖。
- **严格完整**：更重视覆盖率，更容易触发画面分析。

画面抽样数量会随视频时长调整：15 分钟视频约抽取 60 帧，长视频最多 180 帧。
抽帧仍可能漏掉一闪而过的信息；云端视觉模型还可能按图片输入计费。

当程序因为缺少有效文字而使用画面分析时，页面会显示准确率提示。小型或量化本地
视觉模型可能不如大型云端模型可靠；云端准确率则取决于实际 API 模型。人物姓名、
画面文字、数字和关键事件应当与原视频核对。

## 支持的 AI 服务

| 设置选项 | 服务 | 需要准备 |
| --- | --- | --- |
| Gemini | Google Gemini API | API Key 和可用模型 ID |
| OpenAI | OpenAI API | API Key 和可用模型 ID |
| DeepSeek | DeepSeek API | API Key 和可用模型 ID |
| Anthropic | Claude API | API Key 和可用模型 ID |
| Ollama | 本机或局域网 Ollama | 已启动的 Ollama 和至少一个已安装模型 |
| OpenAI 兼容（自定义） | LM Studio、LocalAI、llama.cpp、vLLM 或兼容云端接口 | `/v1` 地址、模型 ID，以及服务要求的 Key |

程序连接推理接口，不会直接加载单独的 `.gguf` 或 `.safetensors` 文件。这些文件
需要先通过 Ollama、LM Studio、llama.cpp、vLLM 或其他推理服务加载。

### Ollama

安装并启动 [Ollama](https://ollama.com/)，然后安装模型，例如：

```powershell
ollama pull qwen3:4b
```

默认接口地址是 `http://localhost:11434/v1`。程序读取当前 Ollama 实例的模型列表，
因此其他用户会看到其自己电脑上安装的模型，而不是开发者电脑上的模型。

上下文预算是程序允许使用的上限，并不表示每次请求都会强制分配全部窗口。程序会
尽可能使用足够且较小的 Ollama 上下文。超大上下文会显著增加内存和显存占用，
可能让本地推理看起来无响应。普通电脑建议从 8,192～32,768 tokens 开始。

### 其他本地或兼容服务

选择“OpenAI 兼容（自定义）”，然后：

1. 在相应软件中加载模型并启动 API 服务。
2. 填写 `/v1` 地址，例如 LM Studio 常用 `http://localhost:1234/v1`，llama.cpp
   常用 `http://localhost:8080/v1`。
3. 只有本地服务允许无认证访问时才可以不填 Key。
4. 选择检测到的模型，或填写服务端显示的准确模型 ID。
5. 上下文预算不能超过服务端实际加载模型时设置的上下文。

不同兼容服务对 `temperature`、输出 token 参数、system 消息和思考控制的支持不完全
一致。程序会尝试常见参数回退，但接口仍需要实现本项目使用的 Chat Completions 协议。

## B站访问身份

绝大多数公开视频应选择“不使用登录信息”。只有 B站要求已授权会话时，才使用浏览器
Cookie 或本地 Netscape 格式 `cookies.txt`。该功能不能绕过会员、付费、地区、私密
或平台访问限制。

Cookie 只由 yt-dlp 用于访问 B站，不会发送给 AI 服务，也不会由本应用保存。
请勿分享 Cookie 文件。

## Whisper

Whisper 将语音转换为文字。模型越大通常越准确，但速度更慢、占用更多内存。
普通电脑建议使用 `small + CPU`。选择 GPU 后如果 CUDA 运行环境不完整，程序会自动
回退到 CPU。Whisper 模型第一次使用时会下载到本机 Hugging Face 缓存。

## 后台任务与关闭行为

任务会记录服务、模型、进度、运行时间和最终状态。任务开始后会锁定当时的模型；
如果要为该对话切换模型，需要先终止其活动任务。不同对话可以使用不同模型，但同时
运行多个本地模型会竞争内存、显存和算力。

每个打开的应用标签页都会保持一条只连接 `127.0.0.1` 的轻量 WebSocket。最后一条
连接关闭且在缓冲时间内没有重新连接时，Streamlit 进程和后台工作线程会一起退出。
此连接不会传输视频内容、提示词或历史数据。`stop.bat`仍是立即关闭的备用方式。

Ollama 是独立程序，`stop.bat`不会关闭 Ollama。需要释放 Ollama 模型显存时运行：

```powershell
ollama ps
ollama stop 模型名称
```

## 历史、归档与回收站

总结和问答保存在本机 `data` 目录。归档只会隐藏记录，不会删除。删除对话时，如果
相关媒体文件不再被其他有效历史共享，程序会将记录与本地文件移动到 `data/.trash`。

已删除内容可以在 15 天内恢复，每条记录有独立倒计时。用户二次确认后也可以立即
永久删除；过期备份会在之后的应用启动或刷新时清理。

## 隐私与本地数据

- `data`、`.venv`、API Key、Cookie、下载媒体、模型文件和个人历史不会提交到 Git。
- 云端服务会收到相应请求使用的字幕、问题和提示词；画面分析还会发送抽取的截图。
- 主动保存 API Key 时，它会写入该电脑上被 Git 忽略的 `data/settings.json`。
- ChatGPT、Gemini、Claude 等网页订阅通常不包含相应 API 的账户、额度或计费。
- 请只处理自己有权访问和使用的内容。

## 常见问题

### `cublas64_12.dll is not found`

选择 CPU 模式或重新启动应用。程序会在常见 CUDA 运行环境错误后自动改用 CPU。

### 总结生成到一半停止

只有服务明确返回“达到输出长度上限”时，程序才会自动续写。当模型开始重复或没有
产生新内容时会停止，防止无限循环。云端模型的每次续写都是新的 API 请求。

### 没有模型列表或上下文上限

并非所有兼容服务都通过 `/models` 暴露完整元数据。请手动填写服务端准确模型 ID 和
官方上下文大小，然后测试连接。网页中的上下文数值不能扩大服务端加载时设置的窗口。

### `Could not copy Chrome cookie database`

浏览器正在锁定 Cookie 数据库。完全退出浏览器，或者导出本地 Netscape 格式
`cookies.txt` 后在设置中选择该文件。

### 下载中断

下载器会自动重试、使用小分块并回退到较低码率音频。如果 B站接口发生变化，可更新：

```powershell
.\.venv\Scripts\python.exe -m pip install -U yt-dlp
```

### 8501 端口被占用

先运行 `stop.bat`。Windows 启动器也会检测当前项目已运行的实例，并直接重新打开，
而不是故意启动重复服务器。

## 运行测试

Windows：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS 与 Linux：

```bash
.venv/bin/python -m unittest discover -s tests -v
```
