# B站视频 AI 阅读器

一个在本机运行的 Bilibili 字幕提取、语音转写和 AI 问答工具。

## 功能

- 输入 B站链接或 BV 号并选择分P
- 优先读取人工字幕或 B站 AI 字幕
- 无字幕时下载音频，使用 `faster-whisper` 本地转写
- B站 CDN 不稳定时自动分块续传并回退到低码率音频
- 默认使用兼容性更好的 CPU `int8`；选择 GPU 后若 CUDA 不完整，也会自动回退 CPU
- 支持 Gemini、DeepSeek、OpenAI 和本地 Ollama
- 生成摘要、章节、知识点和带时间戳的视频问答
- 音频、字幕和转写结果仅保存在本机 `data` 目录

## 环境要求

- Windows 10/11、macOS 或常见 Linux 发行版
- Python 3.10 或更高版本，推荐 Python 3.11+
- FFmpeg
- 首次安装依赖及下载 Whisper 模型时需要联网

Windows 安装 FFmpeg：

```powershell
winget install --id Gyan.FFmpeg
```

安装后需要重新打开终端，确保以下命令可以运行：

```powershell
python --version
ffmpeg -version
```

## 从 GitHub 安装

```powershell
git clone https://github.com/SchwarzeTinte/bilibili-ai-reader.git
cd bilibili-ai-reader
```

### Windows（一键启动）

双击项目中的 `run.bat`，或者在 PowerShell 中运行：

```powershell
.\run.bat
```

脚本会自动创建 `.venv`、安装或更新依赖、检查 FFmpeg，然后打开：

```text
http://localhost:8501
```

只检查环境、不启动页面：

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -CheckOnly
```

### macOS / Linux

先用系统包管理器安装 Python 和 FFmpeg，然后运行：

```bash
bash run.sh
```

## 使用方法

1. 输入 B站链接或 BV 号，点击“读取视频信息”。
2. 有字幕时选择字幕并提取。
3. 没有字幕时点击“下载音频并开始转写”。
4. 在左侧选择 AI 服务。使用云端模型需要自己的 API Key；使用 Ollama 不需要云端 Key。
5. 生成摘要，或针对视频内容提问。

普通公开视频先将“B站登录状态”设为“不使用”。只有视频确实要求登录时，才读取浏览器 Cookie 或指定本机 `cookies.txt`。

## 完全本地运行 AI

安装并启动 [Ollama](https://ollama.com/)，例如拉取：

```powershell
ollama pull qwen3:8b
```

在应用左侧选择 `Ollama`。默认接口地址为 `http://localhost:11434/v1`。

使用 Gemini、DeepSeek 或 OpenAI 时，用于总结或问答的字幕文本会发送给对应服务。API Key 仅保存在当前 Streamlit 会话，不写入磁盘。

## 首次运行说明

- 安装 Python 依赖可能需要几分钟。
- Whisper 模型首次使用时会下载到 Hugging Face 本机缓存。
- `small` 模型适合大多数普通电脑；内存不足时选择 `base` 或 `tiny`。
- 默认使用 CPU，不需要 CUDA；15分钟视频可能需要数分钟。只有完整安装 NVIDIA CUDA 环境时才建议选择“自动检测 GPU”。

## 常见问题

### `cublas64_12.dll is not found`

程序会自动改用 CPU。如果页面仍显示旧错误，停止程序后重新运行 `run.bat`。

### `Could not copy Chrome cookie database`

Windows 正在锁定浏览器 Cookie 数据库。完全退出所选浏览器，或导出 Netscape 格式的 `cookies.txt` 后指定其本机路径。不要把 Cookie 文件分享给别人。

### 下载中途断开

程序会自动使用小分块续传并增加重试次数；高码率音频持续失败时会回退到较低码率。

### 端口 8501 已被占用

先关闭旧的程序窗口，或在任务管理器结束旧的 Streamlit/Python 进程，再运行 `run.bat`。

### 更新 yt-dlp

```powershell
.\.venv\Scripts\python.exe -m pip install -U yt-dlp
```

## 数据与合规

- 本地缓存位于 `data`，该目录不会提交到 Git。
- Cookie 只由 yt-dlp 用于向 B站发起请求，不会发送给 AI 服务，也不会由本程序保存。
- 请只处理你有权访问和使用的内容，不要绕过会员、付费或其他访问限制。
