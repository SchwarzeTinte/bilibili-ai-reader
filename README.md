# B站视频 AI 阅读器

一个完全在本机运行的 Bilibili 字幕提取、语音转写和 AI 问答工具。

## 功能

- 输入 B站链接或 BV 号并读取分P信息
- 优先提取人工字幕或 B站 AI 字幕
- 无字幕时下载音频，使用 `faster-whisper` 在本机转写
- 支持 Gemini、DeepSeek、OpenAI 和本地 Ollama
- 生成摘要、章节、知识点以及带时间戳的视频问答
- 字幕、音频和转写结果仅缓存在本机 `data` 目录

## 快速启动（Windows）

确保已经安装：

- Python 3.10 或更高版本（推荐 3.11+）
- FFmpeg，并且 `ffmpeg` 命令可以在 PowerShell 中运行

在 PowerShell 中进入本项目，然后运行：

```powershell
.\run.ps1
```

首次运行会创建 `.venv` 并安装依赖，耗时可能较长。随后浏览器会自动打开本地页面。

如果 PowerShell 阻止脚本运行，可以使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

## 使用方法

1. 在左侧选择登录状态。普通公开视频先选择“不使用”；需要登录时可读取浏览器，或指定 Netscape 格式的 `cookies.txt`。
2. 输入 B站视频链接或 BV 号，点击“读取视频信息”。
3. 有字幕时选择字幕并点击“提取所选字幕”。
4. 没有字幕时展开 Whisper 区域，点击“下载音频并开始转写”。
5. 在左侧选择 AI 服务并填写 API Key，或使用已经启动的本地 Ollama。
6. 生成摘要或在“视频问答”标签中提问。

API Key 只存在于当前 Streamlit 会话内，不会写入磁盘。

选择 Gemini、DeepSeek 或 OpenAI 时，程序会把用于总结或问答的字幕文本发送给对应服务。要让字幕分析也完全留在本机，请选择 Ollama。

Windows 可能会锁定 Chrome/Edge 的 Cookie 数据库。如果“读取浏览器”报无法复制 Cookie 数据库，请完全退出该浏览器后重试，或者在本机导出 Netscape 格式的 `cookies.txt`，再在应用中选择该文件路径。不要把 Cookie 文件发送给其他人。

## Ollama 本地模式

先安装并启动 Ollama，然后拉取一个模型，例如：

```powershell
ollama pull qwen3:8b
```

在应用左侧选择 `Ollama`，默认地址为 `http://localhost:11434/v1`。

## 数据与注意事项

- Whisper 模型首次运行会从 Hugging Face 下载到本机缓存。
- `small` 模型适合大多数普通电脑；显存或内存不足时选择 `base` 或 `tiny`。
- 浏览器 Cookie 由 yt-dlp 在本机读取。本程序不会保存 Cookie，也不要将 Cookie 提交或分享给他人。
- 网站接口可能变化，遇到解析问题时先升级 yt-dlp：

```powershell
.\.venv\Scripts\python.exe -m pip install -U yt-dlp
```

- 请只处理你有权访问和使用的内容，不要绕过会员、付费或其他访问限制。
