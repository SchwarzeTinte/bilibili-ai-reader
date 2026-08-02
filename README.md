# Bilibili AI Reader

**English** | [中文](README.zh-CN.md) | [Deutsch](README.de.md)

A local-first Streamlit application that reads Bilibili videos through subtitles,
speech transcription, and optional visual frame analysis, then creates detailed
notes and supports grounded video Q&A.

## Highlights

- Accepts a Bilibili URL or BV ID, including multi-part videos.
- Uses an automatic fallback pipeline: existing subtitles, then local Whisper
  transcription, then visual frame analysis when usable text is still insufficient.
- Evaluates both text density and timeline coverage instead of accepting a few
  isolated subtitle lines as complete content.
- Uses adaptive frame sampling for silent or visually driven videos, with a maximum
  of 180 representative frames for long videos.
- Supports Gemini, OpenAI, DeepSeek, Anthropic Claude, Ollama, and custom
  OpenAI-compatible services such as LM Studio, LocalAI, llama.cpp, and vLLM.
- Detects installed Ollama models and, where available, model context and vision
  capabilities.
- Generates detailed timeline-oriented video notes and timestamped answers.
- Automatically continues generation when a model explicitly stops at its output
  length limit, while detecting repeated continuations to prevent infinite loops.
- Runs video inspection, downloads, transcription, visual analysis, summaries, and
  Q&A as background tasks.
- Keeps background work alive across Streamlit reruns, page refreshes, settings
  changes, history navigation, and new conversations.
- Shows task progress, elapsed time, rough duration estimates, and overload warnings
  for concurrent local-model jobs.
- Stores Q&A inside its video conversation and supports editing an earlier question
  as a new branch without destroying the previous branch.
- Provides a ChatGPT-style history sidebar with archive, multi-select management,
  and a 15-day recoverable trash system.
- Saves settings locally and restores them on the next launch.
- Automatically exits the local server after the final application tab is closed.

## Requirements

- Windows 10/11, macOS, or a common Linux distribution
- Python 3.10 or newer; Python 3.11+ is recommended
- FFmpeg
- Internet access for initial dependency installation, Bilibili access, cloud APIs,
  and first-time Whisper model downloads

Install FFmpeg on Windows with:

```powershell
winget install --id Gyan.FFmpeg
```

Open a new terminal afterward and verify:

```powershell
python --version
ffmpeg -version
```

## Installation

```powershell
git clone https://github.com/SchwarzeTinte/bilibili-ai-reader.git
cd bilibili-ai-reader
```

### Windows

Double-click `run.bat`, or run:

```powershell
.\run.bat
```

The launcher creates `.venv`, installs or updates dependencies, verifies FFmpeg,
reuses an already-running project instance, and opens:

```text
http://localhost:8501
```

To validate the environment without starting the app:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -CheckOnly
```

To stop the application and all of its background child processes immediately,
double-click `stop.bat`, or run:

```powershell
.\stop.bat
```

Closing the final browser tab connected to this application also stops the server
after an approximately six-second grace period. Refreshing the page, using Streamlit
Rerun, or keeping another application tab open does not trigger shutdown.

### macOS and Linux

Install Python and FFmpeg through your system package manager, then run:

```bash
bash run.sh
```

## Basic workflow

1. Enter a Bilibili video URL or BV ID and select a part if necessary.
2. Click the smart-read action. The app first checks existing subtitles, then tries
   Whisper when needed, and finally supplements sparse content with frame analysis.
3. Open advanced reading options only when you need to force a specific route.
4. Open **Settings** to select an AI provider, model, context budget, Bilibili access
   method, and Whisper options.
5. Use **Test AI connection** before starting a long summary or Q&A task.
6. Generate detailed video notes or ask questions about the current video.

## Reading logic

The standard completeness profile expects roughly 30 effective text units per
minute. For videos longer than three minutes, usable text should also cover at least
35% of the timeline. When either condition is not met, the app continues to audio or
visual analysis rather than silently ignoring most of the video.

The sensitivity setting provides three profiles:

- **Save cost** reduces visual-model calls.
- **Standard** balances cost and timeline coverage.
- **Strict coverage** favors completeness and is more likely to use frame analysis.

For visual fallback, sampling scales with duration. A 15-minute video uses about 60
frames, while long videos are capped at 180 frames. Sampling can still miss a brief
single-frame event, and cloud vision APIs may charge for each image input.

When visual analysis is used because effective text is unavailable, the interface
shows an accuracy notice. Small or quantized local vision models can be less reliable
than large cloud models; cloud accuracy depends on the exact API model. Always verify
names, on-screen text, numbers, and critical events against the source video.

## Supported AI services

| Setting | Service | Required configuration |
| --- | --- | --- |
| Gemini | Google Gemini API | API key and available model ID |
| OpenAI | OpenAI API | API key and available model ID |
| DeepSeek | DeepSeek API | API key and available model ID |
| Anthropic | Claude API | API key and available model ID |
| Ollama | Local or LAN Ollama server | Running Ollama and at least one installed model |
| Custom OpenAI-compatible | LM Studio, LocalAI, llama.cpp, vLLM, or compatible cloud endpoints | `/v1` endpoint, model ID, and a key when required |

The application connects to inference APIs; it does not directly load standalone
`.gguf` or `.safetensors` files. Load those files through Ollama, LM Studio,
llama.cpp, vLLM, or another inference server first.

### Ollama

Install and start [Ollama](https://ollama.com/), then install a model, for example:

```powershell
ollama pull qwen3:4b
```

Use `http://localhost:11434/v1` as the default endpoint. The app reads the model list
from that Ollama instance, so another user will see the models installed on their own
computer rather than models from the original developer's machine.

The context budget is a maximum the app may use, not a window that is always fully
allocated. Requests use the smallest sufficient Ollama context where possible.
Very large contexts still consume substantially more RAM or VRAM and may make local
inference appear unresponsive. Start around 8,192 to 32,768 tokens unless the task
and hardware clearly require more.

### Other local or compatible servers

Choose **Custom OpenAI-compatible**, then:

1. Load the model and start its API server.
2. Enter its `/v1` URL, such as `http://localhost:1234/v1` for a typical LM Studio
   setup or `http://localhost:8080/v1` for a typical llama.cpp setup.
3. Leave the key empty only when the local service permits unauthenticated access.
4. Select a detected model or enter the exact model ID shown by the service.
5. Set a context budget that does not exceed the server-side loaded context.

Compatible services differ in their support for `temperature`, output-token fields,
system messages, and thinking controls. The app retries common parameter variations,
but the endpoint must still implement the Chat Completions protocol used here.

## Bilibili access identity

Most public videos should use **No login information**. Browser cookies or a local
Netscape-format `cookies.txt` file are only needed when Bilibili requires an already
authorized session. This feature cannot bypass membership, payment, region, private,
or platform access restrictions.

Cookies are used only by yt-dlp when making requests to Bilibili. They are not sent
to the selected AI provider and are not saved by this application. Never share a
cookie file.

## Whisper

Whisper converts speech into text. Larger models are generally more accurate but
slower and require more memory. `small` on CPU is a practical default for ordinary
computers. When GPU mode is selected but the required CUDA runtime is incomplete,
the app falls back to CPU instead of failing permanently.

The first use of a Whisper model downloads it to the machine's Hugging Face cache.

## Background tasks and shutdown behavior

Each task records its provider, model, progress, elapsed time, and terminal status.
A task keeps the model with which it started; changing that conversation's model
requires terminating its active task first. Different conversations may use different
models, although concurrent local models can compete for RAM, VRAM, and compute.

The page maintains a small localhost-only WebSocket for each open application tab.
When the last socket closes and no tab reconnects during the grace period, the
Streamlit process exits along with its daemon background workers. This mechanism does
not send content over the network. `stop.bat` remains the immediate manual fallback.

Ollama is a separate application and is not terminated by `stop.bat`. To unload a
model from Ollama memory, inspect and stop it separately:

```powershell
ollama ps
ollama stop MODEL_NAME
```

## History, archive, and trash

Summaries and Q&A are stored locally under `data`. Archiving hides an item without
deleting it. Deleting a conversation moves its history and, when no active history
still shares them, related local media files into `data/.trash`.

Deleted items can be restored for 15 days. Each item has its own countdown and may
also be permanently deleted after explicit confirmation. Expired backups are cleaned
up on a later application launch or refresh.

## Privacy and local data

- `data`, `.venv`, API keys, cookies, downloaded media, model files, and personal
  history are excluded from Git.
- Cloud providers receive the subtitle text, questions, and prompts used for their
  requests. Visual fallback also sends sampled frames to the selected cloud model.
- Saving an API key writes it to the Git-ignored `data/settings.json` file on that
  computer. Do not enable this on an untrusted shared machine.
- A ChatGPT, Gemini, Claude, or other consumer subscription usually does not include
  the separate API account, quota, or billing required by the corresponding API.
- Only process content you are authorized to access and use.

## Troubleshooting

### `cublas64_12.dll is not found`

Use CPU mode or restart the application. The app automatically retries Whisper on
CPU after common CUDA runtime failures.

### The summary stops halfway

The app continues automatically only when the provider explicitly reports an output
length limit. It stops continuation when the model repeats existing content or adds
nothing new, preventing an infinite request loop. Each cloud continuation is another
billable API request.

### The model list or context limit is missing

Not every compatible service exposes model metadata through `/models`. Enter the
exact server model ID and documented context size manually, then use the connection
test. The application-side context value cannot enlarge a model loaded with a smaller
server-side context.

### `Could not copy Chrome cookie database`

The browser is locking its cookie database. Fully exit that browser, or export a
local Netscape-format `cookies.txt` and select it in Settings.

### A download was interrupted

The downloader uses retries, small fragments, and a lower-bitrate audio fallback.
If Bilibili changes its delivery interface, update yt-dlp:

```powershell
.\.venv\Scripts\python.exe -m pip install -U yt-dlp
```

### Port 8501 is already in use

Run `stop.bat` first. The Windows launcher also detects an existing instance of this
project and reopens it instead of intentionally creating a duplicate.

## Tests

Windows:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS and Linux:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
