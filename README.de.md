# Bilibili AI Reader

[English](README.md) | [中文](README.zh-CN.md) | **Deutsch**

Eine lokal ausgerichtete Streamlit-Anwendung, die Bilibili-Videos über Untertitel,
Spracherkennung und optionale Einzelbildanalyse ausliest. Anschließend können
ausführliche Videonotizen erstellt und inhaltlich fundierte Fragen gestellt werden.

## Hauptfunktionen

- Unterstützt Bilibili-Links, BV-IDs und mehrteilige Videos.
- Automatische Lesereihenfolge: vorhandene Untertitel, lokale Whisper-Transkription
  und anschließend Bildanalyse, falls weiterhin zu wenig verwertbarer Text vorliegt.
- Prüft sowohl Textdichte als auch zeitliche Abdeckung, damit wenige isolierte
  Untertitel nicht als vollständiger Inhalt gelten.
- Adaptive Bildauswahl für stumme oder visuell geprägte Videos; bei langen Videos
  werden höchstens 180 repräsentative Bilder verwendet.
- Unterstützt Gemini, OpenAI, DeepSeek, Anthropic Claude, Ollama sowie
  OpenAI-kompatible Dienste wie LM Studio, LocalAI, llama.cpp und vLLM.
- Erkennt installierte Ollama-Modelle und – soweit verfügbar – Kontext- und
  Bildfähigkeiten des Modells.
- Erstellt ausführliche, zeitlich strukturierte Videonotizen und Antworten mit
  Zeitstempeln.
- Setzt eine Ausgabe automatisch fort, wenn das Modell ausdrücklich wegen seiner
  Ausgabelänge abbricht, und verhindert Endlosschleifen durch Wiederholungserkennung.
- Videoanalyse, Downloads, Transkription, Bildanalyse, Zusammenfassungen und Fragen
  laufen als Hintergrundaufgaben.
- Aktualisieren der Seite, Streamlit Rerun, Einstellungen, Verlauf und neue Dialoge
  unterbrechen laufende Aufgaben nicht.
- Zeigt Fortschritt, Laufzeit, grobe Zeitschätzung und Überlastungswarnungen bei
  mehreren lokalen Modellaufgaben.
- Fragen bleiben beim jeweiligen Video. Das Bearbeiten einer älteren Frage erzeugt
  einen neuen Zweig, ohne die vorherige Version zu löschen.
- ChatGPT-ähnliche Verlaufsleiste mit Archiv, Mehrfachauswahl und einem 15 Tage lang
  wiederherstellbaren Papierkorb.
- Lokale Einstellungen werden beim nächsten Start wiederhergestellt.
- Nach dem Schließen des letzten Anwendungs-Tabs beendet sich der lokale Server.

## Voraussetzungen

- Windows 10/11, macOS oder eine gängige Linux-Distribution
- Python 3.10 oder neuer; Python 3.11+ wird empfohlen
- FFmpeg
- Internetzugang für die Erstinstallation, Bilibili, Cloud-APIs und den ersten
  Download eines Whisper-Modells

FFmpeg unter Windows installieren:

```powershell
winget install --id Gyan.FFmpeg
```

Danach ein neues Terminal öffnen und prüfen:

```powershell
python --version
ffmpeg -version
```

## Installation und Start

```powershell
git clone https://github.com/SchwarzeTinte/bilibili-ai-reader.git
cd bilibili-ai-reader
```

### Windows

`run.bat` doppelt anklicken oder ausführen:

```powershell
.\run.bat
```

Das Startskript erstellt `.venv`, installiert oder aktualisiert Abhängigkeiten,
prüft FFmpeg, verwendet eine bereits laufende Projektinstanz weiter und öffnet:

```text
http://localhost:8501
```

Nur die Umgebung prüfen, ohne die Anwendung zu starten:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -CheckOnly
```

Anwendung und Hintergrundprozesse sofort beenden:

```powershell
.\stop.bat
```

Nach dem Schließen des letzten verbundenen Anwendungs-Tabs beendet sich der Server
nach ungefähr sechs Sekunden. Ein normales Neuladen, Streamlit Rerun oder ein
weiterhin geöffneter Anwendungs-Tab löst das Beenden nicht aus.

### macOS und Linux

Python und FFmpeg über die Paketverwaltung installieren, danach:

```bash
bash run.sh
```

## Grundlegender Ablauf

1. Bilibili-Link oder BV-ID eingeben und gegebenenfalls einen Videoteil auswählen.
2. Intelligentes Auslesen starten. Zuerst werden Untertitel, danach Whisper und erst
   zuletzt ergänzende Videobilder verwendet.
3. Erweiterte Optionen nur öffnen, wenn ein bestimmter Leseweg erzwungen werden soll.
4. Unter „Einstellungen“ KI-Dienst, Modell, Kontextbudget, Bilibili-Zugriff und
   Whisper-Optionen konfigurieren.
5. Vor langen Aufgaben die KI-Verbindung testen.
6. Ausführliche Videonotizen erzeugen oder Fragen zum aktuellen Video stellen.

## Automatische Leselogik

Das Standardprofil erwartet ungefähr 30 wirksame Texteinheiten pro Minute. Bei
Videos über drei Minuten soll verwertbarer Text außerdem mindestens 35 % der
Zeitleiste abdecken. Wird eine Bedingung nicht erfüllt, versucht die Anwendung
weiterhin Audio- oder Bildanalyse, statt große Videoteile stillschweigend auszulassen.

Es gibt drei Empfindlichkeitsstufen:

- **Kosten sparen** reduziert Aufrufe visueller Modelle.
- **Standard** gleicht Kosten und zeitliche Abdeckung aus.
- **Strenge Abdeckung** priorisiert Vollständigkeit und nutzt eher Bildanalyse.

Die Bildanzahl passt sich der Videolänge an. Ein 15-minütiges Video verwendet etwa
60 Bilder; lange Videos sind auf 180 Bilder begrenzt. Sehr kurze Ereignisse können
trotzdem übersehen werden. Cloud-Modelle können Bildinputs zusätzlich berechnen.

Wenn wegen fehlenden Textes Bildanalyse eingesetzt wird, erscheint ein Hinweis zur
Genauigkeit. Kleine oder quantisierte lokale Modelle können unzuverlässiger als große
Cloud-Modelle sein. Bei APIs hängt die Genauigkeit vom tatsächlich gewählten Modell
ab. Namen, eingeblendete Texte, Zahlen und wichtige Ereignisse sollten mit dem
Originalvideo verglichen werden.

## Unterstützte KI-Dienste

| Auswahl | Dienst | Erforderlich |
| --- | --- | --- |
| Gemini | Google Gemini API | API-Schlüssel und verfügbare Modell-ID |
| OpenAI | OpenAI API | API-Schlüssel und verfügbare Modell-ID |
| DeepSeek | DeepSeek API | API-Schlüssel und verfügbare Modell-ID |
| Anthropic | Claude API | API-Schlüssel und verfügbare Modell-ID |
| Ollama | Lokaler oder im LAN erreichbarer Ollama-Server | Laufendes Ollama und mindestens ein installiertes Modell |
| Benutzerdefiniert OpenAI-kompatibel | LM Studio, LocalAI, llama.cpp, vLLM oder kompatible Cloud-Endpunkte | `/v1`-Adresse, Modell-ID und gegebenenfalls Schlüssel |

Die Anwendung verbindet sich mit Inferenzschnittstellen und lädt keine einzelnen
`.gguf`- oder `.safetensors`-Dateien direkt. Solche Dateien müssen zuerst über
Ollama, LM Studio, llama.cpp, vLLM oder einen anderen Inferenzserver geladen werden.

### Ollama

[Ollama](https://ollama.com/) installieren und starten, danach ein Modell laden:

```powershell
ollama pull qwen3:4b
```

Die Standardadresse lautet `http://localhost:11434/v1`. Die Anwendung liest die
Modelle der jeweils verbundenen Ollama-Instanz. Andere Benutzer sehen deshalb die
auf ihrem eigenen Rechner installierten Modelle.

Das Kontextbudget ist eine Obergrenze und wird nicht bei jeder Anfrage vollständig
reserviert. Die Anwendung verwendet möglichst ein kleineres, ausreichendes Fenster.
Sehr große Kontexte benötigen deutlich mehr RAM oder VRAM und können lokale Inferenz
scheinbar blockieren. Für normale Rechner sind 8.192 bis 32.768 Tokens ein sinnvoller
Ausgangspunkt.

### Andere lokale oder kompatible Dienste

„Benutzerdefiniert OpenAI-kompatibel“ auswählen, dann:

1. Modell im jeweiligen Programm laden und den API-Server starten.
2. `/v1`-Adresse eintragen, beispielsweise `http://localhost:1234/v1` für eine
   typische LM-Studio-Konfiguration oder `http://localhost:8080/v1` für llama.cpp.
3. Den Schlüssel nur leer lassen, wenn der lokale Dienst dies erlaubt.
4. Ein erkanntes Modell auswählen oder die genaue serverseitige Modell-ID eintragen.
5. Das Kontextbudget darf den beim Serverstart geladenen Kontext nicht überschreiten.

Kompatible Dienste unterscheiden sich bei `temperature`, Ausgabetoken-Parametern,
Systemnachrichten und Denksteuerung. Die Anwendung versucht übliche Varianten, der
Endpunkt muss jedoch das hier verwendete Chat-Completions-Protokoll implementieren.

## Bilibili-Zugriffsidentität

Für die meisten öffentlichen Videos sollte „Keine Anmeldeinformationen“ verwendet
werden. Browser-Cookies oder eine lokale `cookies.txt` im Netscape-Format sind nur
nötig, wenn Bilibili bereits autorisierten Zugriff verlangt. Mitgliedschafts-,
Zahlungs-, Regions-, Privat- oder Plattformbeschränkungen werden nicht umgangen.

Cookies werden ausschließlich von yt-dlp für Bilibili-Anfragen verwendet, nicht an
KI-Dienste gesendet und nicht von dieser Anwendung gespeichert. Cookie-Dateien dürfen
nicht weitergegeben werden.

## Whisper

Whisper wandelt Sprache in Text um. Größere Modelle sind meist genauer, aber langsamer
und speicherintensiver. `small + CPU` ist ein sinnvoller Standard. Ist eine gewählte
CUDA-Umgebung unvollständig, fällt die Anwendung automatisch auf die CPU zurück.
Beim ersten Einsatz wird das Whisper-Modell in den lokalen Hugging-Face-Cache geladen.

## Hintergrundaufgaben und Beenden

Aufgaben speichern Dienst, Modell, Fortschritt, Laufzeit und Endstatus. Das beim Start
gewählte Modell bleibt für die Aufgabe gesperrt. Vor einem Modellwechsel muss die
laufende Aufgabe dieses Dialogs beendet werden. Verschiedene Dialoge dürfen
verschiedene Modelle nutzen; mehrere lokale Modelle konkurrieren jedoch um RAM,
VRAM und Rechenleistung.

Jeder geöffnete Anwendungs-Tab hält eine leichte WebSocket-Verbindung ausschließlich
zu `127.0.0.1`. Wenn die letzte Verbindung geschlossen wird und innerhalb der
Schonfrist keine neue entsteht, enden Streamlit-Prozess und Hintergrundarbeiter.
Über diese Verbindung werden keine Videos, Prompts oder Verlaufsdaten übertragen.
`stop.bat` bleibt die sofortige manuelle Alternative.

Ollama ist ein eigenständiges Programm und wird durch `stop.bat` nicht beendet.
Ein Modell kann separat aus dem Speicher entladen werden:

```powershell
ollama ps
ollama stop MODELLNAME
```

## Verlauf, Archiv und Papierkorb

Zusammenfassungen und Fragen werden lokal unter `data` gespeichert. Archivieren
blendet einen Eintrag nur aus. Beim Löschen werden Verlauf und nicht mehr gemeinsam
verwendete Mediendateien nach `data/.trash` verschoben.

Gelöschte Inhalte bleiben 15 Tage wiederherstellbar und besitzen jeweils einen
eigenen Countdown. Nach zusätzlicher Bestätigung können sie sofort endgültig gelöscht
werden. Abgelaufene Sicherungen werden bei einem späteren Start oder Neuladen entfernt.

## Datenschutz und lokale Daten

- `data`, `.venv`, API-Schlüssel, Cookies, Medien, Modelle und persönlicher Verlauf
  werden nicht in Git eingecheckt.
- Cloud-Dienste erhalten die für eine Anfrage nötigen Untertitel, Fragen und Prompts;
  bei Bildanalyse zusätzlich die ausgewählten Einzelbilder.
- Gespeicherte API-Schlüssel liegen in der von Git ignorierten Datei
  `data/settings.json` auf dem jeweiligen Rechner.
- Verbraucherabonnements für ChatGPT, Gemini oder Claude enthalten normalerweise
  nicht automatisch das separate API-Konto oder API-Guthaben.
- Es dürfen nur Inhalte verarbeitet werden, für die eine entsprechende Berechtigung
  besteht.

## Fehlerbehebung

### `cublas64_12.dll is not found`

CPU-Modus verwenden oder die Anwendung neu starten. Bei typischen CUDA-Fehlern
versucht die Anwendung die Transkription automatisch auf der CPU erneut.

### Die Zusammenfassung endet zu früh

Automatisch fortgesetzt wird nur, wenn der Anbieter ausdrücklich ein Ausgabelimit
meldet. Wiederholt das Modell vorhandenen Text oder erzeugt nichts Neues, wird die
Fortsetzung beendet. Jede Cloud-Fortsetzung ist eine weitere API-Anfrage.

### Modellliste oder Kontextgrenze fehlen

Nicht jeder kompatible Dienst liefert vollständige Metadaten über `/models`. Die
genaue Modell-ID und dokumentierte Kontextgröße können manuell eingetragen und danach
getestet werden. Der Anwendungswert kann den serverseitig geladenen Kontext nicht
vergrößern.

### `Could not copy Chrome cookie database`

Der Browser sperrt seine Cookie-Datenbank. Browser vollständig beenden oder eine
lokale `cookies.txt` im Netscape-Format exportieren und in den Einstellungen wählen.

### Download wurde unterbrochen

Der Downloader verwendet Wiederholungen, kleine Fragmente und einen Audio-Fallback
mit geringerer Bitrate. Bei Änderungen an Bilibili kann yt-dlp aktualisiert werden:

```powershell
.\.venv\Scripts\python.exe -m pip install -U yt-dlp
```

### Port 8501 ist belegt

Zuerst `stop.bat` ausführen. Das Windows-Startskript erkennt außerdem eine bereits
laufende Instanz dieses Projekts und öffnet sie erneut, statt absichtlich eine zweite
zu starten.

## Tests

Windows:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS und Linux:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
