# Audio Transcription Toolkit

Captures system audio via PulseAudio, transcribes it in real time using faster-whisper, and optionally summarizes the transcription using a locally-running Ollama LLM. Everything runs locally — no cloud APIs required.

The primary use case is generating structured study notes from lecture audio, written as Emacs org-mode files. The Ollama prompts are tuned for AWS Solutions Architect Associate (SAA) exam content by default and can be adjusted for other subjects.

---

## Requirements

**Python packages:**

```
numpy
faster-whisper
requests
```

**System:**

- Linux with PulseAudio or PipeWire (PulseAudio compatibility layer)
- `parec` and `pactl` (PulseAudio utilities, typically in the `pulseaudio-utils` package)
- [Ollama](https://ollama.com/) running locally on port 11434 (required only for `saa_notes.py`)
- The `llama3` model pulled in Ollama (default; see Configuration)

---

## Installation

1. Install system dependencies:

   ```bash
   # Debian/Ubuntu
   sudo apt install pulseaudio-utils

   # Arch Linux
   sudo pacman -S libpulse
   ```

2. Install Python dependencies:

   ```bash
   pip install numpy faster-whisper requests
   ```

3. Install and start Ollama, then pull the default model:

   ```bash
   # Install Ollama from https://ollama.com/download
   ollama pull llama3
   ```

---

## Scripts

### `live_transcribe.py`

Minimal live transcription of system audio to stdout. Useful for quickly verifying that audio capture and Whisper are working.

- Loads faster-whisper (`base` model, CPU, int8)
- Spawns `parec` to capture audio from a PulseAudio monitor source
- Reads audio in 5-second chunks (s16le, 16kHz, mono)
- Skips silent chunks
- Prints transcribed text to stdout

**Before running**, edit the `MONITOR_SOURCE` constant at the top of the file to match your system's monitor source. Use `pactl list sources short` to find it.

### `saa_notes.py`

Full pipeline: audio capture -> Whisper transcription -> periodic Ollama summarization -> Emacs org-mode file output.

Pipeline stages:

1. Audio capture via `parec` (s16le, 16kHz, mono)
2. Transcription via faster-whisper (`small.en` model, CPU, int8) in 10-second chunks
3. Buffered transcription text sent to Ollama every ~120 seconds for live note extraction
4. On exit (Ctrl+C or silence timeout), a final comprehensive summary is generated via Ollama
5. A consolidation pass deduplicates and cleans the full org file
6. Optional org-roam index linking when `--lesson` and `--title` are provided

---

## Usage

### Discover available audio sources

```bash
python saa_notes.py --list-sources
```

This lists all PulseAudio sources and marks monitor sources with `<- monitor`. Use the monitor source for your output device (speakers or headphones) to capture system audio.

### Live transcription to stdout

```bash
# Edit MONITOR_SOURCE in the file first, then:
python live_transcribe.py
```

### Study notes pipeline (auto-detect monitor source)

```bash
python saa_notes.py
```

### Study notes with a specific source and options

```bash
python saa_notes.py --source alsa_output.pci-0000_00_1f.3.analog-stereo.monitor \
                    --model small.en \
                    --interval 120 \
                    --timeout 300
```

### Study notes with org-roam integration

```bash
python saa_notes.py --lesson "IAM, Accounts and AWS Organizations" \
                    --title "When to use IAM Roles"
```

### Full options reference

```
--source <name>      PulseAudio monitor source (default: auto-detect)
--list-sources       List available audio sources and exit
--model <name>       Whisper model size (default: small.en)
--interval <secs>    Seconds between live summaries (default: 120)
--no-raw             Delete the raw transcription .txt file after finishing
--timeout <secs>     Auto-stop after N seconds of silence (default: disabled)
--lesson <name>      Course section name for org-roam index (requires --title)
--title <name>       Title for the org-roam note (requires --lesson)
```

---

## Output

**Default mode** (no `--title`):

- `~/org/saa/saa_YYYYMMDD_HHMM.org` — structured org-mode notes
- `~/org/saa/saa_YYYYMMDD_HHMM_raw.txt` — raw transcription (deleted if `--no-raw`)

**Org-roam mode** (`--title` and `--lesson` provided):

- `~/Notes/YYYYMMDDHHMMSS-slug.org` — org-roam note with PROPERTIES drawer and ID
- `~/org/saa/saa_YYYYMMDD_HHMM_raw.txt` — raw transcription

If a note with the same title already exists in `~/Notes/`, the session appends to it rather than creating a new file.

---

## How It Works

### Audio capture

`parec` streams raw 16-bit little-endian audio at 16kHz mono from a PulseAudio monitor source (a loopback of a playback device). The data is read in fixed-size chunks and converted to float32 for Whisper.

### Silence detection

Each chunk's mean absolute amplitude is compared against a threshold (0.005). Chunks below the threshold are skipped. With `--timeout`, the session auto-stops if no speech is detected for the specified number of seconds.

### Transcription

faster-whisper runs on CPU with int8 quantization. The `small.en` model balances accuracy and speed for English audio. The `base` model is faster but less accurate (used in `live_transcribe.py`).

### Summarization

`saa_notes.py` buffers transcribed text and sends it to Ollama at regular intervals. Two prompt modes are used:

- **Live summary** — extracts key bullet points from a recent chunk of transcription
- **Final summary** — produces fully structured org-mode notes from the complete session transcription; if the text exceeds ~3000 words, it is split into chunks

After the final summary is written, a **consolidation pass** sends all accumulated notes back to Ollama to merge duplicates and clean up incremental section headers.

### Crash safety

Raw transcription text is flushed to disk line-by-line as it is produced, so a crash or power loss does not discard the session transcript.

---

## Configuration

Key constants at the top of each script:

| Constant | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `small.en` | faster-whisper model size |
| `OLLAMA_MODEL` | `llama3` | Ollama model for summarization |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `CHUNK_SECONDS` | `10` | Audio chunk size for transcription |
| `SUMMARY_INTERVAL` | `120` | Seconds between live summaries |
| `SILENCE_THRESHOLD` | `0.005` | Mean amplitude below which a chunk is silent |
| `MAX_SUMMARY_WORDS` | `3000` | Word limit per Ollama request |
| `ORG_DIR` | `~/org/saa/` | Output directory for default mode |
| `NOTES_DIR` | `~/Notes/` | Output directory for org-roam mode |

To use a different Ollama model, change `OLLAMA_MODEL`. To adapt the summarization prompts for a different subject, edit the prompt strings in the `summarize_with_ollama()` and `consolidate_notes()` functions in `saa_notes.py`.
