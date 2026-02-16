# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Audio transcription tools that capture system audio via PulseAudio, transcribe with faster-whisper, and optionally summarize with Ollama. Designed for Linux with PulseAudio/PipeWire.

## Scripts

- **`live_transcribe.py`** — Minimal live transcription to stdout. Captures audio from a hardcoded PulseAudio monitor source, transcribes in chunks, prints text.
- **`saa_notes.py`** — Full pipeline: audio capture → Whisper transcription → periodic Ollama summarization → Emacs org-mode file output to `~/org/saa/`. Run with `python saa_notes.py --list-sources` to discover audio sources.

## Dependencies

Python packages: `numpy`, `faster-whisper`, `requests`
System: `parec` (PulseAudio), `pactl` (for source listing), Ollama running locally on port 11434 (for saa_notes.py)

## Running

```bash
python live_transcribe.py                        # live transcription (edit MONITOR_SOURCE in file)
python saa_notes.py                              # auto-detect monitor source
python saa_notes.py --source <name> --model small.en --interval 120
```

## Architecture Notes

- Audio is captured as raw s16le/16kHz/mono via `parec` subprocess, read in fixed-size chunks
- Silence detection skips chunks below amplitude threshold (0.005 mean absolute value)
- `saa_notes.py` uses `StudyNotesSession` class that buffers transcription text and periodically sends it to Ollama for summarization; on Ctrl+C it generates a final comprehensive summary from all accumulated text
- Ollama prompts are tuned for AWS SAA exam content extraction; adjust prompts in `summarize_with_ollama()` for other subjects
