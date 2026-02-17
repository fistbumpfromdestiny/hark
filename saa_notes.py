#!/usr/bin/env python3
"""
AWS SAA Study Notes Pipeline
Captures system audio → Whisper transcription → Ollama summarization → Org file

Usage:
    python saa_notes.py                          # uses default monitor source
    python saa_notes.py --source <monitor_name>  # specify monitor source
    python saa_notes.py --list-sources            # list available sources
"""

import subprocess
import sys
import re
import signal
import argparse
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from faster_whisper import WhisperModel

# --- Config ---
WHISPER_MODEL = "small.en"
OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 10
SUMMARY_INTERVAL = 120  # summarize every 2 minutes of transcription
SILENCE_THRESHOLD = 0.005
MAX_SUMMARY_WORDS = 3000  # safe for 8k context window
ORG_DIR = Path.home() / "org" / "saa"
NOTES_DIR = Path.home() / "Notes"
INDEX_FILE = NOTES_DIR / "20260206134039-aws_saa.org"
# --------------


def list_sources():
    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True
        )
    except FileNotFoundError:
        print("Error: pactl not found. Install PulseAudio utilities.", file=sys.stderr)
        return
    if result.returncode != 0:
        print(f"Error: pactl failed (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        return
    print("Available audio sources:\n")
    for line in result.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            marker = " ← monitor" if "monitor" in parts[1] else ""
            print(f"  {parts[1]}{marker}")


def find_default_monitor():
    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True
        )
    except FileNotFoundError:
        print("Error: pactl not found. Install PulseAudio utilities.", file=sys.stderr)
        return None
    if result.returncode != 0:
        print(f"Error: pactl failed (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        return None
    for line in result.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2 and "monitor" in parts[1]:
            return parts[1]
    return None


def summarize_with_ollama(text, is_final=False):
    if is_final:
        prompt = f"""You are creating final study notes for the AWS Solutions Architect Associate (SAA) exam.

Take this raw transcription from a study session and create well-structured org-mode notes.

Rules:
- Extract ONLY exam-relevant information: AWS services, configurations, best practices, architecture patterns
- Use org-mode headings (** for topics, *** for subtopics)
- Use concise bullet points (- ) for key facts
- Mark things to memorize with TODO
- Include any mentioned AWS service limits, defaults, or gotchas
- Skip filler, repetition, off-topic talk
- Be concise but don't lose technical details

Transcription:
{text}

Respond with ONLY the org-mode content, no explanation."""
    else:
        prompt = f"""Extract key AWS SAA study points from this transcription chunk.
Be very concise. Only include exam-relevant facts, service details, and architecture concepts.
Format as org-mode bullet points (- ). Skip filler and repetition.

Transcription:
{text}

Respond with ONLY the bullet points, no explanation."""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }, timeout=120)
        try:
            return response.json().get("response", "").strip()
        except (ValueError, requests.exceptions.JSONDecodeError):
            print(f"\n[Ollama returned non-JSON: {response.text[:200]}]", file=sys.stderr)
            return None
    except Exception as e:
        print(f"\n[Ollama error: {e}]", file=sys.stderr)
        return None


class StudyNotesSession:
    def __init__(self, source, keep_raw=True, lesson=None, title=None, silence_timeout=0):
        self.source = source
        self.keep_raw = keep_raw
        self.lesson = lesson
        self.title = title
        self.silence_timeout = silence_timeout
        self.all_transcription = []
        self.current_buffer = []
        self.buffer_start_time = None
        self.org_file = None
        self.raw_file = None
        self.chunk_count = 0
        self.note_id = None
        self.appending = False

        if self.title:
            # Check for existing note with same title
            existing = self._find_existing_note()
            if existing:
                self.org_path, self.note_id = existing
                self.appending = True
                print(f"[Appending to existing note: {self.org_path.name}]", file=sys.stderr)
            else:
                slug = re.sub(r'[^a-z0-9_]', '', self.title.lower().replace(" ", "_").replace("-", "_"))
                roam_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                self.note_id = str(uuid.uuid4())
                self.org_path = NOTES_DIR / f"{roam_timestamp}-{slug}.org"
            self.raw_path = ORG_DIR / f"saa_{datetime.now().strftime('%Y%m%d_%H%M')}_raw.txt"
            try:
                ORG_DIR.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass  # only needed for raw, non-fatal
        else:
            # Default: standalone org file in ~/org/saa/
            try:
                ORG_DIR.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"Error: cannot create directory {ORG_DIR}: {e}", file=sys.stderr)
                sys.exit(1)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            self.org_path = ORG_DIR / f"saa_{timestamp}.org"
            self.raw_path = ORG_DIR / f"saa_{timestamp}_raw.txt"

    def _find_existing_note(self):
        """Find existing org-roam note with matching title. Returns (path, id) or None."""
        for f in NOTES_DIR.glob("*.org"):
            try:
                note_id = None
                with open(f) as fh:
                    for i, line in enumerate(fh):
                        if i > 10:
                            break
                        if line.startswith(":ID:"):
                            note_id = line.split(":ID:")[1].strip()
                        if line.startswith("#+title:") and line[8:].strip() == self.title:
                            return (f, note_id)
            except OSError:
                continue
        return None

    def init_org_file(self):
        mode = "a" if self.appending else "w"
        try:
            self.org_file = open(self.org_path, mode)
        except OSError as e:
            print(f"Error: cannot open {self.org_path}: {e}", file=sys.stderr)
            sys.exit(1)
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        if self.appending:
            # Append a continuation marker
            self.org_file.write(f"\n* Continued {date_str}\n\n")
        elif self.title:
            # New org-roam note
            self.org_file.write(f":PROPERTIES:\n")
            self.org_file.write(f":ID:       {self.note_id}\n")
            self.org_file.write(f":END:\n")
            self.org_file.write(f"#+title: {self.title}\n")
            self.org_file.write(f"#+filetags: :aws:saa:study:\n\n")
        else:
            self.org_file.write(f"#+TITLE: AWS SAA Study Notes\n")
            self.org_file.write(f"#+DATE: {date_str}\n")
            self.org_file.write(f"#+FILETAGS: :aws:saa:study:\n\n")
            self.org_file.write(f"* Session {date_str}\n\n")
        self.org_file.flush()

    def write_live_summary(self, text):
        if not text:
            return
        time_str = datetime.now().strftime("%H:%M")
        self.org_file.write(f"** Notes ~{time_str}\n")
        self.org_file.write(text + "\n\n")
        self.org_file.flush()
        print(f"\n[Live notes written at {time_str}]", file=sys.stderr)

    def write_final_summary(self):
        full_text = " ".join(self.all_transcription)
        if not full_text.strip():
            print("\n[No transcription to summarize]", file=sys.stderr)
            return

        words = full_text.split()
        heading = "* Final Summary (continued)" if self.appending else "* Final Summary"

        if len(words) <= MAX_SUMMARY_WORDS:
            print("\n[Generating final structured notes...]", file=sys.stderr)
            summary = summarize_with_ollama(full_text, is_final=True)
            if summary:
                self.org_file.write(f"\n{heading}\n\n")
                self.org_file.write(summary + "\n")
                self.org_file.flush()
                print(f"[Final notes saved to {self.org_path}]", file=sys.stderr)
        else:
            # Text too long, chunk it
            num_chunks = (len(words) + MAX_SUMMARY_WORDS - 1) // MAX_SUMMARY_WORDS
            print(f"\n[Text too long for single summary, splitting into {num_chunks} chunks]", file=sys.stderr)
            self.org_file.write(f"\n{heading}\n\n")
            for i in range(num_chunks):
                start_idx = i * MAX_SUMMARY_WORDS
                end_idx = min((i + 1) * MAX_SUMMARY_WORDS, len(words))
                chunk_text = " ".join(words[start_idx:end_idx])
                print(f"[🔄 Generating notes for part {i+1}/{num_chunks}...]", file=sys.stderr)
                summary = summarize_with_ollama(chunk_text, is_final=True)
                if summary:
                    self.org_file.write(f"** Part {i+1}\n\n")
                    self.org_file.write(summary + "\n\n")
                    self.org_file.flush()
            print(f"[Final notes saved to {self.org_path}]", file=sys.stderr)

    def process_buffer(self):
        if not self.current_buffer:
            return
        text = " ".join(self.current_buffer)
        self.current_buffer = []

        if len(text.split()) < 20:
            return

        summary = summarize_with_ollama(text)
        self.write_live_summary(summary)


    def consolidate_notes(self):
        """Read back the org file, send notes to Ollama to deduplicate/clean, rewrite."""
        try:
            content = self.org_path.read_text()
        except OSError:
            return

        # Split header from body — header ends at first blank line after metadata
        lines = content.split("\n")
        body_start = 0
        for i, line in enumerate(lines):
            if line.startswith("* ") or (line.startswith("** ") and i > 0):
                body_start = i
                break
        if body_start == 0:
            return

        header = "\n".join(lines[:body_start])
        body = "\n".join(lines[body_start:])

        if len(body.split()) < 30:
            return

        print("[Consolidating notes (removing repetition)...]", file=sys.stderr)

        prompt = f"""You are editing org-mode study notes for the AWS Solutions Architect Associate (SAA) exam.

These notes were generated incrementally during a lecture and contain significant repetition.
Consolidate them into a single clean set of notes.

Rules:
- Merge duplicate points — keep the most detailed/accurate version
- Preserve ALL unique technical facts, service details, limits, and gotchas
- Use org-mode headings (* for topics, ** for subtopics)
- Use concise bullet points (- ) for key facts
- Remove redundant section headers like "Notes ~HH:MM", "Final Summary", "Part N"
- Do NOT add information that isn't in the original notes
- Be concise but don't lose technical details

Notes to consolidate:
{body}

Respond with ONLY the cleaned org-mode content, no explanation."""

        try:
            response = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            }, timeout=180)
            try:
                cleaned = response.json().get("response", "").strip()
            except (ValueError, requests.exceptions.JSONDecodeError):
                print("[Consolidation failed: non-JSON response]", file=sys.stderr)
                return
        except Exception as e:
            print(f"[Consolidation failed: {e}]", file=sys.stderr)
            return

        if not cleaned or len(cleaned.split()) < 10:
            print("[Consolidation returned too little content, keeping original]", file=sys.stderr)
            return

        try:
            self.org_path.write_text(header + "\n" + cleaned + "\n")
            print("[Notes consolidated successfully]", file=sys.stderr)
        except OSError as e:
            print(f"Error: cannot write consolidated notes: {e}", file=sys.stderr)

    def update_index(self):
        if not self.lesson or not self.note_id or self.appending:
            return
        try:
            content = INDEX_FILE.read_text()
        except OSError as e:
            print(f"Error: cannot read index {INDEX_FILE}: {e}", file=sys.stderr)
            return

        entry = f"** [[id:{self.note_id}][{self.title}]]"
        lines = content.split("\n")
        insert_at = None

        for i, line in enumerate(lines):
            if line.startswith("* ") and line[2:].strip() == self.lesson:
                # Found the lesson section, find the end of it
                insert_at = i + 1
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("* "):
                        insert_at = j
                        break
                    if lines[j].strip():
                        insert_at = j + 1
                break

        if insert_at is None:
            # Lesson not found, append new section at end
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(f"* {self.lesson}")
            lines.append(entry)
        else:
            lines.insert(insert_at, entry)

        try:
            INDEX_FILE.write_text("\n".join(lines))
            print(f"[Index updated: {self.title} added to {self.lesson}]", file=sys.stderr)
        except OSError as e:
            print(f"Error: cannot update index {INDEX_FILE}: {e}", file=sys.stderr)

    def run(self):
        print(f"Loading Whisper model '{WHISPER_MODEL}'...", file=sys.stderr)
        try:
            model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        except Exception as e:
            print(f"Error: failed to load Whisper model '{WHISPER_MODEL}': {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Model loaded", file=sys.stderr)

        self.init_org_file()

        # Open raw file for incremental writes (crash safety)
        try:
            self.raw_file = open(self.raw_path, "w")
        except OSError as e:
            print(f"Error: cannot open {self.raw_path}: {e}", file=sys.stderr)
            self.org_file.close()
            sys.exit(1)

        try:
            process = subprocess.Popen(
                ["parec", f"--device={self.source}",
                 "--format=s16le", f"--rate={SAMPLE_RATE}", "--channels=1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except FileNotFoundError:
            print("Error: parec not found. Install PulseAudio utilities.", file=sys.stderr)
            self.org_file.close()
            sys.exit(1)
        if process.poll() is not None:
            print(f"Error: parec exited immediately (code {process.returncode})", file=sys.stderr)
            self.org_file.close()
            sys.exit(1)

        samples_per_chunk = SAMPLE_RATE * CHUNK_SECONDS
        last_summary_time = time.time()

        print(f"Listening on {self.source}", file=sys.stderr)
        print(f"Writing to {self.org_path}", file=sys.stderr)
        print(f"   Live summaries every ~{SUMMARY_INTERVAL}s", file=sys.stderr)
        if self.silence_timeout:
            print(f"   Auto-stop after {self.silence_timeout}s of silence", file=sys.stderr)
        print(f"   Press Ctrl+C to stop and generate final notes\n", file=sys.stderr)

        last_speech_time = time.time()

        try:
            while True:
                data = process.stdout.read(samples_per_chunk * 2)
                if not data:
                    err = process.stderr.read().decode().strip()
                    if err:
                        print(f"\n[parec error: {err}]", file=sys.stderr)
                    break

                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                if np.abs(audio).mean() < SILENCE_THRESHOLD:
                    if self.silence_timeout and self.all_transcription:
                        elapsed = time.time() - last_speech_time
                        if elapsed >= self.silence_timeout:
                            print(f"\n[{self.silence_timeout}s of silence, auto-stopping]", file=sys.stderr)
                            break
                    continue

                last_speech_time = time.time()

                segments, _ = model.transcribe(audio, language="en")
                for seg in segments:
                    text = seg.text.strip()
                    if text:
                        print(f"  {text}", file=sys.stderr)
                        self.all_transcription.append(text)
                        self.current_buffer.append(text)
                        # Write to raw file immediately (crash safety)
                        self.raw_file.write(text + "\n")
                        self.raw_file.flush()

                # Periodic live summary
                if time.time() - last_summary_time >= SUMMARY_INTERVAL and self.current_buffer:
                    self.process_buffer()
                    last_summary_time = time.time()

        except KeyboardInterrupt:
            print("\n\n[Stopping...]", file=sys.stderr)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

            # Restore default SIGINT so Ctrl+C during cleanup exits immediately
            signal.signal(signal.SIGINT, signal.SIG_DFL)

            # Process any remaining buffer
            if self.current_buffer:
                self.process_buffer()

            # Generate final structured summary
            self.write_final_summary()

            # Close raw file
            if self.raw_file:
                self.raw_file.close()

            # Delete raw file if not keeping it
            if not self.keep_raw:
                try:
                    self.raw_path.unlink()
                    print(f"[Raw transcription cleaned up]", file=sys.stderr)
                except OSError:
                    pass
            else:
                print(f"[Raw transcription saved to {self.raw_path}]", file=sys.stderr)

            self.org_file.close()

            # Consolidate notes (deduplicate across live summaries + final summary)
            self.consolidate_notes()

            # Update org-roam index if lesson/title provided
            self.update_index()

            print(f"\nNotes saved: {self.org_path}", file=sys.stderr)


def main():
    global WHISPER_MODEL, SUMMARY_INTERVAL

    parser = argparse.ArgumentParser(description="AWS SAA Study Notes from Audio")
    parser.add_argument("--source", help="PulseAudio monitor source name")
    parser.add_argument("--list-sources", action="store_true", help="List audio sources")
    parser.add_argument("--model", default=WHISPER_MODEL, help="Whisper model (default: small.en)")
    parser.add_argument("--interval", type=int, default=SUMMARY_INTERVAL,
                        help="Seconds between live summaries (default: 120)")
    parser.add_argument("--no-raw", action="store_true",
                        help="Don't keep the raw transcription .txt file")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Auto-stop after N seconds of silence (default: disabled)")
    parser.add_argument("--lesson", help="Course section in index (e.g. 'IAM, Accounts and AWS Organizations')")
    parser.add_argument("--title", help="Title for the org-roam note (e.g. 'When to use IAM Roles')")
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return

    source = args.source or find_default_monitor()
    if not source:
        print("No monitor source found. Use --list-sources to find one.", file=sys.stderr)
        sys.exit(1)

    WHISPER_MODEL = args.model
    SUMMARY_INTERVAL = args.interval

    if args.lesson and not args.title:
        print("Error: --lesson requires --title", file=sys.stderr)
        sys.exit(1)
    if args.title and not args.lesson:
        print("Error: --title requires --lesson", file=sys.stderr)
        sys.exit(1)

    session = StudyNotesSession(source, keep_raw=not args.no_raw,
                                lesson=args.lesson, title=args.title,
                                silence_timeout=args.timeout)
    session.run()


if __name__ == "__main__":
    main()
