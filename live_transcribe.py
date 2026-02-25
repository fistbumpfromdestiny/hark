#!/usr/bin/env python3
"""Live transcription of system audio to stdout."""

import subprocess
import sys
import numpy as np
from faster_whisper import WhisperModel

MONITOR_SOURCE = "alsa_output.usb-SteelSeries_Arctis_Nova_Pro_Wireless-00.analog-stereo.monitor"  # paste from step 2
MODEL_SIZE = "base"  # small for better accuracy, base for speed
CHUNK_SECONDS = 5

def main():
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print(f"Model loaded. Listening on {MONITOR_SOURCE}...\n", file=sys.stderr)

    process = subprocess.Popen(
        ["parec", f"--device={MONITOR_SOURCE}",
        "--format=s16le", "--rate=16000", "--channels=1"],
        stdout=subprocess.PIPE
    )

    samples_per_chunk = 16000 * CHUNK_SECONDS

    try:
        while True:
            data = process.stdout.read(samples_per_chunk * 2)
            if not data:
                break
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # Skip silence
            if np.abs(audio).mean() < 0.005:
                continue

            segments, _ = model.transcribe(audio, language="en")
            for seg in segments:
                print(seg.text, end=" ", flush=True)
    except KeyboardInterrupt:
        print("\n\nDone.", file=sys.stderr)
    finally:
        process.terminate()
if __name__ == "__main__":
    main()
