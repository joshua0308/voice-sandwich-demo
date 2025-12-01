"""
Microphone audio capture using PyAudio.

Captures audio from the default microphone and streams it as PCM bytes
suitable for AssemblyAI's streaming STT API.
"""

import asyncio
from typing import AsyncIterator

import pyaudio


async def capture_microphone_stream(
    sample_rate: int = 16000,
    channels: int = 1,
    chunk_size: int = 1600,  # ~100ms at 16kHz
    format: int = pyaudio.paInt16,
    duration_seconds: float | None = None
) -> AsyncIterator[bytes]:
    """
    Capture audio from the default microphone.

    Args:
        sample_rate: Audio sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1 for mono)
        chunk_size: Number of frames per buffer (default: 1600 = 100ms at 16kHz)
        format: Audio format (default: paInt16 for 16-bit)
        duration_seconds: Optional duration to capture (None = indefinite)

    Yields:
        bytes: PCM audio data chunks

    Example:
        ```python
        async for audio_chunk in capture_microphone_stream(duration_seconds=10):
            # Process audio_chunk
            pass
        ```
    """
    p = pyaudio.PyAudio()

    try:
        # Open stream
        stream = p.open(
            format=format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )

        print(f"Recording from microphone (rate={sample_rate}Hz, channels={channels})...")
        print("Press Ctrl+C to stop")

        total_chunks = None
        if duration_seconds:
            total_chunks = int(sample_rate / chunk_size * duration_seconds)

        chunk_count = 0
        try:
            while True:
                # Check if we've reached duration limit
                if total_chunks and chunk_count >= total_chunks:
                    break

                # Read audio data (blocking call)
                # Run in executor to avoid blocking the event loop
                audio_data = await asyncio.get_event_loop().run_in_executor(
                    None, stream.read, chunk_size
                )

                yield audio_data
                chunk_count += 1

        except KeyboardInterrupt:
            print("\nStopping recording...")

    finally:
        # Clean up
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("Microphone closed")


async def capture_microphone_until_silence(
    sample_rate: int = 16000,
    channels: int = 1,
    chunk_size: int = 1600,
    silence_threshold: int = 500,
    silence_chunks: int = 30  # ~3 seconds at 100ms chunks
) -> AsyncIterator[bytes]:
    """
    Capture audio from microphone until silence is detected.

    Args:
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels
        chunk_size: Frames per buffer
        silence_threshold: RMS threshold below which is considered silence
        silence_chunks: Number of consecutive silent chunks before stopping

    Yields:
        bytes: PCM audio data chunks
    """
    p = pyaudio.PyAudio()

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )

        print("Recording... (will stop after silence)")

        silent_chunk_count = 0

        try:
            while True:
                audio_data = await asyncio.get_event_loop().run_in_executor(
                    None, stream.read, chunk_size
                )

                # Simple RMS calculation for silence detection
                # Convert bytes to integers for RMS calculation
                samples = [int.from_bytes(audio_data[i:i+2], 'little', signed=True)
                          for i in range(0, len(audio_data), 2)]
                rms = (sum(s*s for s in samples) / len(samples)) ** 0.5

                if rms < silence_threshold:
                    silent_chunk_count += 1
                    if silent_chunk_count >= silence_chunks:
                        print("Silence detected, stopping...")
                        break
                else:
                    silent_chunk_count = 0

                yield audio_data

        except KeyboardInterrupt:
            print("\nStopping recording...")

    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("Microphone closed")
