"""
AssemblyAI Real-Time Streaming STT Transform

Python implementation that mirrors the TypeScript AssemblyAISTTTransform.
Connects to AssemblyAI's v3 WebSocket API for streaming speech-to-text.

Input: PCM 16-bit audio buffer (bytes)
Output: Transcribed text string (final/formatted transcripts only)
"""

import asyncio
import json
import os
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

import websockets
from websockets.client import WebSocketClientProtocol


class AssemblyAISTTTransform:
    """
    AssemblyAI Real-Time Streaming STT Transform (v3 API)

    Provides async streaming transcription of PCM audio data.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        format_turns: bool = True
    ):
        """
        Initialize AssemblyAI STT transform.

        Args:
            api_key: AssemblyAI API key (defaults to ASSEMBLYAI_API_KEY env var)
            sample_rate: Audio sample rate in Hz (default: 16000)
            format_turns: Whether to format turns (default: True)
        """
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("AssemblyAI API key is required")

        self.sample_rate = sample_rate
        self.format_turns = format_turns
        self.ws: Optional[WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None
        self._connection_ready = asyncio.Event()

    async def connect(self) -> None:
        """Establish WebSocket connection to AssemblyAI."""
        if self.ws and self.ws.close_code is None:
            return

        params = urlencode({
            "sample_rate": self.sample_rate,
            "format_turns": str(self.format_turns).lower(),
        })

        url = f"wss://streaming.assemblyai.com/v3/ws?{params}"
        print("AssemblyAI: Connecting to v3 streaming API...")

        self.ws = await websockets.connect(
            url,
            additional_headers={"Authorization": self.api_key}
        )

        print("AssemblyAI: WebSocket connected")

    async def _receive_messages(self) -> AsyncIterator[str]:
        """
        Receive and process messages from AssemblyAI WebSocket.

        Yields:
            Final/formatted transcript text
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected")

        try:
            async for raw_message in self.ws:
                try:
                    message = json.loads(raw_message)
                    message_type = message.get("type")

                    if message_type == "Begin":
                        self.session_id = message.get("id")
                        expires_at = message.get("expires_at")
                        print(
                            f"AssemblyAI: Session started ({self.session_id}), "
                            f"expires at {expires_at}"
                        )
                        self._connection_ready.set()

                    elif message_type == "Turn":
                        transcript = message.get("transcript", "")
                        turn_is_formatted = message.get("turn_is_formatted", False)

                        if turn_is_formatted:
                            # Final/formatted transcript - yield to pipeline
                            if transcript and transcript.strip():
                                print(f'AssemblyAI [final]: "{transcript}"')
                                yield transcript
                        else:
                            # Partial transcript - log for debugging
                            if transcript:
                                print(f'AssemblyAI [partial]: "{transcript}"')

                    elif message_type == "Termination":
                        audio_duration = message.get("audio_duration_seconds")
                        session_duration = message.get("session_duration_seconds")
                        print(
                            f"AssemblyAI: Session terminated "
                            f"(audio: {audio_duration}s, session: {session_duration}s)"
                        )
                        break

                    else:
                        # Handle errors or unknown message types
                        if "error" in message:
                            print(f"AssemblyAI error: {message['error']}")

                except json.JSONDecodeError as e:
                    print(f"AssemblyAI: Error parsing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            print("AssemblyAI: WebSocket connection closed")
        finally:
            await self.close()

    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Send PCM audio chunk to AssemblyAI.

        Args:
            audio_chunk: Raw PCM audio bytes (16-bit, mono)
        """
        if not self.ws or self.ws.close_code is not None:
            await self.connect()
            # Wait for Begin message
            await asyncio.wait_for(self._connection_ready.wait(), timeout=10.0)

        if self.ws and self.ws.close_code is None:
            # v3 API: Send raw PCM audio bytes directly (not base64)
            await self.ws.send(audio_chunk)
        else:
            print("AssemblyAI: WebSocket not open, dropping audio chunk")

    async def terminate(self) -> None:
        """Send termination message to AssemblyAI."""
        if self.ws and self.ws.close_code is None:
            print("AssemblyAI: Sending terminate message...")
            await self.ws.send(json.dumps({"type": "Terminate"}))
            # Wait briefly for termination response
            await asyncio.sleep(0.5)

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self.ws and self.ws.close_code is None:
            await self.ws.close()
        self.ws = None
        self.session_id = None
        self._connection_ready.clear()

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[str]:
        """
        Transcribe a stream of audio chunks.

        Args:
            audio_stream: Async iterator of PCM audio bytes

        Yields:
            Final transcribed text strings
        """
        try:
            # Connect to AssemblyAI
            await self.connect()

            # Start receiving messages in background
            receive_task = asyncio.create_task(
                self._collect_transcripts(self._receive_messages())
            )

            # Send audio chunks
            async for audio_chunk in audio_stream:
                await self.send_audio(audio_chunk)

            # Signal end of audio
            await self.terminate()

            # Wait for final transcripts with timeout
            try:
                transcripts = await asyncio.wait_for(receive_task, timeout=5.0)
                for transcript in transcripts:
                    yield transcript
            except asyncio.TimeoutError:
                print("AssemblyAI: Timeout waiting for final transcripts")

        finally:
            await self.close()

    @staticmethod
    async def _collect_transcripts(
        transcript_stream: AsyncIterator[str]
    ) -> list[str]:
        """Collect all transcripts from the stream into a list."""
        transcripts = []
        async for transcript in transcript_stream:
            transcripts.append(transcript)
        return transcripts


async def transcribe_audio_stream(
    audio_stream: AsyncIterator[bytes],
    api_key: Optional[str] = None,
    sample_rate: int = 16000
) -> AsyncIterator[str]:
    """
    Helper function to transcribe an audio stream using AssemblyAI.

    This is the primary function you should use to integrate with the voice pipeline.

    Args:
        audio_stream: Async iterator of PCM audio bytes (16-bit, mono)
        api_key: AssemblyAI API key (defaults to ASSEMBLYAI_API_KEY env var)
        sample_rate: Audio sample rate in Hz (default: 16000)

    Yields:
        Transcribed text strings (final transcripts only)

    Example:
        ```python
        async def process_audio():
            async for transcript in transcribe_audio_stream(audio_stream):
                print(f"Transcribed: {transcript}")
        ```
    """
    transform = AssemblyAISTTTransform(
        api_key=api_key,
        sample_rate=sample_rate
    )

    async for transcript in transform.transcribe_stream(audio_stream):
        yield transcript
