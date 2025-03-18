from fish_audio_sdk import WebSocketSession, TTSRequest, ReferenceAudio
from fish_audio_sdk import Session, TTSRequest, ReferenceAudio
from typing import Annotated, AsyncGenerator, Literal
import httpx
import ormsgpack
from pydantic import AfterValidator, BaseModel, conint
from dotenv import load_dotenv, find_dotenv
import warnings
import requests
modelId='5a897df8817c441cbd8f5f5d81816404&taskId=2025e593c1124f29a5606e61220f0d02'
import os
load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")
token = os.getenv("FISH_TOKEN")

if not token:
    print("Token is missing!")
else:
    print(f"Loaded token: {token[:10]}... (truncated)")

class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "mp3"
    mp3_bitrate: Literal[64, 128, 192] = 128
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # Balance mode will reduce latency to 300ms, but may decrease stability
    latency: Literal["normal", "balanced"] = "normal"


prompt="""Hi buddy! hows your day so far? """
voice_ref_audio="""/media/codemonkeyxl/TBofCode/MainCodingFolder/new_coding/expert_augmenting_system/voice_chat_rag_web/dj-tools-female-vocals-cj-30265.mp3"""
voice_ref_text="""A digital cyberpunk art splash in the style of bo-cyborgsplash, a vibrant and mystical landscape depicted in a series of vertical banners, each featuring a different color palette, the first banner on the left shows a glowing tree with intricate patterns, the second banner features a swirling orange and red design, the third banner features an abstract landscape with a tree in the center, """

request = ServeTTSRequest(
    text=prompt,
    references=[
        ServeReferenceAudio(
            audio=open(voice_ref_audio, "rb").read(),
            text=voice_ref_text,
        )
    ],
)

with (
    httpx.Client() as client,
    open("hello.mp3", "wb") as f,
):
    with client.stream(
        "POST",
        "https://api.fish.audio/v1/tts",
        content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        headers={
            "authorization": f"Bearer {token}",
            "content-type": "application/msgpack",
        },
        timeout=None,
    ) as response:
        for chunk in response.iter_bytes():
            f.write(chunk)





import pyaudio
import numpy as np
from time import time

CHUNK = 1024  # Number of frames per buffer during audio capture
FORMAT = pyaudio.paInt16  # Format of the audio stream
CHANNELS = 1  # Number of audio channels (1 for monaural audio)
RATE = 16000  # Sample rate of the audio stream (16000 samples/second)
RECORD_SECONDS = 0.5  # Duration of audio capture in seconds

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

def record_audio_stream():
    frames = []
    print("Recording audio ...")
    start_time = time()
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    end_time = time()
    print("Recording time:", end_time - start_time, "seconds")

    # Combine all recorded frames into a single numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    return audio_data

# Example usage
audio_data = record_audio_stream()