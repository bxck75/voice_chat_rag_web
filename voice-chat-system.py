import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
import sounddevice as sd
import numpy as np
import ormsgpack
import httpx
from pydantic import Annotated, BaseModel, conint
from typing import List, Literal, Optional
import os
from dotenv import load_dotenv, find_dotenv
import warnings
import time
import queue

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
    reference_id: str | None = None
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # Balance mode will reduce latency to 300ms, but may decrease stability
    latency: Literal["normal", "balanced"] = "normal"

class FishAudioTTS:
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the FishAudioTTS class.
        
        Args:
            token (Optional[str]): API token for Fish Audio. If None, it will try to load from environment.
        """
        warnings.filterwarnings("ignore")
        
        # Load token from environment if not provided
        if token is None:
            load_dotenv(find_dotenv())
            token = os.getenv("FISH_TOKEN")
            if not token:
                raise ValueError("Fish Audio token is missing. Please provide a token or set the FISH_TOKEN environment variable.")
            else:
                print(f"Loaded token: {token[:10]}... (truncated)")
        
        self.token = token
        
    def add_reference_audio(self, audio_path: str, text: str) -> ServeReferenceAudio:
        """
        Create a reference audio object from a file path and text.
        
        Args:
            audio_path (str): Path to the audio file
            text (str): Text corresponding to the audio
            
        Returns:
            ServeReferenceAudio: A reference audio object
        """
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        return ServeReferenceAudio(
            audio=audio_bytes,
            text=text
        )
    
    def generate_tts(self, 
                     text: str, 
                     output_path: str,
                     reference_audios: List[ServeReferenceAudio] = None,
                     reference_id: str = None,
                     chunk_length: int = 200,
                     output_format: Literal["wav", "pcm", "mp3"] = "mp3",
                     mp3_bitrate: Literal[64, 128, 192] = 128,
                     normalize: bool = True,
                     latency: Literal["normal", "balanced"] = "normal"):
        """
        Generate text-to-speech audio and save to a file.
        
        Args:
            text (str): Text to synthesize
            output_path (str): Path to save the output audio file
            reference_audios (List[ServeReferenceAudio], optional): List of reference audios for voice cloning
            reference_id (str, optional): ID of a reference voice to use
            chunk_length (int, optional): Chunk length for processing. Defaults to 200.
            output_format (Literal["wav", "pcm", "mp3"], optional): Output format. Defaults to "mp3".
            mp3_bitrate (Literal[64, 128, 192], optional): MP3 bitrate. Defaults to 128.
            normalize (bool, optional): Whether to normalize text. Defaults to True.
            latency (Literal["normal", "balanced"], optional): Latency mode. Defaults to "normal".
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Create the TTS request
        request = ServeTTSRequest(
            text=text,
            references=reference_audios if reference_audios else [],
            reference_id=reference_id,
            chunk_length=chunk_length,
            format=output_format,
            mp3_bitrate=mp3_bitrate,
            normalize=normalize,
            latency=latency
        )
        
        try:
            with httpx.Client() as client, open(output_path, "wb") as f:
                with client.stream(
                    "POST",
                    "https://api.fish.audio/v1/tts",
                    content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
                    headers={
                        "authorization": f"Bearer {self.token}",
                        "content-type": "application/msgpack",
                    },
                    timeout=None,
                ) as response:
                    if response.status_code != 200:
                        print(f"Error: API returned status code {response.status_code}")
                        return False
                        
                    for chunk in response.iter_bytes():
                        f.write(chunk)
            
            print(f"TTS generated successfully and saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating TTS: {str(e)}")
            return False

class MicrophoneTranscriber:
    def __init__(self, model_name="openai/whisper-tiny.en", sample_rate=16000):
        """
        Initialize the microphone transcriber with Whisper.
        
        Args:
            model_name (str): Name of the Whisper model to use
            sample_rate (int): Audio sample rate (should be 16000 for Whisper)
        """
        print(f"Loading Whisper model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice to capture audio data"""
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.copy())
        
    def start_recording(self, duration=5):
        """
        Start recording from the microphone.
        
        Args:
            duration (int): Recording duration in seconds
            
        Returns:
            numpy.ndarray: Recorded audio array
        """
        self.audio_queue = queue.Queue()
        self.is_recording = True
        
        print(f"Recording for {duration} seconds... Speak now!")
        
        # Start the recording stream
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self._audio_callback):
            # Record for the specified duration
            sd.sleep(int(duration * 1000))
        
        # Combine all audio data from the queue
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
            
        # Convert to a single numpy array and flatten
        if audio_data:
            audio_array = np.concatenate(audio_data).flatten()
        else:
            audio_array = np.zeros(0)
            
        self.is_recording = False
        print("Recording finished.")
        
        return audio_array
    
    def transcribe(self, audio_array=None, duration=5):
        """
        Transcribe audio from the microphone or from a provided array.
        
        Args:
            audio_array (numpy.ndarray, optional): Audio array to transcribe. If None, will record from mic.
            duration (int, optional): Recording duration in seconds if recording from mic.
            
        Returns:
            str: Transcribed text
        """
        # If no audio array is provided, record from the microphone
        if audio_array is None:
            audio_array = self.start_recording(duration)
            
        if len(audio_array) == 0:
            return "No audio detected."
            
        # Process the audio with Whisper
        print("Transcribing audio...")
        inputs = self.processor(audio_array, return_tensors="pt", sampling_rate=self.sample_rate)
        input_features = inputs.input_features
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(inputs=input_features)
            
        # Decode the transcription
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return transcription

class VoiceChatSystem:
    def __init__(self, 
                 fish_token=None, 
                 whisper_model="openai/whisper-tiny.en", 
                 voice_ref_audio=None, 
                 voice_ref_text=None):
        """
        Initialize the complete voice chat system.
        
        Args:
            fish_token (str, optional): Fish API token
            whisper_model (str, optional): Whisper model name
            voice_ref_audio (str, optional): Path to reference audio for TTS voice
            voice_ref_text (str, optional): Reference text for TTS voice
        """
        # Initialize the TTS component
        self.tts = FishAudioTTS(token=fish_token)
        
        # Initialize the transcription component
        self.transcriber = MicrophoneTranscriber(model_name=whisper_model)
        
        # Store reference voice info if provided
        self.voice_ref_audio = voice_ref_audio
        self.voice_ref_text = voice_ref_text
        self.reference_audios = []
        
        # If reference voice info is provided, create the reference audio object
        if voice_ref_audio and voice_ref_text:
            self.reference_audios = [self.tts.add_reference_audio(voice_ref_audio, voice_ref_text)]
            
    def listen(self, duration=5):
        """
        Listen to the microphone and transcribe.
        
        Args:
            duration (int): Duration to listen in seconds
            
        Returns:
            str: Transcribed text
        """
        return self.transcriber.transcribe(duration=duration)
        
    def speak(self, text, output_path="response.mp3"):
        """
        Convert text to speech using the configured voice.
        
        Args:
            text (str): Text to speak
            output_path (str): Path to save the audio file
            
        Returns:
            bool: True if successful
        """
        return self.tts.generate_tts(text=text, 
                                    output_path=output_path, 
                                    reference_audios=self.reference_audios)
                                    
    def voice_chat_loop(self, response_path="response.mp3"):
        """
        Run an interactive voice chat loop.
        
        Args:
            response_path (str): Path to save TTS responses
        """
        print("\nVoice Chat System")
        print("=================")
        print("Say something to start the conversation.")
        print("Say 'goodbye' or 'exit' to end the conversation.")
        
        import pygame
        pygame.mixer.init()
        
        while True:
            print("\nListening...")
            transcription = self.listen()
            
            if not transcription or transcription.strip() == "":
                print("I didn't catch that. Please try again.")
                continue
                
            print(f"You said: {transcription}")
            
            # Check for exit commands
            if any(word in transcription.lower() for word in ["goodbye", "exit", "quit", "bye"]):
                print("Ending voice chat. Goodbye!")
                break
                
            # Here you would send the transcription to your chat system
            # and get a response. For now, we'll just echo it back
            response = f"I heard you say: {transcription}"
            
            # Convert the response to speech
            print(f"Generating speech response...")
            success = self.speak(response, output_path=response_path)
            
            if success:
                # Play the response
                print("Playing response...")
                pygame.mixer.music.load(response_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            else:
                print("Failed to generate speech response.")
