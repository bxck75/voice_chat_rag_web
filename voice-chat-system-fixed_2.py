import torch
import pygame
from transformers import AutoProcessor, WhisperForConditionalGeneration
import sounddevice as sd
import numpy as np
import ormsgpack
import httpx
from llama_cpp import Llama
        
import multiprocessing

from langchain_community.chat_models import ChatLlamaCpp
from typing import List, Literal, Optional, Annotated
from gradio_client import Client

from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv, find_dotenv
import time
import queue
import warnings
import requests
modelId='5a897df8817c441cbd8f5f5d81816404&taskId=2025e593c1124f29a5606e61220f0d02'
import os
load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")
model_base_path ="/media/codemonkeyxl/DATA2/new_comfyui/ComfyUI/models" 
# Ensure environment variables are loaded (e.g., FISH_TOKEN)

def get_fish_token(): 
    if fish_token is None:
        fish_token = os.getenv("FISH_TOKEN")
        if not fish_token:
            raise ValueError("Fish Audio token is missing. Provide a token or set FISH_TOKEN in your env.")
        else:
            print(f"Loaded token: {fish_token[:10]}... (truncated)")
        
    return fish_token

# Ensure environment variables are loaded (e.g., FISH_TOKEN)
def get_hf_token():
    if hf_token is None:
        hf_token = os.getenv("hf_TOKEN")
        if not hf_token:
            raise ValueError("Fish Audio token is missing. Provide a token or set hf_TOKEN in your env.")
        else:
            print(f"Loaded token: {hf_token[:10]}... (truncated)")
        
    return hf_token


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: int = Field(default=200, ge=100, le=300)
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "mp3"
    mp3_bitrate: Literal[64, 128, 192] = 128
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    reference_id: Optional[str] = None
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
                     mp3_bitrate: Literal[64, 128, 192] = 192,
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
        print(f"Loading Whisper model: {model_name}")
        from transformers import AutoProcessor, WhisperForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.copy())
        
    def start_recording(self, duration=5):
        self.audio_queue = queue.Queue()
        self.is_recording = True
        print(f"Recording for {duration} seconds... Speak now!")
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self._audio_callback):
            sd.sleep(int(duration * 1000))
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
        audio_array = np.concatenate(audio_data).flatten() if audio_data else np.zeros(0)
        self.is_recording = False
        print("Recording finished.")
        return audio_array
    
    def transcribe(self, audio_array=None, duration=5):
        if audio_array is None:
            audio_array = self.start_recording(duration)
        if len(audio_array) == 0:
            return "No audio detected."
        print("Transcribing audio...")
        inputs = self.processor(audio_array, return_tensors="pt", sampling_rate=self.sample_rate)
        input_features = inputs.input_features
        with torch.no_grad():
            generated_ids = self.model.generate(inputs=input_features)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription

class ChatBot:
    def __init__(self, model_path: str,
                 max_tokens: int = 64,
                 context_windows_size: int = 2048,
                 temperature: float = 0.2,
                 repetition_penalty= 3.8,
                 top_p: float = 0.92,
                 reasoning_time: int = 10):
        """
        Initialize the ChatBot with a gguf Llama model.
        
        Args:
            model_path (str): Path to the gguf model file.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling parameter.
            reasoning_time (int): Parameter to tune reasoning intensity/time.
                                  (You can experiment with adjusting this to affect output depth.)
        """
        # Load the Llama model via llama-cpp-python.
        self.context_windows_size = context_windows_size
        self.repetition_penalty = repetition_penalty
        self.model_path = model_path
        self.model = Llama(
            model_path=model_path,
            repetition_penalty=self.repetition_penalty,  # Repetition penalty; set to 1.0 to disable.
            n_ctx=context_windows_size,
            ) # Context window size; adjust if necessary.
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.reasoning_time = reasoning_time

    def generate_response(self, prompt: str) -> str:
        """
        Generate a chatbot response using the Llama model.
        
        Args:
            prompt (str): Input prompt text.
        
        Returns:
            str: Generated response text.
        """
        # Optionally, you could incorporate self.reasoning_time to tweak parameters.


        llm = ChatLlamaCpp(
            temperature=0.5,
            model_path=self.model_path,
            n_ctx=32768,
            n_gpu_layers=8,
            n_batch=300,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            max_tokens=self.max_tokens,
            n_threads=multiprocessing.cpu_count() - 1,
            repeat_penalty=self.repetition_penalty,
            top_p=self.top_p,
            verbose=True,
        )

        messages = [f'''<|im_start|>system{self.system_prompt}<|im_end|>
                    <|im_start|>user{prompt}<|im_end|>
                    <|im_start|>assistant''']
        

        ai_msg = llm.invoke(messages)

        output = self.model(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        return output['choices'][0]['text'].strip()
    

class VoiceChatSystem:
    def __init__(
        self,
        fish_token: str,
        whisper_model: str = "openai/whisper-tiny.en",
        chatbot: Optional[ChatBot] = None,
        voice_ref_audio: Optional[str] = None,
        voice_ref_text: Optional[str] = None,
        volume_threshold: float = 0.02,
        silence_threshold: float = 0.02,
        silence_duration: float = 6.0,
        max_record_duration: float = 60.0
    ):
        self.tts = FishAudioTTS(token=fish_token)
        self.transcriber_system_prompt = """
        you a master of the ancient art of transcribing audio.
        your task is to transcribe the audio as accurately as possible.
        you are given the following audio: """
        self.transcriber = MicrophoneTranscriber(model_name=whisper_model)
        self.chatbot = chatbot
        self.voice_ref_audio = voice_ref_audio
        self.voice_ref_text = voice_ref_text
        self.reference_audios = []
        self.volume_threshold = volume_threshold
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_record_duration = max_record_duration
        
        if voice_ref_audio and voice_ref_text:
            self.reference_audios = [self.tts.add_reference_audio(self.voice_ref_audio, self.voice_ref_text )]
        
        # Initialize the Whisper processor and model with correct attention mask handling
        self._initialize_whisper_model(whisper_model)
    
    def _initialize_whisper_model(self, model_name):
        """Initialize the Whisper model with proper attention mask handling"""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            import torch
            
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            # Set device to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = self.whisper_model.to(self.device)
            
            print(f"Initialized Whisper model {model_name} on {self.device}")
        except ImportError:
            print("Warning: transformers library not found. Falling back to default transcriber.")
            self.processor = None
            self.whisper_model = None
    
    def transcribe_audio_with_attention_mask(self, audio_file):
        """Transcribe audio with proper attention mask handling"""
        if self.processor is None or self.whisper_model is None:
            # Fall back to the default transcriber if custom whisper isn't available
            return self.transcriber.transcribe_file(audio_file)
        
        try:
            import librosa
            import torch
            
            # Load audio
            audio, sample_rate = librosa.load(audio_file, sr=16000)
            
            # Process audio
            input_features = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate a proper attention mask (all 1s for the input)
            attention_mask = torch.ones_like(input_features[:, :, 0])
            
            # Generate token ids with attention mask
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    max_length=448,  # Adjust based on your needs
                    num_beams=5
                )
            
            # Decode the predicted ids
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription
        except Exception as e:
            print(f"Error in custom transcription: {e}")
            # Fall back to default transcriber
            return self.transcriber.transcribe_file(audio_file)
    
    def listen_continuous(self) -> str:
        """
        Continuously monitors audio input and starts recording when volume exceeds threshold.
        Stops recording when silent for specified duration or max recording time is reached.
        """
        import pyaudio
        import numpy as np
        import wave
        import tempfile
        import time
        
        # Audio parameters
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024
        
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        
        print("Monitoring audio levels... (speak to begin)")
        
        frames = []
        is_recording = False
        silent_chunks = 0
        silent_time = 0
        start_time = None
        silence_start = None
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        transcription = ""
        
        try:
            while True:
                data = stream.read(CHUNK)
                audio_array = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_array).mean() / 32767.0  # Normalize volume
                
                # Start recording if volume is above threshold
                if not is_recording and volume > self.volume_threshold:
                    print("Recording started...")
                    is_recording = True
                    frames = [data]  # Start with current chunk
                    start_time = time.time()
                    silent_time = 0
                    silence_start = None
                
                # While recording, collect audio and check stop conditions
                elif is_recording:
                    frames.append(data)
                    
                    # Check for silence
                    if volume < self.silence_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        
                        silent_time = time.time() - silence_start
                        
                        # Visual indicator of silence duration
                        silent_chunks += 1
                        if silent_chunks % 10 == 0:
                            print(f"Silence detected for {silent_time:.1f}s")
                    else:
                        silence_start = None
                        silent_time = 0
                        silent_chunks = 0
                    
                    # Check if we should stop recording
                    elapsed_time = time.time() - start_time
                    
                    # Stop if silent for too long or max duration reached
                    if (silent_time > self.silence_duration) or (elapsed_time > self.max_record_duration):
                        print(f"Recording stopped after {elapsed_time:.1f} seconds")
                        
                        # First properly close the stream and release resources
                        stream.stop_stream()
                        stream.close()
                        audio.terminate()
                        
                        # Save audio to temporary file
                        wf = wave.open(temp_file.name, 'wb')
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(audio.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))
                        wf.close()
                        
                        # Reset recording state
                        is_recording = False
                        
                        # Transcribe the audio using the method with attention mask
                        transcription = self.transcribe_audio_with_attention_mask(temp_file.name)
                        break  # Exit the while loop
                
                # Visual indicator of audio level when not recording
                if not is_recording and volume > 0.01:
                    bars = int(volume * 50)
                    print(f"\rAudio level: {'|' * bars}{' ' * (50-bars)} [{volume:.2f}]", end="")
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
        
        except Exception as e:
            print(f"\nError during recording: {e}")
        
        finally:
            # Ensure resources are properly cleaned up
            try:
                if 'stream' in locals() and stream is not None:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
            except:
                pass
                
            try:
                if 'audio' in locals() and audio is not None:
                    audio.terminate()
            except:
                pass
        
        return transcription
        
    def transcribe_file(self, audio_file: str) -> str:
        """Transcribe a saved audio file using the whisper model with attention mask"""
        try:
            return self.transcribe_audio_with_attention_mask(audio_file)
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""
            
    def speak(self, text: str, output_path="response.mp3", debug=False) -> bool:
        if debug:
            print(f"got prompted with text: {text}")
        return self.tts.generate_tts(text=text, output_path=output_path, reference_audios=self.reference_audios)
    
    
    def voice_chat_loop(self, response_path="response.mp3"):
        print("\nVoice Chat System with ChatBot Integration")
        print("============================================")
        print("Speak to start the conversation. Say 'goodbye' to exit.")
        pygame.mixer.init()
        
        while True:
            try:
                print("\nListening for speech...")
                transcription = self.listen_continuous()
                
                if not transcription or transcription.strip() == "":
                    didnt_catch_txt = "I didn't catch that. Please try again."
                    self.speak(didnt_catch_txt, response_path)
                    continue
                    
                print(f"You said: {transcription}")
                
                if any(word in transcription.lower() for word in ["goodbye", "exit", "quit", "bye"]):
                    print("Ending conversation. Goodbye!")
                    break
                    
                # Prepare the prompt for the chatbot.
                prompt = f"System: {self.transcriber_system_prompt}\nUser: {transcription}\nChatBot: "
                
                if self.chatbot:
                    response = self.chatbot.generate_response(prompt)
                else:
                    response = f"I heard you say: {transcription}"
                    
                print(f"ChatBot Response: {response}")
                print("Generating speech response...")
                
                if self.speak(response, output_path=response_path):
                    print("Playing response...")
                    pygame.mixer.music.load(response_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                else:
                    print("Failed to generate speech response.")
                    
            except Exception as e:
                print(f"Error in voice chat loop: {e}")
                print("Restarting listening...")

def get_key_value_from_dict_list(dict_list, index):
    # Get the dictionary at the specified index
    if 0 <= index < len(dict_list):
        selected_dict = dict_list[index]
        
        # For dictionaries with a single key-value pair
        if len(selected_dict) == 1:
            # Get the first (and only) key and its value
            audio_path = list(selected_dict.keys())[0]
            text = selected_dict[audio_path]
            return audio_path, text
        else:
            # If the dictionary has multiple key-value pairs
            # You might need to specify which one you want
            print(f"Dictionary at index {index} has multiple key-value pairs")
            return list(selected_dict.items())
    else:
        print(f"Index {index} out of range for list of length {len(dict_list)}")
        return None, None
                             
# Example usage when running this file directly
if __name__ == "__main__":
    base_model_path="/media/codemonkeyxl/DATA2/new_comfyui/ComfyUI/models"
    base_app_path = "/media/codemonkeyxl/TBofCode/MainCodingFolder/new_coding/expert_augmenting_system/voice_chat_rag_web"
    # Define the path to the gguf model in your comfyui models/LLM folder.
    model_path = os.path.join(base_model_path, "LLM", "qwen2.5-coder-1.5b-instruct-q4_0.gguf")
    
    # Initialize the ChatBot with tuning parameters (adjust max_tokens, temperature, etc. as needed)
    chatbot = ChatBot(model_path=model_path, max_tokens=16, temperature=2.8, top_p=0.95, reasoning_time=40)
    
    # Define your reference audio path and text here
    ACTIVE_VOICE_INDEX=1
    VOICE_REF =[
            {
                "dj-tools-female-vocals-cj-30265.mp3":"""A digital cyberpunk art splash in the style of bo-cyborgsplash, 
                                                        a vibrant and mystical landscape depicted in a series of vertical banners, 
                                                        each featuring a different color palette, 
                                                        the first banner on the left shows a glowing tree with intricate patterns, 
                                                        the second banner features a swirling orange and red design, 
                                                        the third banner features an abstract landscape with a tree in the center.""",
            },
            {
                "hornyclown2.mp3": """A digital cyberpunk art splash in the style of bo-cyborgsplash, a vibrant and mystical landscape depicted in a series of vertical banners, each featuring a different color palette, the first banner on the left shows a glowing tree with intricate patterns, the second banner features a swirling orange and red design, the third banner features an abstract landscape with a tree in the center."""
            }
    ]


    # Get the key (audio path) and value (text) from the second dictionary (index 1)
    index = 1
    audio_path, text = get_key_value_from_dict_list(VOICE_REF, ACTIVE_VOICE_INDEX)

    print(f"Audio path: {audio_path}")
    print(f"Text: {text}")


    #print(str(VOICE_REF[ACTIVE_VOICE_INDEX].items()[1])) 
    # Initialize the voice chat system with the chatbot integrated
    system = VoiceChatSystem(
        chatbot=chatbot,
        voice_ref_audio=audio_path,
        voice_ref_text=text,
        
        fish_token=os.getenv("FISH_TOKEN"),
    )

    # Start the interactive voice chat loop
    system.voice_chat_loop()

