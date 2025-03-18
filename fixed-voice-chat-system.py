import torch
import pygame
import numpy as np
import ormsgpack
import httpx
from llama_cpp import Llama
import sounddevice as sd
import queue
import warnings
import os
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
from transformers import AutoProcessor, WhisperForConditionalGeneration

# Load environment variables
load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")

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
        warnings.filterwarnings("ignore")
        
        # Get token from parameter or environment
        self.token = token
        if self.token is None:
            self.token = os.getenv('FISH_TOKEN')
        
        if not self.token:
            raise ValueError("Fish Audio token is missing. Provide a token or set FISH_TOKEN in your env.")
        else:
            print(f"Loaded Fish token: {self.token[:5]}... (truncated)")
    
    def add_reference_audio(self, audio_path: str, text: str) -> ServeReferenceAudio:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        return ServeReferenceAudio(audio=audio_bytes, text=text)
    
    def generate_tts(self, text: str, output_path: str,
                     reference_audios: List[ServeReferenceAudio] = None,
                     reference_id: str = None,
                     chunk_length: int = 200,
                     output_format: Literal["wav", "pcm", "mp3"] = "mp3",
                     mp3_bitrate: Literal[64, 128, 192] = 128,
                     normalize: bool = True,
                     latency: Literal["normal", "balanced"] = "normal") -> bool:
        
        # Use empty list if no reference audios
        if reference_audios is None:
            reference_audios = []
            
        request = ServeTTSRequest(
            text=text,
            references=reference_audios,
            reference_id=reference_id,
            chunk_length=chunk_length,
            format=output_format,
            mp3_bitrate=mp3_bitrate,
            normalize=normalize,
            latency=latency
        )
        
        try:
            with httpx.Client() as client:
                response = client.post(
                    "https://api.fish.audio/v1/tts",
                    content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
                    headers={
                        "authorization": f"Bearer {self.token}",
                        "content-type": "application/msgpack",
                    },
                    timeout=30.0,  # Adding a timeout to prevent infinite waiting
                )
                
                if response.status_code != 200:
                    print(f"Error: API returned status code {response.status_code}")
                    print(f"Response content: {response.text}")
                    return False
                
                # Write response content to file
                with open(output_path, "wb") as f:
                    f.write(response.content)
                    
                print(f"TTS generated successfully and saved to {output_path}")
                return True
                
        except Exception as e:
            print(f"Error generating TTS: {str(e)}")
            return False


class MicrophoneTranscriber:
    def __init__(self, model_name="openai/whisper-tiny.en", sample_rate=16000):
        print(f"Loading Whisper model: {model_name}")
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
                 max_tokens: int = 150,
                 temperature: float = 0.8,
                 top_p: float = 0.9,
                 reasoning_time: int = 10):
        """
        Initialize the ChatBot with a gguf Llama model.
        
        Args:
            model_path (str): Path to the gguf model file.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling parameter.
            reasoning_time (int): Parameter influencing complexity of reasoning.
        """
        print(f"Loading LLM from: {model_path}")
        try:
            # Properly configure llama-cpp parameters based on reasoning time
            n_threads = min(os.cpu_count() or 4, 8)  # Default to using up to 8 threads 
            self.model = Llama(
                model_path=model_path,
                n_ctx=2048,        # Context window size
                n_threads=n_threads,
                n_batch=512,       # Batch size for prompt processing
                verbose=False      # Set to True for debugging
            )
            
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.top_p = top_p
            
            # Use reasoning_time to adjust parameters
            # Lower values = faster but less careful reasoning
            if reasoning_time < 5:
                self.temperature = min(1.0, temperature * 1.2)  # Higher temperature, more randomness
            elif reasoning_time > 15:
                self.temperature = max(0.1, temperature * 0.8)  # Lower temperature, more focused
                
            print("ChatBot LLM loaded successfully")
        except Exception as e:
            print(f"Error loading LLM: {str(e)}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a chatbot response using the Llama model.
        
        Args:
            prompt (str): Input prompt text.
        
        Returns:
            str: Generated response text.
        """
        try:
            output = self.model(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=["User:", "System:"],  # Stop generation at these tokens
                echo=False  # Don't include the prompt in the output
            )
            
            if not output or 'choices' not in output or len(output['choices']) == 0:
                return "I'm sorry, I couldn't generate a response."
                
            return output['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Sorry, I encountered an error while generating a response."


class VoiceChatSystem:
    def __init__(self,
                 fish_token: str = None,
                 whisper_model: str = "openai/whisper-tiny.en",
                 llm_model_path: str = None,
                 voice_ref_audio: Optional[str] = None,
                 voice_ref_text: Optional[str] = None
                ):
        # Initialize TTS system
        self.tts = FishAudioTTS(token=fish_token)
        
        # Initialize transcriber
        self.transcriber = MicrophoneTranscriber(model_name=whisper_model)
        
        # Initialize chatbot if model path provided
        self.chatbot = None
        if llm_model_path and os.path.exists(llm_model_path):
            try:
                self.chatbot = ChatBot(model_path=llm_model_path, max_tokens=512, temperature=0.2)
                print(f"ChatBot initialized with model: {llm_model_path}")
            except Exception as e:
                print(f"Failed to initialize ChatBot: {str(e)}")
        else:
            print("No LLM model path provided or file not found. ChatBot will be disabled.")
        
        # System prompt for chat interactions
        self.transcriber_system_prompt = """
        You are an AI assistant having a conversation with a user.
        Respond to the user's message naturally and helpfully.
        Keep responses concise but informative.
        """
        
        # Set up voice reference if provided
        self.voice_ref_audio = voice_ref_audio
        self.voice_ref_text = voice_ref_text
        self.reference_audios = []
        
        if voice_ref_audio and voice_ref_text and os.path.exists(voice_ref_audio):
            try:
                self.reference_audios = [self.tts.add_reference_audio(voice_ref_audio, voice_ref_text)]
                print(f"Voice reference added from: {voice_ref_audio}")
            except Exception as e:
                print(f"Failed to add voice reference: {str(e)}")
        
    def listen(self, duration=5) -> str:
        """Record audio and transcribe it."""
        return self.transcriber.transcribe(duration=duration)
        
    def speak(self, text: str, output_path="response.mp3", debug=False) -> bool:
        """Generate speech from text using TTS."""
        if debug:
            print(f"Speaking text: {text}")
        
        try:
            result = self.tts.generate_tts(
                text=text, 
                output_path=output_path, 
                reference_audios=self.reference_audios
            )
            return result
        except Exception as e:
            print(f"Error in speak method: {str(e)}")
            return False
                                    
    def voice_chat_loop(self, response_path="response.mp3"):
        """Run an interactive voice chat loop."""
        print("\nVoice Chat System")
        print("================")
        print("Speak to start the conversation. Say 'goodbye' to exit.")
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        while True:
            print("\nListening...")
            try:
                # Listen for user input
                transcription = self.listen()
                
                # Check if we got valid input
                if not transcription or transcription.strip() == "":
                    print("No speech detected.")
                    self.speak("I didn't catch that. Please try again.", response_path)
                    pygame.mixer.music.load(response_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    continue
                    
                print(f"You said: {transcription}")
                
                # Check for exit commands
                if any(word in transcription.lower() for word in ["goodbye", "exit", "quit", "bye"]):
                    print("Ending conversation. Goodbye!")
                    self.speak("Goodbye! Have a great day.", response_path)
                    pygame.mixer.music.load(response_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    break
                
                # Generate response
                if self.chatbot:
                    # Prepare prompt for the chatbot
                    prompt = f"System: {self.transcriber_system_prompt}\nUser: {transcription}\nAssistant: "
                    response = self.chatbot.generate_response(prompt)
                else:
                    # Echo mode if no chatbot is available
                    response = f"I heard you say: {transcription}"
                
                print(f"Response: {response}")
                
                # Convert response to speech
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
                print(f"Error in voice chat loop: {str(e)}")
                print("Continuing to next iteration...")


# Example usage
if __name__ == "__main__":
    # Base paths
    base_model_path = "/media/codemonkeyxl/DATA2/new_comfyui/ComfyUI/models"
    
    # LLM model path
    llm_model_path = os.path.join(base_model_path, "LLM", "DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-IQ3_XS.gguf")
    
    # Voice reference
    voice_ref_audio = "fishvoice.wav"
    voice_ref_text = "Are you familiar with it? Slice the steak and place the strips on top, then garnish with the dried cranberries, pine nuts, and blue cheese. I wonder how people rationalise the decision? "
    
    # Initialize the voice chat system
    system = VoiceChatSystem(
        fish_token=os.getenv("FISH_TOKEN"),
        whisper_model="openai/whisper-tiny.en",
        llm_model_path=llm_model_path,
        voice_ref_audio=voice_ref_audio,
        voice_ref_text=voice_ref_text
    )
    

    # Start the interactive voice chat loop
    system.voice_chat_loop()
