import os
import getpass
import faiss
import numpy as np
import io, re
import warnings
import requests
from hugchat import hugchat
from rich import print as rp
from hugchat.login import Login
from dotenv import load_dotenv, find_dotenv
import speech_recognition
from TTS.api import TTS
from git import Repo
import time
from playsound import playsound
from langchain import hub
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers.document_compressors import EmbeddingsFilter

from system_prompts import __all__ as prompts

load_dotenv(find_dotenv())

warnings.filterwarnings("ignore")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")

class LLMChatBot:
    def __init__(self, email, password, cookie_path_dir='./cookies/', default_llm=1):
        self.email = os.getenv("EMAIL")
        self.password = os.getenv("PASSWD")
        self.current_model = 1
        self.cookie_path_dir = cookie_path_dir
        self.cookies = self.login()
        self.chatbot = hugchat.ChatBot(cookies=self.cookies.get_dict(),
                                       default_llm=default_llm, 
                                       )
        self.repo_url = 'https://github.com/langchain-ai/langchain'
        self.default_system_prompt = prompts['default_rag_prompt']
        self.conv_id = None

        self.setup_tts()
        self.setup_speech_recognition()

    def login(self):
        rp("Attempting to log in...")
        sign = Login(self.email , self.password)
        try:
            cookies = sign.login(cookie_dir_path=self.cookie_path_dir, save_cookies=True)
            rp("Login successful!")
            return cookies
        except Exception as e:
            rp(f"Login failed: {e}")
            rp("Attempting manual login with requests...")
            self.manual_login()
            raise

    def manual_login(self):
        login_url = "https://huggingface.co/login"
        session = requests.Session()
        response = session.get(login_url)
        rp("Response Cookies:", response.cookies)
        rp("Response Content:", response.content.decode())

        csrf_token = response.cookies.get('csrf_token')
        if not csrf_token:
            rp("CSRF token not found in cookies.")
            return

        login_data = {
            'email':  os.getenv("EMAIL"),
            'password': os.getenv("PASSWD"),
            'csrf_token': csrf_token
        }

        response = session.post(login_url, data=login_data)
        if response.ok:
            rp("Manual login successful!")
        else:
            rp("Manual login failed!")

    def setup_speech_recognition(self):
        self.recognizer = speech_recognition.Recognizer()

    def setup_folders(self):
        self.dirs = ["test_input","vectorstore"]
        for d in self.dirs:
            os.makedirs(d, exist_ok=True)

    def setup_tts(self, model_name="tts_models/en/ljspeech/fast_pitch"):
        self.tts = TTS(model_name=model_name)

    def __call__(self, text, system_prompt=""):  # llama 3
        self.conv_id = self.chatbot.new_conversation(system_prompt=system_prompt, modelIndex=self.current_model, switch_to=True)
        return self.send_message(text)

    def send_message(self, message):
        message_result = self.chatbot.chat(message)
        return message_result.wait_until_done()

    def stream_response(self, message):
        for resp in self.chatbot.query(message, stream=True):
            rp(resp)

    def web_search(self, query):
        query_result = self.chatbot.query(query, web_search=True)
        results = []
        for source in query_result.web_search_sources:
            results.append({
                'link': source.link,
                'title': source.title,
                'hostname': source.hostname
            })
        return results

    def create_new_conversation(self, switch_to=True, system_prompt=""):
        self.chatbot.new_conversation(switch_to=switch_to, modelIndex=self.current_model, system_prompt=system_prompt)

    def get_remote_conversations(self):
        return self.chatbot.get_remote_conversations(replace_conversation_list=True)

    def get_local_conversations(self):
        return self.chatbot.get_conversation_list()

    def get_available_models(self):
        return self.chatbot.get_available_llm_models()

    def switch_model(self, index):
        self.chatbot.switch_llm(index)

    def switch_conversation(self, id):
        self.conv_id = id
        self.chatbot.change_conversation(self.conv_id)

    def get_assistants(self):
        return self.chatbot.get_assistant_list_by_page(1)

    def switch_role(self, system_prompt):
        self.chatbot.delete_all_conversations()
        result=self.chatbot.new_conversation(switch_to=True, system_prompt=self.default_system_prompt,  modelIndex=1)
        return result
    
    def listen_for_speech(self):
        with speech_recognition.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source, timeout=0.9)
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except speech_recognition.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except speech_recognition.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    def optimized_tts(self, text: str, output_file: str = "output.wav", speaking_rate: float = 3) -> str:
        start_time = time.time()

        self.tts.tts_to_file(
            text=text,
            emotion='scared',
            file_path=output_file,
            speaker=self.tts.speakers[0] if self.tts.speakers else None,
            speaker_wav="tortoise-tts/examples/favorites/emma_stone_courage.mp3",
            language=self.tts.languages[0] if self.tts.languages else None,
            speed=speaking_rate,
            split_sentences=True
        )

        end_time = time.time()
        print(f"TTS generation took {end_time - start_time:.2f} seconds")
        return output_file

    @staticmethod
    def Play(file_path):
        playsound(file_path)

# Example usage:
if __name__ == "__main__":
    email = os.getenv("EMAIL")
    password = os.getenv("PASSWORD")
    
    if not email or not password:
        email = input("Enter your email: ")
        password = getpass.getpass("Enter your password: ")

    bot = LLMChatBot(email=email, password=password)
    bot.interactive_mode()