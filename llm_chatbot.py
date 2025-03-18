import os
import getpass
import faiss
import numpy as np
import io,re
import faiss
import warnings
import requests
from hugchat import hugchat
from rich import print as rp
from hugchat.login import Login
from dotenv import load_dotenv,find_dotenv
import speech_recognition
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

from system_prompts import (default_rag_prompt,story_teller_prompt,todo_parser_prompt,
                            code_generator_prompt,software_tester_prompt,script_debugger_prompt,iteration_controller_prompt,copilot_prompt)
prompts={'default_rag_prompt':default_rag_prompt,
        'story_teller_prompt':story_teller_prompt,
        'todo_parser_prompt':todo_parser_prompt,
        'code_generator_prompt':code_generator_prompt,
        'software_tester_prompt':software_tester_prompt,
        'script_debugger_prompt':script_debugger_prompt,
        'iteration_controller_prompt':iteration_controller_prompt,
        'copilot_prompt':copilot_prompt
        }

load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")
token = os.getenv("HF_TOKEN")

class LLMChatBot:
    def __init__(self, email, password, cookie_path_dir='./cookies/',default_llm=1):
        self.email = email
        self.password = password
        self.current_model = 1
        self.cookie_path_dir = cookie_path_dir
        self.cookies = self.login()
        self.chatbot = hugchat.ChatBot(cookies=self.cookies.get_dict(), 
                                       default_llm = default_llm, #CohereForAI/c4ai-command-r-plus
                                       )
        self.repo_url='https://github.com/langchain-ai/langchain'
        self.default_system_prompt = prompts['default_rag_prompt']
        self.conv_id = None
        self.latest_splitter=None
        self.setup_folders()
        self.embeddings=HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.create_vectorstore_from_github()
      
        self.setup_retriever()
        #self.setup_tts()
        self.setup_speech_recognition()

    def login(self):
        rp("Attempting to log in...")
        sign = Login(self.email, self.password)
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
            'email': self.email,
            'password': self.password,
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
        self.dirs=["test_input"]
        for d in self.dirs:
            os.makedirs(d, exist_ok=True)

    #def setup_tts(self, model_name="tts_models/en/ljspeech/fast_pitch"):
        #self.tts = TTS(model_name=model_name)

    def __call__(self, text, system_prompt=""): # llama 3
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

    def create_new_conversation(self,switch_to=True,  system_prompt = ""):
        self.chatbot.new_conversation(switch_to=switch_to, modelIndex = self.current_model, system_prompt = system_prompt)

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

    def switch_role(self,system_prompt):
        self.chatbot.delete_all_conversations()
        return self.chatbot.new_conversation(switch_to=True, system_prompt=self.default_system_prompt)
    
    def listen_for_speech(self):
        with speech_recognition.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
            
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

        '''self.tts.tts_to_file(
            text=text,
            emotion='scared',
            file_path=output_file,
            speaker=self.tts.speakers[0] if self.tts.speakers else None,
            speaker_wav="tortoise-tts/examples/favorites/emma_stone_courage.mp3",
            language=self.tts.languages[0] if self.tts.languages else None,
            speed=speaking_rate,
            split_sentences=True
        )'''

        end_time = time.time()
        print(f"TTS generation took {end_time - start_time:.2f} seconds")
        return output_file
    
    @staticmethod
    def Play(file_path):
        playsound(file_path)

    def add_documents_folder(self, folder_path):
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                self.add_document(file_path)

    def add_document(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            document = Document(page_content=content)
            self.vector_store.add_documents([document])

    def add_document_from_url(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            content = response.text
            document = Document(page_content=content)
            self.vector_store.add_documents([document])
        else:
            print(f"Failed to fetch URL content: {response.status_code}")

    def delete_document(self, document):
        if document in self.vector_store:
            self.vector_store.delete_document(document)
            print(f"Deleted document: {document}")
        else:
            print(f"Document not found: {document}")

    def _add_to_vector_store(self, name, content):
        document = Document(page_content=content)
        self.vector_store.add_documents([document])
        print(f"Added document to vector store: {name}")
        # Example of updating the vectorizer (you might need to adjust based on your actual implementation)
        self.vectorizer.fit_transform(self.vector_store.get_all_documents())
    
    def clone_github_repo(self, repo_url, local_path='./repo'):
        if os.path.exists(local_path):
            print("Repository already cloned.") 
            return local_path
        Repo.clone_from(repo_url, local_path)
        return local_path

    def load_documents_from_github(self, repo_url, file_types=['*.py', '*.md', '*.txt', '*.html']):
        local_repo_path = self.clone_github_repo(repo_url)
        loader = DirectoryLoader(path=local_repo_path, glob=f"**/{{{','.join(file_types)}}}", show_progress=True, recursive=True)
        return loader.load()

    def split_documents(self, documents: list,chunk_s=512,chunk_o=0):
        split_docs = []
        splitter=None
        for doc in documents:
            ext = os.path.splitext(getattr(doc, 'source', '') or getattr(doc, 'filename', ''))[1].lower()
            if ext == '.py':
                splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=chunk_s, chunk_overlap=chunk_o)
            elif ext in ['.md', '.markdown']:
                splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=chunk_s, chunk_overlap=chunk_o)
            elif ext in ['.html', '.htm']:
                splitter = RecursiveCharacterTextSplitter.from_language(language=Language.HTML, chunk_size=chunk_s, chunk_overlap=chunk_o)
            else:
                splitter = CharacterTextSplitter(chunk_size=chunk_s, chunk_overlap=chunk_o, add_start_index=True)
            
            split_docs.extend(splitter.split_documents([doc]))
        return split_docs,splitter


    def setup_retriever(self, k=5, similarity_threshold=0.76):
        self.retriever = self.vectorstore.as_retriever(k=k)
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=self.retriever)

    def create_retrieval_chain(self):
        rag_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(self.bot, rag_prompt)
        self.high_retrieval_chain = create_retrieval_chain(self.compression_retriever, combine_docs_chain)
        self.low_retrieval_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

    def create_vectorstore_from_github(self):
        documents = self.load_documents_from_github(self.repo_url)
        split_docs,splitter = self.split_documents(documents,512,0)
        self.latest_splitter=splitter
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        print(f"Vectorstore created with {len(split_docs)} documents.")

    def update_vectorstore(self, new_documents):
        split_docs,splitter = self.split_documents(new_documents)
        self.latest_splitter=splitter
        self.vectorstore.add_documents(split_docs)
        print(f"Vectorstore updated with {len(split_docs)} new documents.")
    

    def retrieve_with_chain(self, query, mode='high'):
        if mode == 'high':
            return self.high_retrieval_chain.invoke({"input": query})
        else:
            return self.low_retrieval_chain.invoke({"input": query})
if __name__ == '__main__':
    EMAIL = os.getenv("EMAIL")
    PASSWD =  os.getenv("PASSWD")
    model=1
    chatbot = LLMChatBot(EMAIL, PASSWD, default_llm=model)
    chatbot.create_new_conversation(system_prompt=chatbot.default_system_prompt, switch_to=True)
    #all_models=chatbot.get_available_models()
    #rp(all_models[chatbot.current_model].name) 
    results=chatbot("""Tel me a short crafting survival Scify story of K.U.T.H.O.E.R """)
    audio_path = chatbot.optimized_tts(str(results))
    chatbot.Play(audio_path)
    rp(results)