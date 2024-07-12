import streamlit as st
from llm_chatbot import LLMChatBot
from streamlit_option_menu import option_menu
import speech_recognition as sr
import pyttsx3
import os
import getpass
from uuid import uuid4
import faiss
import numpy as np
import requests
import io
import warnings
import torch
import pickle
import asyncio
import json
from git import Repo
from rich import print as rp
from typing import Union, List, Generator, Any, Mapping, Optional, Dict
from requests.sessions import RequestsCookieJar
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.vectorstores.base import VectorStore
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.llms import BaseLLM
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from uber_toolkit_class import UberToolkit
from glob import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from langchain_core.documents import Document
from scipy.stats import gaussian_kde
from huggingface_hub import InferenceClient
from hugchat import hugchat
from hugchat.login import Login
from hugchat.message import Message
from hugchat.types.assistant import Assistant
from hugchat.types.model import Model
from hugchat.types.message import MessageNode, Conversation
from langchain_community.document_loaders import TextLoader
from TTS.api import TTS
import time
from playsound import playsound
from system_prompts import (default_rag_prompt, story_teller_prompt, todo_parser_prompt,
                            code_generator_prompt, software_tester_prompt, script_debugger_prompt, iteration_controller_prompt, copilot_prompt)
prompts={'default_rag_prompt':default_rag_prompt,
        'story_teller_prompt':story_teller_prompt,
        'todo_parser_prompt':todo_parser_prompt,
        'code_generator_prompt':code_generator_prompt,
        'software_tester_prompt':software_tester_prompt,
        'script_debugger_prompt':script_debugger_prompt,
        'iteration_controller_prompt':iteration_controller_prompt,
        'copilot_prompt':copilot_prompt
        }

from profiler import VoiceProfileManager, VoiceProfile

# Load environment variables
load_dotenv(find_dotenv())

class ChatbotApp:
    def __init__(self, email, password, default_llm=1):
        self.email = email
        self.password = password
        self.default_llm = default_llm
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
       


    def create_vectorstore_from_github(self):
        repo_url = "YOUR_REPO_URL"
        local_repo_path = self.clone_github_repo(repo_url)
        loader = DirectoryLoader(path=local_repo_path, glob=f"**/*", show_progress=True, recursive=True)
        loaded_files = loader.load()
        documents = [Document(page_content=file_content) for file_content in loaded_files]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_documents = text_splitter.split_documents(documents)
        texts = [doc.page_content for doc in split_documents]
        print(f"Texts for embedding: {texts}")  # Debug print
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
    
    def create_vectorstore(self, docs):
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # Wrap text content in Document objects
        documents = [Document(page_content=doc) for doc in docs]
        # Split documents using the text splitter
        split_documents = text_splitter.split_documents(documents)
        # Convert split documents back to plain text
        texts = [doc.page_content for doc in split_documents]
        vectorstore = FAISS.from_texts(texts, self.setup_embeddings())
        return vectorstore
    
    def setup_session_state(self):
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'voice_mode' not in st.session_state:
            st.session_state.voice_mode = False
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None
        if 'retriever' not in st.session_state:
            st.session_state.retriever = None
        if 'compression_retriever' not in st.session_state:
            st.session_state.compression_retriever = None

    def text_to_speech(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def speech_to_text(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                return text
            except:
                return "Sorry, I didn't catch that."



    def setup_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
 
    def create_vector_store(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # Wrap text content in Document objects
        documents = [Document(page_content=doc) for doc in docs]
        # Split documents using the text splitter
        split_documents = text_splitter.split_documents(documents)
        print(f"Split documents: {split_documents}")  # Debug print
        # Convert split documents back to plain text
        texts = [doc.page_content for doc in split_documents]
        print(f"Texts: {texts}")  # Debug print
        if not texts:
            print("No valid texts found for embedding. Check your repository content.")
            return

        try:
            self.vectorstore = FAISS.from_texts(texts, self.embeddings)
            print("Vector store created successfully")
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")


    def setup_retriever(self, k=5, similarity_threshold=0.76):
        self.retriever = st.session_state.vectorstore.as_retriever(k=k)
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.setup_embeddings())
        relevant_filter = EmbeddingsFilter(embeddings=self.setup_embeddings(), similarity_threshold=similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        st.session_state.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=self.retriever)

    def create_retrieval_chain(self):
        rag_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(self.llm, rag_prompt)
        self.high_retrieval_chain = create_retrieval_chain(st.session_state.compression_retriever, combine_docs_chain)
        self.low_retrieval_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

    def setup_tts(self, model_name="tts_models/en/ljspeech/fast_pitch"):
        self.tts = TTS(model_name=model_name, progress_bar=False, vocoder_path='vocoder_models/en/ljspeech/univnet')

    def setup_speech_recognition(self):
        self.recognizer = sr.Recognizer()

    def setup_folders(self):
        self.dirs = ["test_input", "vectorstore", "test"]
        for d in self.dirs:
            os.makedirs(d, exist_ok=True)

    def send_message(self, message, web=False):
        message_result = self.llm.chat(message, web_search=web)
        return message_result.wait_until_done()

    def stream_response(self, message, web=False, stream=False):
        responses = []
        for resp in self.llm.query(message, stream=stream, web_search=web):
            responses.append(resp['token'])
        return ' '.join(responses)

    def web_search(self, text):
        result = self.send_message(text, web=True)
        return result

    def retrieve_context(self, query: str):
        context = []
        lowres = self.retriever._get_relevant_documents(query)
        highres = st.session_state.compression_retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in lowres + highres])
        return context

    def get_conversation_chain(self):
        EMAIL = os.getenv("EMAIL")
        PASSWD = os.getenv("PASSWD")
        model = 1
        self.llm = LLMChatBot(EMAIL, PASSWD, default_llm=model)
        self.llm.create_new_conversation(system_prompt=self.llm.default_system_prompt, switch_to=True)

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain

    async def handle_user_input(self, user_input):
        response = st.session_state.conversation({'question': user_input})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"Human: {message.content}")
            else:
                st.write(f"AI: {message.content}")
                if st.session_state.voice_mode:
                    self.text_to_speech(message.content)

    def clone_github_repo(self, repo_url, local_path='./repo'):
        if os.path.exists(local_path):
            st.write("Repository already cloned.")
            return local_path
        Repo.clone_from(repo_url, local_path)
        return local_path
    def glob_recursive_multiple_extensions(base_dir, extensions):
        all_files = []
        for ext in extensions:
            pattern = os.path.join(base_dir, '**', f'*.{ext}')
            files = glob(pattern, recursive=True)
            all_files.extend(files)
        return all_files

    def load_documents_from_github(self, repo_url, file_types=['*.py', '*.md', '*.txt', '*.html']):
        local_repo_path = self.clone_github_repo(repo_url)
        globber=f"**/*/{{{','.join(file_types)}}}"
        rp(globber)
        loader = DirectoryLoader(path=local_repo_path, glob=globber, show_progress=True, recursive=True,loader_cls=TextLoader)
        loaded_files = loader.load()
        st.write(f"Nr. files loaded: {len(loaded_files)}")
        print(f"Loaded files: {len(loaded_files)}")  # Debug print

        # Convert the loaded files to Document objects
        documents = [Document(page_content=file_content) for file_content in loaded_files]
        print(f"Documents: {documents}")  # Debug print

        return documents
    
    def split_documents(self, documents, chunk_s=512, chunk_o=0):
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
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_s, chunk_overlap=chunk_o)
            split_docs.extend(splitter.split_documents([doc]))
        return split_docs, splitter

    def visualize_vectorstore(self):
        if st.session_state.vectorstore is None:
            st.write("Vectorstore is not initialized.")
            return

        documents = st.session_state.vectorstore.get_all_documents()
        embeddings = [doc.embedding for doc in documents]

        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)

        scaler = MinMaxScaler()
        embeddings_3d_normalized = scaler.fit_transform(embeddings_3d)

        colors = embeddings_3d_normalized[:, 0]

        hover_text = [f"Document {i}:<br>{doc.page_content[:100]}..." for i, doc in enumerate(documents)]

        fig = go.Figure(data=[go.Scatter3d(
            x=embeddings_3d_normalized[:, 0],
            y=embeddings_3d_normalized[:, 1],
            z=embeddings_3d_normalized[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
                colorscale='Viridis',
                opacity=0.8
            ),
            text=hover_text,
            hoverinfo='text'
        )])

        fig.update_layout(
            title="Interactive 3D Vectorstore Document Distribution",
            scene=dict(
                xaxis_title="PCA Component 1",
                yaxis_title="PCA Component 2",
                zaxis_title="PCA Component 3"
            ),
            width=800,
            height=600,
        )

        st.plotly_chart(fig)

    def chatbot_page(self):
        st.title("Chatbot")

        # Toggle for voice mode
        st.session_state.voice_mode = st.toggle("Voice Mode")

        # File uploader for context injection
        uploaded_file = st.file_uploader("Choose a file for context injection")
        if uploaded_file is not None:
            documents = [uploaded_file.read().decode()]
            st.session_state.vectorstore = self.create_vector_store(documents)
            st.session_state.conversation = self.get_conversation_chain()

        # GitHub repository URL input
        repo_url = st.text_input("Enter GitHub repository URL")
        if repo_url:
            documents = self.load_documents_from_github(repo_url)
            split_docs, _ = self.split_documents(documents)
            st.session_state.vectorstore = self.create_vector_store(split_docs)
            st.session_state.conversation = self.get_conversation_chain()

        # Chat interface
        user_input = st.text_input("You: ", key="user_input")

        if user_input:
            asyncio.run(self.handle_user_input(user_input))

        if st.session_state.voice_mode:
            if st.button("Speak"):
                user_speech = self.speech_to_text()
                st.text_input("You: ", value=user_speech, key="user_speech_input")
                if user_speech != "Sorry, I didn't catch that.":
                    asyncio.run(self.handle_user_input(user_speech))

    def dashboard_page(self):
        st.title("Dashboard")

        if st.session_state.vectorstore is not None:
            st.write("Vectorstore Visualization")
            self.visualize_vectorstore()
        else:
            st.write("Vectorstore is not initialized. Please add documents in the Chatbot page.")

    def main(self):
        st.set_page_config(page_title="Enhanced Multi-page Chatbot App", layout="wide")

        # Sidebar navigation
        with st.sidebar:
            selected = option_menu(
                menu_title="Navigation",
                options=["Chatbot", "Dashboard"],
                icons=["chat", "bar-chart"],
                menu_icon="cast",
                default_index=0,
            )

        if selected == "Chatbot":
            self.chatbot_page()
        elif selected == "Dashboard":
            self.dashboard_page()


if __name__ == "__main__":
    app = ChatbotApp(os.getenv("EMAIL"),os.getenv("PASSWD"))
    app.main()
