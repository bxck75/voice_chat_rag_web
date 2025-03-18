# document_processing.py
import os
from git import Repo
import os
import getpass

import faiss
import numpy as np
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import os
import tempfile
from datetime import datetime
import webbrowser
from tkinter import Toplevel
import warnings
import faiss,logging
import numpy as np
import wandb
from typing import List, Dict, Any, Optional, Union
from git import Repo
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
import requests
from rich import print as rp
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from dotenv import load_dotenv, find_dotenv
import speech_recognition
from TTS.api import TTS
from sklearn.decomposition import PCA
from playsound import playsound
from hugchat import hugchat
from hugchat.login import Login
from langchain_core.documents import Document
from langchain_community.llms.huggingface_text_gen_inference import (HuggingFaceTextGenInference)
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma,FAISS
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
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language,CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.llms.self_hosted_hugging_face import SelfHostedHuggingFaceLLM
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    PythonLoader
)
import plotly.graph_objs as go


from langchain.chains import LLMChain

# Load environment variables
load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")
os.environ['FAISS_NO_AVX2'] = '1'
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
wandb.require("core")
# Import system prompts
from system_prompts import __all__ as prompts


class DocumentProcessor:
    def __init__(self, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', llm: BaseLLM = None):
        self.loader = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=50, chunk_overlap=0)
        self.markdown_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0)
        self.html_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.HTML, chunk_size=60, chunk_overlap=0)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vectorstore = None
        self.retriever = None
        self.llm = llm
        self.persistant_dir = os.path.join('./vectorstore',"voice_hugchat")
        self.document_ids = {}  # Add this line to initialize document_ids

    def clone_github_repo(self, repo_url: str, local_path: str = './repo'):
        if os.path.exists(local_path):
            print("Repository already cloned.")
            return local_path
        Repo.clone_from(repo_url, local_path)
        return local_path

    def load_documents_from_github(self, repo_url: str, local_path: str = './repo', file_types: list = ['*.py', '*.md', '*.txt','*.html']):
        local_repo_path = self.clone_github_repo(repo_url, local_path)
        self.loader = DirectoryLoader(path=local_repo_path,  glob=f"**/{{{','.join(file_types)}}}", show_progress=True, recursive=True)
        return self.loader.load()
    
    def delete_document(self, document_name: str):
        if document_name in self.document_ids:
            doc_id = self.document_ids[document_name]
            self.vectorstore.delete(doc_id)
            del self.document_ids[document_name]
            print(f"Document '{document_name}' has been deleted.")
        else:
            print(f"Document '{document_name}' not found in the vector store.")

    def split_documents(self, documents: list):
        #return self.text_splitter.split_documents(documents)
        split_docs = []
        for doc in documents:
            ext = self._get_file_extension(doc)
            splitter = self._get_splitter_by_extension(ext)
            split_docs.extend(splitter.split_documents([doc]))
        return split_docs
    
    def _get_file_extension(self, document):
        # Assuming the document obself.persistant_dirame)[1].lower()
        print(ext)
        return ext
    
    def _get_splitter_by_extension(self, ext):
        if ext == '.py':
            return self.python_splitter
        elif ext in ['.md', '.markdown']:
            return self.markdown_splitter
        elif ext in ['.html', '.htm']:
            return self.html_splitter
        else:
            return self.text_splitter
        
    def load_vectorstore(self):
        self.vectorstore = FAISS.load(self.persistant_dir)
        
    def create_vectorstore(self, documents: list):
        self.vectorstore = FAISS.from_documents(documents, self.embeddings) 
        self.vectorstore.save(os.path.join(self.persistant_dir))
        # Store document names and their IDsself.embeddings
        for doc in documents:
            print(doc)
            doc_id = self.vectorstore._collection.count()  # Get the current count as the new ID
            self.document_ids[doc.metadata.get('source', f'doc_{doc_id}')] = doc_id
            self.vectorstore.add_documents([doc])  # Add each document individually to get its ID

    def pretty_print_docs(self, docs):
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
            )
        )

    
    def setup_retriever(self, k: int = 5):
        self.retriever = self.vectorstore.as_retriever(k=k)
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.76)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=self.retriever)
    
    def create_retrieval_chain(self, retrieval_qa_chat_prompt):
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, retrieval_qa_chat_prompt
        )
        self.retrieval_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

    def chat(self, input):
        return self.retrieval_chain.invoke({"input": input})  
       
    def retrieve_similar_documents(self, query: str,max_m=2,max_t=128,max_q=2):
        result = self.compression_retriever.invoke(
            query, max_docs=max_m, max_tokens=max_t, max_queries=max_q
        )
        self.pretty_print_docs(result)
        #result = self.retriever.invoke(query)
        return result

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    documents = processor.load_documents_from_github('https://github.com/bxck75/voice_chat_rag_web')
    split_docs = processor.split_documents(documents)
    
    processor.create_vectorstore(split_docs)
    processor.setup_retriever()
    query = "Enter your query here"
    results = processor.retrieve_similar_documents(query)
    for result in results:
        print(result)
