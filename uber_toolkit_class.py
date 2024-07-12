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
import speech_recognition
from git import Repo
from glob import glob
from rich import print as rp
from typing import Union, List, Generator, Any, Mapping, Optional,Dict
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

# Data manipulation and analysis
import numpy as np
import pandas as pd
# Plotting and visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
# Machine learning and dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
# Optional: for 3D projections
from scipy.stats import gaussian_kde
# Uncomment the following line if you need Plotly's built-in datasets
# import plotly.data as data


from huggingface_hub import InferenceClient
from hugchat import hugchat
from hugchat.login import Login
from hugchat.message import Message
from hugchat.types.assistant import Assistant
from hugchat.types.model import Model
from hugchat.types.message import MessageNode, Conversation

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from TTS.api import TTS
import time
from playsound import playsound
from system_prompts import __all__ as prompts

from profiler import VoiceProfileManager, VoiceProfile

# Example usage
manager = VoiceProfileManager("my_custom_profiles.json")
manager.load_profiles()

# Generate a random profile
new_profile = manager.generate_random_profile()
rp(f"Generated new profile: {new_profile.name}")

# List profiles
manager.list_profiles()

# Save profiles
manager.save_profiles()

load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")
class ChatBotWrapper:
    def __init__(self, chat_bot):
        self.chat_bot = chat_bot
    
    def __call__(self, *args, **kwargs):
        return self.chat_bot(*args, **kwargs)

class UberToolkit:
    def __init__(self, email, password, cookie_path_dir='./cookies/', default_llm=1):
        self.prompts = prompts

        # rp(self.prompts)
        self.email = os.getenv("EMAIL")
        self.password = os.getenv("PASSWD")
        self.default_llm = default_llm
        self.cookie_path_dir = cookie_path_dir
        self.system_prompt = self.prompts['default_rag_prompt']  # default_rag_prompt
        # rp(self.system_prompt)
        self.cookies = self.login()
        self.bot = hugchat.ChatBot(cookies=self.cookies.get_dict(), default_llm=self.default_llm)
        self.bot_wrapper = ChatBotWrapper(self.bot)  # Wrap the ChatBot object
    
        self.repo_url = ''    
        self.conv_id = None
        self.latest_splitter=None
        self.setup_folders()
        self.setup_embeddings()
        self.setup_vector_store()
        self.setup_retrievers()
        self.vector_store = None
        self.compressed_retriever = self.create_high_retrieval_chain()
        self.retriever = self.create_low_retrieval_chain()
        self.setup_tts()
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

    def setup_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )


    def setup_retrievers(self, k=5, similarity_threshold=0.76):
        self.retriever = self.vector_store.as_retriever(k=k)
        splitter = self.latest_splitter if self.latest_splitter else CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=self.retriever)

    def create_high_retrieval_chain(self):
        rag_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        rp(rag_prompt)
        combine_docs_chain = create_stuff_documents_chain(self.bot_wrapper, rag_prompt)
        return create_retrieval_chain(self.compression_retriever, combine_docs_chain)
        #self.low_retrieval_chain = create_retrieval_chain(self.retriever, combine_docs_chain)
    
    def create_low_retrieval_chain(self):  
        rag_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(self.bot_wrapper, rag_prompt)
        #return create_retrieval_chain(self.compression_retriever, combine_docs_chain)
        return create_retrieval_chain(self.retriever, combine_docs_chain)

    def setup_tts(self, model_name="tts_models/en/ljspeech/fast_pitch"):
        self.tts = TTS(model_name=model_name,progress_bar=False, vocoder_path='vocoder_models/en/ljspeech/univnet')

    def setup_speech_recognition(self):
        self.recognizer = speech_recognition.Recognizer()
    
    def setup_folders(self):
        self.dirs=["test_input","vectorstore","test"]
        for d in self.dirs:
            os.makedirs(d, exist_ok=True)

    def __call__(self, text):
        if self.conv_id:
            self.bot.change_conversation(self.bot.get_conversation_from_id(self.conv_id))
        else:
            self.conv_id = self.bot.new_conversation(system_prompt=self.system_prompt, modelIndex=self.default_llm, switch_to=True)
        return self.send_message(text)

    def send_message(self, message, web=False):
        message_result = self.bot.chat(message, web_search=web)
        return message_result.wait_until_done()

    def stream_response(self, message, web=False, stream=False):
        responses = []
        for resp in self.bot.query(message, stream=stream, web_search=web):
            responses.append(resp['token'])
        return ' '.join(responses)

    def web_search(self, text):
        result = self.send_message(text, web=True)
        return result

    def retrieve_context(self, query: str):
        context=[]
        return context
        try:
            lowres = self.retriever.invoke({'input': query})
            vector_context = "\n".join(lowres) if lowres else "No Context Available!"
        except Exception as e:
            vector_context = f"Error retrieving context: {str(e)}"
        context.append(vector_context)
        try:
            highres=self.compression_retriever.invoke({'input':query})
            vector_context = "\n".join(highres) if highres else "No Context Available!"
        except Exception as e:
            vector_context = f"Error retrieving context: {str(e)}"
        context.append(vector_context)
        
        context = "\n".join([doc.page_content for doc in context])
        rp(f"CONTEXT:{context}")
        return context

    def delete_all_conversations(self):
        self.bot.delete_all_conversations()

    def delete_conversation(self, conversation_object: Conversation = None):
        self.bot.delete_conversation(conversation_object)

    def get_available_llm_models(self) -> list:
        return self.bot.get_available_llm_models()

    def get_remote_conversations(self, replace_conversation_list=True):
        return self.bot.get_remote_conversations(replace_conversation_list)

    def get_conversation_info(self, conversation: Union[Conversation, str] = None) -> Conversation:
        return self.bot.get_conversation_info(conversation)

    def get_assistant_list_by_page(self, page: int) -> List[Assistant]:
        return self.bot.get_assistant_list_by_page(page)

    def search_assistant(self, assistant_name: str = None, assistant_id: str = None) -> Assistant:
        return self.bot.search_assistant(assistant_name, assistant_id)

    def switch_model(self, index):
        self.conv_id = None
        self.default_llm = index

    def switch_conversation(self, id):
        self.conv_id = id

    def switch_role(self, system_prompt_id):
        self.system_prompt = system_prompt_id

    def chat(self, text: str, web_search: bool = False, _stream_yield_all: bool = False, retry_count: int = 5, conversation: Conversation = None, *args, **kwargs) -> Message:
        return self.bot.chat(text, web_search, _stream_yield_all, retry_count, conversation, *args, **kwargs)
    
    def get_all_documents(self) -> List[Document]:
        """
        Retrieve all documents from the vectorstore.
        """
        if not self.vector_store:
            self.setup_vector_store()
        
        all_docs_query = "* *"  # This is a common wildcard query, but may need adjustment based on your specific setup

        # Use the base retriever to get all documents
        # Set a high limit to ensure we get all documents
        all_docs = self.retriever.get_relevant_documents(all_docs_query, k=10000)  # Adjust the k value if needed
        return all_docs
    
    def generate_3d_scatterplot(self, num_points=1000):
        """
        Generate a 3D scatter plot of the vector store content.
        
        :param num_points: Maximum number of points to plot (default: 1000)
        :return: None (displays the plot)
        """
        import plotly.graph_objects as go
        import numpy as np
        from sklearn.decomposition import PCA

        # Get all documents using the get_all_documents method
        all_docs = self.get_all_documents()

        if not all_docs:
            raise ValueError("No documents found in the vector store.")

        # Extract vectors from documents
        vectors = []
        for doc in all_docs:
            # Assuming each document has a vector attribute or method to get its vector
            # You might need to adjust this based on your Document structure
            if hasattr(doc, 'embedding') and doc.embedding is not None:
                vectors.append(doc.embedding)
            else:
                # If the document doesn't have an embedding, we'll need to create one
                vectors.append(self.embeddings.embed_query(doc.page_content))

        vectors = np.array(vectors)

        # If we have more vectors than requested points, sample randomly
        if len(vectors) > num_points:
            indices = np.random.choice(len(vectors), num_points, replace=False)
            vectors = vectors[indices]

        # Perform PCA to reduce to 3 dimensions
        pca = PCA(n_components=3)
        vectors_3d = pca.fit_transform(vectors)

        # Create the 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=vectors_3d[:, 0],
            y=vectors_3d[:, 1],
            z=vectors_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=vectors_3d[:, 2],  # Color by z-dimension
                colorscale='Viridis',
                opacity=0.8
            )
        )])

        # Update layout
        fig.update_layout(
            title='3D Scatter Plot of Vector Store Content',
            scene=dict(
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                zaxis_title='PCA Component 3'
            ),
            width=900,
            height=700,
        )

        # Show the plot
        fig.show()

        print(f"Generated 3D scatter plot with {len(vectors)} points.")

    def listen_for_speech(self):
        with speech_recognition.Microphone() as source:
            rp("Listening...")
            audio = self.recognizer.listen(source)
            
        try:
            text = self.recognizer.recognize_google(audio)
            rp(f"You said: {text}")
            return text
        except speech_recognition.UnknownValueError:
            rp("Sorry, I couldn't understand that.")
            return None
        except speech_recognition.RequestError as e:
            rp(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    def optimized_tts(self, text: str, output_file: str = "output.wav", speaking_rate: float = 5) -> str:
        start_time = time.time()
        rp(f"Starting TTS at {start_time}")
        try:
            self.tts.tts_to_file(
                text=text,
                file_path=output_file,
                speaker=self.tts.speakers[0] if self.tts.speakers else None,
                language=self.tts.languages[0] if self.tts.languages else None,
                speed=speaking_rate,
                split_sentences=True
            )
            end_time = time.time()
            rp(f"TTS generation took {end_time - start_time:.2f} seconds")

        except RuntimeError as e:
            if "Kernel size can't be greater than actual input" in str(e):
                rp(f"Text too short for TTS: {text}")
            else:
                raise  # Re-raise if it's a different RuntimeError
        
        return output_file

    @staticmethod
    def play_mp3(file_path):
        playsound(file_path)

    def continuous_voice_chat(self):
        self.input_method = None
        while True:
            rp("Speak your query (or say 'exit' to quit):")
            self.input_method = self.listen_for_speech()
            self.voice_chat_exit = False
            query = self.input_method
            
            if query is None:
                continue

            """ if 'switch prompt ' in query.lower():
                q = query.lower()
                new_prompt = q.split("switch prompt ").pop().replace(" ", "_")
                #rp(new_prompt)
                if new_prompt in self.prompts.keys():
                    self.system_prompt = self.prompts[new_prompt]
                    rp(f"new system prompt:{self.system_prompt}")
                
                
                #self.switch_role(new_prompt_id)
                self.optimized_tts(f"Switched Role to {new_prompt}!")
                self.play_mp3('output.wav')
                continue """

            if query.lower() == "voice":
                rp("Speak your query (or say 'exit' to quit):")
                self.input_method = self.listen_for_speech()
                continue
            
            if query.lower() == "type":
                self.input_method = input("Type your question(or type 'exit' to quit): \n")
                continue
            
            if query.lower() == 'exit':
                rp("Goodbye!")
                self.optimized_tts("Ok, exiting!")
                self.play_mp3('output.wav')
                self.voice_chat_exit = True
                break
            
            result = self.web_search(query)
            web_context = "\n".join(result) if result else "No Context Available from the websearch!"
            #vector_context = self.retrieve_context(query)
            
            #self.system_prompt = self.system_prompt.replace("<<VSCONTEXT>>", vector_context if vector_context else "No Context Available in the vectorstore!")
            self.system_prompt = self.system_prompt.replace("<<WSCONTEXT>>", web_context)
            
            response = self.bot.chat(query)
            
            if "/Store:" in response:
                url = response.split("/Store:").pop().split(" ")[0]
                rp(f"Fetching and storing data from link: {url}")
                try:
                    self.add_document_from_url(url)
                except Exception as e:
                    rp(f"Error while fetching data from {url}! {e}")
                continue
            
            if "/Delete:" in response:
                document = response.split("/Delete:").pop().split(" ")[0]
                rp(f"Deleting {document} from vectorstore!")
                try:
                    self.delete_document(document)
                except Exception as e:
                    rp(f"Error while deleting {document} from vectorstore! {e}")

            rp(f"Chatbot: {response}")
            
            self.play_mp3(self.optimized_tts(str(response)))


    def initialize_vector_store(
        self,
        initial_docs: Union[List[Union[str, Document]], str],
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "faiss_index",
        index_name: str = "document_store"
    ) -> FAISS:
        """
        Initialize a FAISS vector store. If a persistent store exists, load and update it.
        Otherwise, create a new one from the initial documents.

        Args:
        initial_docs (Union[List[Union[str, Document]], str]): Initial documents to add if creating a new store.
        embedding_model_name (str): Name of the HuggingFace embedding model to use.
        persist_directory (str): Directory to save/load the persistent vector store.
        index_name (str): Name of the index file.

        Returns:
        FAISS: The initialized or loaded FAISS vector store.
        """
        allow_dangerous_deserialization=True
        index_file_path = os.path.join(persist_directory, f"{index_name}.faiss")
        
        # Convert initial_docs to a list of Document objects
        if isinstance(initial_docs, str):
            initial_docs = [Document(page_content=initial_docs)]
        elif isinstance(initial_docs, list):
            initial_docs = [
                doc if isinstance(doc, Document) else Document(page_content=doc)
                for doc in initial_docs
            ]
        
        if os.path.exists(index_file_path):
            print(f"Loading existing vector store from {index_file_path}")
            vector_store = FAISS.load_local(
                persist_directory, 
                self.embeddings, 
                index_name,
                allow_dangerous_deserialization=allow_dangerous_deserialization
                )
            
            # Update with new documents if any
            if initial_docs:
                print(f"Updating vector store with {len(initial_docs)} new documents")
                vector_store.add_documents(initial_docs)
                vector_store.save_local(persist_directory, index_name)
        else:
            print(f"Creating new vector store with {len(initial_docs)} documents")
            vector_store = FAISS.from_documents(initial_docs, self.embeddings)
            
            # Ensure the directory exists
            os.makedirs(persist_directory, exist_ok=True)
            vector_store.save_local(persist_directory, index_name)
        
        return vector_store
    
    def setup_vector_store(self):   
        from langchain.docstore import InMemoryDocstore        
        embedding_size = 384  # Size for all-MiniLM-L6-v2 embeddings
        index = faiss.IndexFlatL2(embedding_size)
        docstore = InMemoryDocstore({})
        
        self.vector_store = FAISS(
            self.embeddings, 
            index, 
            docstore, 
            {}
        )

    """     def setup_vector_store(self):
            self.vector_store = self.initialize_vector_store(['this your Birth, Rise and Shine a mighty bot']) 

    """
    def add_documents_folder(self, folder_path):
        paths=[]
        for root, _, files in os.walk(folder_path):
            for file in files:
                paths.append(os.path.join(root, file))
        
        self.add_documents(paths)

    def fetch_document(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return Document(page_content=content)
            #self.vector_store.add_documents([document])
    
    def add_documents(self, documents: List[str]):
        docs_to_add=[]
        if not self.vector_store:
            self.setup_vector_store()
        for document in documents:
            docs_to_add.append(self.fetch_document(document))

        self.vector_store.add_documents(docs_to_add)
        
        # Print the added documents for verification
        for i in range(len(docs_to_add)):
            doc_id = self.vector_store.index_to_docstore_id[i]
            rp(f"Added document {i}: {self.vector_store.docstore._dict[doc_id]}")

    def add_document_from_url(self, url):
        if not self.vector_store:
            self.setup_vector_store()
        response = requests.get(url)
        if response.status_code == 200:
            content = response.text
            document = Document(page_content=content)
            self.vector_store.add_documents([document])
        else:
            rp(f"Failed to fetch URL content: {response.status_code}")

    def delete_document(self, document):
        if document in self.vector_store:
            self.vector_store.delete_document(document)
            rp(f"Deleted document: {document}")
        else:
            rp(f"Document not found: {document}")

    def _add_to_vector_store(self, name, content):
        document = Document(page_content=content)
        self.vector_store.add_documents([document])
        rp(f"Added document to vector store: {name}")
        # Example of updating the vectorizer (you might need to adjust based on your actual implementation)
        self.vectorizer.fit_transform(self.compressed_retriever.invoke("*"))
    
    def clone_github_repo(self, repo_url, local_path='./repo'):
        if os.path.exists(local_path):
            rp("Repository already cloned.") 
            return local_path
        Repo.clone_from(repo_url, local_path)
        return local_path
    
    def load_documents(self, repo_url, file_types=['*.py', '*.md', '*.txt', '*.html']):
        local_repo_path = self.clone_github_repo(repo_url)
        loader = DirectoryLoader(path=local_repo_path, glob=f"**/{{{','.join(file_types)}}}", show_progress=True, recursive=True)
        loaded=loader.load()
        rp(f"Nr. files loaded: {len(loaded)}")
        return loaded
    
    def recursive_glob(self,root_dir, patterns):
        import fnmatch
        """Recursively search for files matching the patterns in root_dir.

        Args:
            root_dir (str): The root directory to start the search from.
            patterns (list): List of file patterns to search for, e.g., ['*.py', '*.md'].

        Returns:
            list: List of paths to the files matching the patterns.
        """
        matched_files = []
        for root, dirs, files in os.walk(root_dir):
            for pattern in patterns:
                for filename in fnmatch.filter(files, pattern):
                    matched_files.append(os.path.join(root, filename))
        return matched_files


    def load_documents_from_github(self, repo_url, file_types=['*.py', '*.md', '*.txt', '*.html']):
        local_repo_path = self.clone_github_repo(repo_url)
        document_paths = self.recursive_glob(local_repo_path, file_types)
        rp(f"Found {len(document_paths)} documents")
        self.add_documents(document_paths)
        """ loader = DirectoryLoader(path=local_repo_path, glob=f"**/{{{','.join(file_types)}}}", show_progress=True, recursive=True)
        loaded=loader.load(document_paths)
        rp(f"Nr. files loaded: {len(loaded)}")
        return loaded """

    def split_documents(self, documents: list,chunk_s=512,chunk_o=0):
        split_docs = []
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
    

    def save_vectorstore_local(self, folder_path: str="vectorstore", index_name: str = "faiss_index"):
        """
        Save the FAISS vectorstore locally with all necessary components.

        Args:
            folder_path (str): Folder path to save index, docstore, and index_to_docstore_id to.
            index_name (str): Name for the saved index file (default is "faiss_index").
        """

        # Get all documents from the vectorstore
        documents = self.compressed_retriever.invoke("*")<--error

        # Create a new docstore and index_to_docstore_id mapping
        docstore: Dict[str, Document] = {}
        index_to_docstore_id: Dict[int, str] = {}

        for i, doc in enumerate(documents):
            # Generate a unique ID for each document
            doc_id = str(uuid4())
            docstore[doc_id] = doc
            index_to_docstore_id[i] = doc_id

        # Save the FAISS index
        self.vector_store.save_local(folder_path, index_name)

        # Save the docstore
        import pickle
        with open(os.path.join(folder_path, f"{index_name}_docstore.pkl"), "wb") as f:
            pickle.dump(docstore, f)

        # Save the index_to_docstore_id mapping
        with open(os.path.join(folder_path, f"{index_name}_index_to_docstore_id.pkl"), "wb") as f:
            pickle.dump(index_to_docstore_id, f)

        rp(f"Vectorstore saved successfully to {folder_path}")
        return folder_path
    

    @classmethod
    def load_vectorstore_local(cls, folder_path: str, index_name: str = "faiss_index", embeddings=None):
        """
        Load a previously saved FAISS vectorstore.
        Args:
            folder_path (str): Folder path where the index, docstore, and index_to_docstore_id are saved.
            index_name (str): Name of the saved index file (default is "faiss_index").
            embeddings: The embeddings object to use (must be the same type used when saving).
        Returns:
            FAISS: Loaded FAISS vectorstore
        """
        # Ensure you trust the source of the pickle file before setting this to True
        allow_dangerous_deserialization = True

        # Load the docstore
        with open(os.path.join(folder_path, f"{index_name}_docstore.pkl"), "rb") as f:
            docstore = pickle.load(f)
        # Load the index_to_docstore_id mapping
        with open(os.path.join(folder_path, f"{index_name}_index_to_docstore_id.pkl"), "rb") as f:
            index_to_docstore_id = pickle.load(f)

        # Load the FAISS index
        vectorstore = FAISS.load_local(
            folder_path, 
            embeddings, 
            index_name,
            allow_dangerous_deserialization=allow_dangerous_deserialization
            )
        # Reconstruct the FAISS object with the loaded components
        vectorstore.docstore = docstore
        vectorstore.index_to_docstore_id = index_to_docstore_id

        return vectorstore
    
    def create_vectorstore_from_github(self):
        documents = self.load_documents_from_github(self.repo_url)
        split_docs,splitter = self.split_documents(documents,512,0)
        self.latest_splitter=splitter
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        self.vector_store.save_local()
        rp(f"Vectorstore created with {len(split_docs)} documents.")

    def update_vectorstore(self, new_documents):
        split_docs,splitter = self.split_documents(new_documents)
        self.latest_splitter=splitter
        self.vector_store.add_documents(split_docs)
        rp(f"Vectorstore updated with {len(split_docs)} new documents.")
    

    def retrieve_with_chain(self, query, mode='high'):
        if mode == 'high':
            return self.compressed_retriever.invoke({"input": query})
        else:
            return self.retriever.invoke({"input": query})

    def generate_code(self, prompt):
        self.system_prompt=self.prompts["code_generator_prompt"]
        return self.send_message(prompt)

    def debug_script(self, script):
        self.system_prompt = self.prompts["script_debugger_prompt"]
        return self.send_message(f"Debug the following script:\n\n{script}")

    def test_software(self, software_description):
        self.system_prompt = self.prompts["software_tester_prompt"]
        return self.send_message(f"Create a test plan for the following software:\n\n{software_description}")

    def parse_todo(self, todo_list):
        self.system_prompt = self.prompts["todo_parser_prompt"]
        return self.send_message(f"Parse and organize the following TODO list:\n\n{todo_list}")

    def tell_story(self, prompt):
        self.system_prompt = self.prompts["story_teller_prompt"]
        return self.stream_response(f"Tell a story based on this prompt:\n\n{prompt}")

    def act_as_copilot(self, task):
        self.system_prompt = self.prompts["copilot_prompt"]
        return self.send_message(f"Assist me as a copilot for the following task:\n\n{task}")

    def control_iterations(self, task, max_iterations=5):
        self.system_prompt = self.prompts["iteration_controller_prompt"]
        iteration = 0
        result = ""
        while iteration < max_iterations:
            response = self.send_message(f"Iteration {iteration + 1} for task:\n\n{task}\n\nCurrent result:\n{result}")
            result += f"\nIteration {iteration + 1}:\n{response}"
            if "TASK_COMPLETE" in response:
                break
            iteration += 1
        return result

    def voice_command_mode(self):
        rp("Entering voice command mode. Speak your commands.")
        while True:
            command = self.listen_for_speech()
            if command is None:
                continue
            if command.lower() == "exit voice mode":
                rp("Exiting voice command mode.")
                break
            response = self.process_voice_command(command)
            rp(f"Assistant: {response}")
            self.optimized_tts(response)
            self.play_mp3('output.wav')

    def process_voice_command(self, command):
        if "generate code" in command.lower():
            return self.generate_code(command)
        elif "debug script" in command.lower():
            return self.debug_script(command)
        elif "test software" in command.lower():
            return self.test_software(command)
        elif "parse todo" in command.lower():
            return self.parse_todo(command)
        elif "tell story" in command.lower():
            return self.tell_story(command)
        elif "act as copilot" in command.lower():
            return self.act_as_copilot(command)
        else:
            return self.send_message(command)

    def interactive_mode(self):
        rp("Entering interactive mode. Type 'exit' to quit, 'voice' for voice input, or 'command' for specific functions.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                rp("Exiting interactive mode.")
                break
            elif user_input.lower() == 'voice':
                self.voice_command_mode()
            elif user_input.lower() == 'command':
                self.command_mode()
            else:
                response = self.send_message(user_input)
                rp(f"Assistant: {response}")

    def command_mode(self):
        rp("Entering command mode. Available commands: generate_code, debug_script, test_software, parse_todo, tell_story, copilot, iterate")
        while True:
            command = input("Enter command (or 'exit' to return to interactive mode): ")
            if command.lower() == 'exit':
                rp("Exiting command mode.")
                break
            self.execute_command(command)

    def execute_command(self, command):
        if command == "add_to_vectorstore":
            prompt = input("Enter list of files, folders, urls or repos with knowledge to add:")
            response = self.generate_code(prompt)
        if command == "generate_code":
            file_name = input("Enter script filename:")
            prompt = input("Enter code generation prompt:")
            response = self.generate_code(prompt)
        elif command == "debug_script":
            script = input("Enter script to debug:")
            response = self.debug_script(script)
        elif command == "test_script":
            description = input("Enter path to script:")
            response = self.test_software(description)
        elif command == "parse_todo":
            todo_list = input("Enter TODO list:")
            response = self.parse_todo(todo_list)
        elif command == "tell_story":
            prompt = input("Enter story prompt:")
            response = self.tell_story(prompt)
        elif command == "copilot":
            task = input("Enter task for copilot:")
            response = self.act_as_copilot(task)
        elif command == "iterate":
            task = input("Enter task for iteration:")
            max_iterations = int(input("Enter maximum number of iterations: "))
            response = self.control_iterations(task, max_iterations)
        else:
            response = "Unknown command. Please try again."
        
        rp(f"Assistant: {response}")

    def run(self):
        rp("Welcome to the Advanced AI Toolkit!")
        rp("Choose a mode to start:")
        rp("1. Interactive Chat")
        rp("2. Voice Chat")
        rp("3. Command Mode")
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            self.interactive_mode()
        elif choice == '2':
            self.continuous_voice_chat()
        elif choice == '3':
            self.command_mode()
        else:
            rp("Invalid choice. Exiting.")

if __name__ == "__main__":
    email = os.getenv("EMAIL")
    password = os.getenv("PASSWD")
    toolkit = UberToolkit(email, password)
    toolkit.run()

    