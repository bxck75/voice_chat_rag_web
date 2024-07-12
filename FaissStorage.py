import os
import tempfile
from datetime import datetime
import webbrowser
from tkinter import Toplevel
import warnings
import faiss,logging
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
from playsound import playsound
from hugchat import hugchat
from hugchat.login import Login
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    PythonLoader
)
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

# Load environment variables
load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")

# Import system prompts
from system_prompts import __all__ as prompts


class LLMChatBot:
    def __init__(self, email, password, cookie_path_dir='./cookies/', default_llm=1,default_system_prompt='default_rag_prompt'):
        self.email = email
        self.password = password
        self.current_model = 1
        self.current_system_prompt=default_system_prompt
        self.cookie_path_dir = cookie_path_dir
        self.cookies = self.login()
        self.chatbot = hugchat.ChatBot(cookies=self.cookies.get_dict(), default_llm=default_llm,system_prompt=prompts[default_system_prompt])
        self.conversation_id=None
        self.check_conv_id(self.conversation_id)
        rp("[self.conversation_id:{self.conversation_id}]")
    
    def check_conv_id(self, id=None):
        if not self.conversation_id and not id:
            self.conversation_id = self.chatbot.new_conversation(modelIndex=self.current_model,system_prompt=self.current_system_prompt)
        else:
            if id:
                self.conversation_id=id
                self.chatbot.change_conversation(self.conversation_id)
            elif not self.chatbot.get_conversation_info(self.conversation_id) == self.chatbot.get_conversation_info():
                self.chatbot.change_conversation(self.conversation_id)
        
        return self.conversation_id        

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
        self.dirs = ["test_input"]
        for d in self.dirs:
            os.makedirs(d, exist_ok=True)

    def setup_tts(self, model_name="tts_models/en/ljspeech/fast_pitch"):
        self.tts = TTS(model_name=model_name)

    def chat(self, message):
        return self.chatbot.chat(message)
    
    def query(self,message):
        return self.chatbot.query(
            text=message,
            web_search = False,
            temperature = 0.1,
            top_p = 0.95,
            repetition_penalty = 1.2,
            top_k = 50,
            truncate = 1000,
            watermark = False,
            max_new_tokens = 1024,
            stop = ["</s>"],
            return_full_text = False,
            stream = False,
            _stream_yield_all = False,
            use_cache = False,
            is_retry = False,
            retry_count = 5,
            conversation = None
        )
    def __run__(self, message):

        return self.query(message)
    
    def __call__(self, message,system_prompt,model,):
        if not self.conversation_id:
            self.conversation_id = self.chatbot.new_conversation(modelIndex=self.current_model,system_prompt=self.current_system_prompt,switch_to=True)
        return self.chat(message)

class AdvancedVectorStore:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2", 
                 email: str = None, 
                 password: str = None,
                 chunk_size=384,
                 chunk_overlap=0, 
                 device='cpu',
                 normalize_embeddings=True,
                 log_level=logging.INFO, 
                 log_file='llm_chatbot.log'):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device
        self.basic_splitter= RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.storage_path='./vectorstore'
        self.documents: List[Document] = []
        self.llm_chatbot = LLMChatBot(email, password) if email and password else None
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': normalize_embeddings}
        )
        self.vectorstore, self.docstore, self.index = self.create_indexed_vectorstore(self.chunk_size)
        self.document_count = 0
        self.chunk_count = 0        
        self.setup_logging(log_level,log_file)
        self.logger.info("Initializing AdvancedVectorStore")
        self.set_bot_role()
    
    def setup_logging(self,level,file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        # Create console handler and set level
        ch = logging.StreamHandler()
        ch.setLevel(level)
        # Create file handler and set level
        fh = logging.FileHandler(file)
        fh.setLevel(level)
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Add formatter to console handler
        ch.setFormatter(formatter)
        # Add formatter to file handler
        fh.setFormatter(formatter)
        # Add handlers to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        self.logger.info("Done setting up logger for {__name__} [AdvancedVectorStore]")
        

    def set_bot_role(self,prompt='default_rag_prompt',context="",history=""):
        self.logger.info(f"Setting Bot Role!\n[{prompts[prompt]}]")
        self.llm_chatbot.current_system_prompt = prompts[prompt].replace("<<VSCONTEXT>>",context).replace("<<WSCONTEXT>>",history)
        result=self.llm_chatbot("Confirm you understand ACT and ROLE and TASK",
                                system_prompt=self.llm_chatbot.current_system_prompt,
                                model=1)
        self.logger.info(f"Test results chatbot role set:{result}")
        

    def load_documents(self, directory: str) -> None:
        """Load documents from a directory with specific loaders for each file type."""
        loaders = {
            ".py": (PythonLoader, {}),
            ".txt": (TextLoader, {}),
            ".pdf": (PyPDFLoader, {}),
            ".html": (UnstructuredHTMLLoader, {}),
            ".docx": (UnstructuredWordDocumentLoader, {})
        }

        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file)[1].lower()

                if file_extension in loaders:
                    # Check if the file can be read as UTF-8
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            f.read()
                    except (UnicodeDecodeError, IOError):
                        rp(f"Skipping non-UTF-8 or unreadable file: {file_path}")
                        continue

                    loader_class, loader_args = loaders[file_extension]
                    loader = loader_class(file_path, **loader_args)
                    self.documents.extend(loader.load())

    def split_documents(self) -> None:
        """Split documents using appropriate splitters for each file type."""
        splitters = {
            ".py": RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=2000, chunk_overlap=200),
            ".txt": RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200),
            ".pdf": RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200),
            ".html": RecursiveCharacterTextSplitter.from_language(language=Language.HTML, chunk_size=2000, chunk_overlap=200),
            ".docx": RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        }

        split_docs = []
        for doc in self.documents:
            file_extension = os.path.splitext(doc.metadata.get("source", ""))[1].lower()
            splitter = splitters.get(file_extension, RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200))
            split_docs.extend(splitter.split_documents([doc]))

        self.documents = split_docs

    def create_vectorstore(self, store_type: str = "FAISS") -> None:
        """Create a vectorstore of the specified type."""
        if store_type == "FAISS":
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        else:
            raise ValueError(f"Unsupported vectorstore type: {store_type}")
    
    def create_indexed_vectorstore(self,embedding_size):
        rp("Creating indexed vectorstore...")
        #embedding_size = 384  # Size for all-MiniLM-L6-v2 embeddings
        index = faiss.IndexFlatL2(embedding_size)
        docstore = InMemoryDocstore({})
        vectorstore = FAISS(
            self.embeddings.embed_query, 
            index, 
            docstore, 
            {}
        )
        rp("Indexed vectorstore created.")
        return vectorstore,docstore,index


    def get_basic_retriever(self, k: int = 4) -> VectorStore:
        """Get a basic retriever from the vectorstore."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore() first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_multi_query_retriever(self, k: int = 4) -> MultiQueryRetriever:
        """Get a MultiQueryRetriever."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore() first.")
        return MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            llm=self.llm_chatbot
        )

    def get_self_query_retriever(self, k: int = 4) -> SelfQueryRetriever:
        """Get a SelfQueryRetriever."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore() first.")
        return SelfQueryRetriever.from_llm(
            self.llm_chatbot.chatbot,
            self.vectorstore,
            document_contents="Document about various topics.",
            metadata_field_info=[],
            search_kwargs={"k": k}
        )
    def get_multi_query_compression_retriever(self, k=5, similarity_threshold=0.76):
        """Get a ContextualCompressionRetriever."""
        retriever = self.get_multi_query_retriever(k=k)
        splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        return ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

    def get_contextual_compression_retriever(self, k: int = 4,similarity_threshold=0.78) -> ContextualCompressionRetriever:
        """Get a ContextualCompressionRetriever."""
        base_compressor = LLMChainExtractor.from_llm(self.llm_chatbot)
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings, similarity_threshold=similarity_threshold)
        relevant_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[self.basic_splitter, base_compressor, redundant_filter, relevant_filter]
        )
        
        return ContextualCompressionRetriever(
            name="CompressedRetriever",
            base_compressor=pipeline_compressor,
            base_retriever=self.get_basic_retriever(k=k)
        )
    
    def set_current_retriever(self,mode='basic',k=4,sim_rate=0.78):
        if mode == 'compressed':
            retriever = self.get_contextual_compression_retriever(k, sim_rate)
        elif mode == 'self_query':
            retriever = self.get_self_query_retriever(k)
        elif mode == 'multi_query':
            retriever = self.get_multi_query_retriever(k)
        else:
            retriever = self.get_basic_retriever(k)

        rp(retriever.get_prompts)
        return retriever

    def search(self, query: str, mode='basic', retriever: Optional[Any] = None, k: int = 4, sim_rate: float = 0.78) -> List[Document]:
        """Search the vectorstore using the specified retriever."""
        if not retriever:
            retriever = self.set_current_retriever(mode=mode, k=k, sim_rate=sim_rate)

        return retriever.get_relevant_documents(query)

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the existing vectorstore."""

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("[cyan]Adding documents to vectorstore...", total=len(documents))
            
            for doc in documents:
                self.vectorstore.add_documents([doc])
                progress.update(task, advance=1)

        rp(f"Added {len(documents)} documents to the vectorstore.")

    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vectorstore by their IDs."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore() first.")
        self.vectorstore.delete(document_ids)

    def save_vectorstore(self, path: str) -> None:
        """Save the vectorstore to disk."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore() first.")
        self.vectorstore.save_local(path)

    def load_vectorstore(self, path: str) -> None:
        """Load the vectorstore from disk."""
        self.vectorstore = FAISS.load_local(path, self.embeddings)

    def create_retrieval_chain(self, prompt: str = "default_rag_prompt", retriever: Optional[Any] = None) -> Any:
        """Create a retrieval chain using the specified prompt and retriever."""
        if not retriever:
            retriever = self.get_basic_retriever()
        
        combine_docs_chain = create_stuff_documents_chain(self.llm_chatbot.chatbot, prompt=prompts[prompt])
        return create_retrieval_chain(retriever, combine_docs_chain)

    def run_retrieval_chain(self, chain: Any, query: str) -> Dict[str, Any]:
        """Run a retrieval chain with the given query."""
        return chain.invoke({"input": query})

    def load_documents_folder(self, folder_path):
            rp("Loading documents from cloned repository")
            self.load_documents(folder_path)

            rp("Splitting documents")
            self.split_documents()
            rp("Adding documents to vectorstore")
            self.add_documents(self.documents)

    def load_github_repo(self, repo_url: str) -> None:
        """
        Clone a GitHub repository to a temporary folder, load documents, and remove the folder.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            rp(f"Cloning repository {repo_url} to {temp_dir}")
            Repo.clone_from(repo_url, temp_dir)
            
            rp("Loading documents from cloned repository")
            self.load_documents(temp_dir)
            
            rp("Splitting documents")
            self.split_documents()
            
            rp("Adding documents to vectorstore")
            self.add_documents(self.documents)
            
        rp("Temporary folder removed")
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents from the vectorstore.
        
        :return: List of all documents in the vectorstore
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore() first.")
        
        # This assumes FAISS vectorstore has a method to retrieve all documents
        # You might need to adjust this based on the actual implementation
        all=self.vectorstore.similarity_search(query='* *',k=10000)
        return all

    def generate_3d_scatterplot(self, num_points=1000):
        """
        Generate a 3D scatter plot of the vector store content.
        
        :param num_points: Maximum number of points to plot (default: 1000)
        :return: None (displays the plot)
        """
        all_docs = self.get_all_documents()

        if not all_docs:
            raise ValueError("No documents found in the vector store.")

        # Extract vectors from documents
        vectors = []
        for doc in all_docs:
            if hasattr(doc, 'embedding') and doc.embedding is not None:
                vectors.append(doc.embedding)
            else:
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

        rp(f"Generated 3D scatter plot with {len(vectors)} points.")
    def test_chat(self,text,context='This is a chat with a nice Senior programmer.',history='Your Birth as fresh outof the box agent.'):
        
        self.set_bot_role(context=context,history=history)
        
        return self.llm_chatbot(text)
    def chat(self, message: str) -> str:
        """
        Send a message to the HugChat bot and get a response.
        
        :param message: The message to send to the bot
        :return: The bot's response
        """
        if not self.llm_chatbot:
            raise ValueError("HugChat bot not initialized. Provide email and password when creating AdvancedVectorStore.")
        return self.llm_chatbot.chat(message)

    def setup_speech_recognition(self):
        """Set up speech recognition for the HugChat bot."""
        if not self.llm_chatbot:
            raise ValueError("HugChat bot not initialized. Provide email and password when creating AdvancedVectorStore.")
        self.llm_chatbot.setup_speech_recognition()

    def setup_tts(self, model_name="tts_models/en/ljspeech/fast_pitch"):
        """Set up text-to-speech for the HugChat bot."""
        if not self.llm_chatbot:
            raise ValueError("HugChat bot not initialized. Provide email and password when creating AdvancedVectorStore.")
        self.llm_chatbot.setup_tts(model_name)

    def voice_chat(self):
        """
        Initiate a voice chat session with the HugChat bot.
        """
        if not self.llm_chatbot or not hasattr(self.llm_chatbot, 'recognizer') or not hasattr(self.llm_chatbot, 'tts'):
            raise ValueError("Speech recognition and TTS not set up. Call setup_speech_recognition() and setup_tts() first.")

        rp("Voice chat initiated. Speak your message (or say 'exit' to end the chat).")
        
        while True:
            with speech_recognition.Microphone() as source:
                rp("Listening...")
                audio = self.llm_chatbot.recognizer.listen(source)

            try:
                user_input = self.llm_chatbot.recognizer.recognize_google(audio)
                rp(f"You said: {user_input}")

                if user_input.lower() == 'exit':
                    rp("Ending voice chat.")
                    break

                response = self.chat(user_input)
                rp(f"Bot: {response}")

                # Generate speech from the bot's response
                speech_file = "bot_response.wav"
                self.llm_chatbot.tts.tts_to_file(text=response, file_path=speech_file)
                playsound(speech_file)
                os.remove(speech_file)  # Clean up the temporary audio file

            except speech_recognition.UnknownValueError:
                rp("Sorry, I couldn't understand that. Please try again.")
            except speech_recognition.RequestError as e:
                rp(f"Could not request results from the speech recognition service; {e}")

    def rag_chat(self, query: str, prompt: str = "default_rag_prompt") -> str:
        """
        Perform a RAG (Retrieval-Augmented Generation) chat using the vectorstore and HugChat bot.
        
        :param query: The user's query
        :param prompt: The prompt to use for the retrieval chain (default: "default_rag_prompt")
        :return: The bot's response
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore() first.")

        retriever = self.get_basic_retriever()
        chain = self.create_retrieval_chain(prompt, retriever)
        result = self.run_retrieval_chain(chain, query)
        return result['answer']
    def search_web(self):
        search_query = input("Enter your web search query: ")
        future_date = "July 12, 2024"
        search_url = f"https://www.google.com/search?q={search_query}+before:{future_date}"
        webbrowser.open(search_url)
        print(f"Search results for '{search_query}' on {future_date}:")
        print("=" * 50)
        print(search_url)
        print("=" * 50)

    def advanced_rag_chatbot(self):
        rp("Welcome to the Advanced RAG Chatbot!")
        rp("This chatbot uses a compressed retriever and integrates all components of the vector store.")
        rp("Type 'exit' to end the conversation.")

        # Ensure the vectorstore is initialized
        if self.vectorstore is None:
            rp("Initializing vector store...")
            self.vectorstore, self.docstore, self.index = self.create_indexed_vectorstore(self.chunk_size)

        # Create a compressed retriever
        compressed_retriever = self.get_contextual_compression_retriever(k=5, similarity_threshold=0.75)

        # Initialize conversation history
        conversation_history = []

        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'exit':
                rp("Thank you for using the Advanced RAG Chatbot. Goodbye!")
                break

            # Step 1: Retrieve relevant documents
            retrieved_docs = compressed_retriever.get_relevant_documents(user_input)

            # Step 2: Prepare context from retrieved documents
            context = "\n".join([doc.page_content for doc in retrieved_docs])

            # Step 3: Prepare the prompt
            prompt = prompts['default_rag_prompt'].replace("<<<VSCONTEXT>>",context).replace("<<WSCONTEXT>>",' '.join(conversation_history[-5:]))
            

            # Step 4: Generate response using the chatbot
            
            response = self.llm_chatbot(f"{prompt}\nUser Query: {user_input}\n")

            rp(f"Chatbot: {response}")

            # Update conversation history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Chatbot: {response}")

            # Step 5: Demonstrate use of individual components
            rp("\nAdditional Information:")
            rp(f"- Number of documents in docstore: {len(self.docstore.docstore)}")
            rp(f"- Number of vectors in index: {self.index.ntotal}")
            
            # Demonstrate direct use of vectorstore for similarity search
            similar_docs = self.vectorstore.similarity_search(user_input, k=1)
            if similar_docs:
                rp(f"- Most similar document: {similar_docs[0].metadata.get('source', 'Unknown')}")
            
            # Generate a 3D scatter plot of the vectorstore content
            avs.generate_3d_scatterplot()
            
            # Optional: Add user feedback loop
            feedback = input("Was this response helpful? (yes/no): ").strip().lower()
            if feedback == 'no':
                rp("I'm sorry the response wasn't helpful. Let me try to improve it.")
                # Here you could implement logic to refine the response or adjust the retrieval process
                with open(file="./feedback_NO.txt",mode="a+")as f:
                    f.write(f"chat_feedback_NO\nChatHistory--->{' '.join(conversation_history[-10:])}")



# Example usage:
if __name__ == "__main__":
    email = os.getenv("EMAIL")
    password = os.getenv("PASSWD")
    github_token = os.getenv("GITHUB_TOKEN")
    
    # Initialize AdvancedVectorStore with HugChat bot
    avs = AdvancedVectorStore(email=email, password=password)

    # Create the indexed vectorstore
    #avs.create_indexed_vectorstore()

    # Clone a GitHub repository and load its contents
    
    avs.load_documents_folder("/nr_ywo/coding/voice_chat_rag_web/venv/lib/python3.10/site-packages/TTS/tts")
    avs.load_github_repo("https://github.com/bxck75/agents_of_doom")
    avs.save_vectorstore(path=avs.storage_path)
    avs.load_vectorstore(path=avs.storage_path)
    # rp document and chunk counts
    rp(f"Total documents: {avs.document_count}")
    rp(f"Total chunks: {avs.chunk_count}")
    retriever=avs.get_multi_query_compression_retriever()
    q="Demonstrate your knowledge of faiss in a python  OOP example script"
    rel_docs=retriever.get_relevant_documents(query=q)
      # Start the advanced RAG chatbot
    avs.advanced_rag_chatbot()

    # Perform a RAG chat
    rag_response = avs.rag_chat(query="Explain the concept of neural networks.")
    rp("RAG chat response:", rag_response)

    # Set up speech recognition and TTS for voice chat
    #avs.setup_speech_recognition()
    #avs.setup_tts()

    # Start a voice chat session
    #avs.voice_chat()
    """ 
        # Using different retrievers
        multi_query_retriever = avs.get_multi_query_retriever()
        results = avs.search("What is deep learning?", mode="multi_query")
        rp("Multi-query retriever results:", results)

        self_query_retriever = avs.get_self_query_retriever()
        results = avs.search("Find documents about reinforcement learning", self_query_retriever)
        rp("Self-query retriever results:", results)

        contextual_compression_retriever = avs.get_contextual_compression_retriever()
        results = avs.search("Explain the difference between supervised and unsupervised learning", contextual_compression_retriever)
        rp("Contextual compression retriever results:", results)

    """
    """     # Perform a basic search
        k = 4
        similarity_threshold = 0.78
        q = "What is machine learning?"

        basic_results = avs.search(q, mode='basic', k=k)
        rp("Basic search results:", basic_results)
        rp("self_query search results:", self_query_results)
        rp("multi_query search results:", multi_results)
        rp("Compressed search results:", commpressed_results)
    """

  
""" This advanced example demonstrates:

Use of the compressed retriever for efficient document retrieval.
Integration of conversation history for context-aware responses.
Direct use of the vectorstore for similarity search.
Access to the docstore and index for additional information.
A feedback loop to potentially improve responses (though the improvement logic is not implemented in this example).

This chatbot loop showcases how all components of the system can work together to provide informative responses based on the loaded documents. It also demonstrates how you can access and use individual components (docstore, index, vectorstore) for additional functionality or information.
To further optimize this system, you could consider:

Implementing caching mechanisms to speed up repeated queries.
Adding more sophisticated feedback handling to improve retrieval and response generation over time.
Implementing dynamic index updates if new information becomes available during the chat session.
Adding options for users to see the sources of information or request more details on specific topics.

This example provides a solid foundation that you can further customize and expand based on your specific needs and use cases. """