import os
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

# Core dependencies
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, PythonLoader

class CompactRAGChat:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 storage_path: str = "./chat_store",
                 chunk_size: int = 384,
                 chunk_overlap: int = 32):
        
        # Initialize basic configuration
        self.storage_path = storage_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.conversation_history = []
        self.max_history_items = 20
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Set up vector store
        self._initialize_vectorstore()
        
        # Text splitter for documents
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.logger.info("CompactRAGChat initialized successfully")




        #To view the list of separators for a given language, pass a value from this enum into

        RecursiveCharacterTextSplitter.get_separators_for_language

        #To instantiate a splitter that is tailored for a specific language, pass a value from the enum into

        RecursiveCharacterTextSplitter.from_language

        #Below we demonstrate examples for the various languages.

        #%pip install -qU langchain-text-splitters

        from langchain_text_splitters import (
            Language,
            RecursiveCharacterTextSplitter,
        )

        #API Reference:Language | RecursiveCharacterTextSplitter

        #To view the full list of supported languages:

        all_langs=[e.value for e in Language]

        print(all_langs)





    def _setup_logging(self):
        """Set up basic logging."""
        logger = logging.getLogger("CompactRAGChat")
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def _initialize_vectorstore(self):
        """Initialize or load the vector store."""
        try:
            if os.path.exists(os.path.join(self.storage_path, "index.faiss")):
                self.logger.info("Loading existing vector store")
                self.vectorstore = FAISS.load_local(
                    folder_path=self.storage_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                self.logger.info("Creating new vector store")
                self.vectorstore = FAISS.from_documents(
                    documents=[Document(page_content="Initialization document", metadata={"type": "system"})],
                    embedding=self.embeddings
                )
                self.vectorstore.save_local(self.storage_path)
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            # Fallback to new vector store
            self.vectorstore = FAISS.from_documents(
                documents=[Document(page_content="Initialization document", metadata={"type": "system"})],
                embedding=self.embeddings
            )

    def add_document(self, file_path: str) -> int:
        """Add a document to the vector store."""
        try:
            # Select appropriate loader based on file extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.py':
                loader = PythonLoader(file_path)
            else:
                loader = TextLoader(file_path)
            
            # Load and split the document
            documents = loader.load()
            chunks = self.splitter.split_documents(documents)
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk.metadata["source"] = file_path
                chunk.metadata["added_at"] = datetime.now().isoformat()
                chunk.metadata["type"] = "document"
            
            # Add to vector store
            self.vectorstore.add_documents(chunks)
            self.vectorstore.save_local(self.storage_path)
            
            self.logger.info(f"Added {len(chunks)} chunks from {file_path}")
            return len(chunks)
        except Exception as e:
            self.logger.error(f"Error adding document {file_path}: {str(e)}")
            return 0

    def add_conversation_item(self, role: str, content: str) -> None:
        """Add a conversation item to the history and vector store."""
        # Add to conversation history
        timestamp = datetime.now().isoformat()
        conversation_item = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        self.conversation_history.append(conversation_item)
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history_items:
            self.conversation_history = self.conversation_history[-self.max_history_items:]
        
        # Add to vector store
        doc = Document(
            page_content=content,
            metadata={
                "type": "conversation",
                "role": role,
                "timestamp": timestamp
            }
        )
        self.vectorstore.add_documents([doc])
        self.vectorstore.save_local(self.storage_path)

    def get_context(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant context for a query."""
        try:
            # Get relevant documents
            documents = self.vectorstore.similarity_search(query, k=k)
            return documents
        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            return []

    def get_conversation_history(self, max_items: int = None) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        if max_items is None or max_items > len(self.conversation_history):
            return self.conversation_history
        return self.conversation_history[-max_items:]

    def format_context_for_prompt(self, documents: List[Document]) -> str:
        """Format retrieved documents into a context string for the prompt."""
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Format differently based on document type
            if doc.metadata.get("type") == "conversation":
                role = doc.metadata.get("role", "unknown")
                timestamp = doc.metadata.get("timestamp", "unknown time")
                context_parts.append(f"[Previous {role} at {timestamp}]: {doc.page_content}")
            else:
                source = doc.metadata.get("source", "unknown source")
                context_parts.append(f"[Document from {source}]: {doc.page_content}")
        
        return "\n\n".join(context_parts)

    def chat(self, user_input: str, external_llm_func=None) -> str:
        """Process user input and generate a response."""
        # Add user input to conversation history
        self.add_conversation_item("user", user_input)
        
        # Get relevant context
        context_docs = self.get_context(user_input)
        context_str = self.format_context_for_prompt(context_docs)
        
        # Generate response
        if external_llm_func:
            # Use provided external LLM function
            prompt = f"""
            Conversation history and relevant context:
            {context_str}
            
            Current user message: {user_input}
            
            Please provide a helpful response based on the conversation history and context provided.
            """
            response = external_llm_func(prompt)
        else:
            # Fallback to simple echo mode if no LLM function provided
            response = f"Echo: {user_input}\n\nContext used:\n{context_str}"
        
        # Add response to conversation history
        self.add_conversation_item("assistant", response)
        
        return response

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []

    def run_interactive(self, external_llm_func=None):
        """Run an interactive chat session."""
        print("Welcome to CompactRAGChat! Type 'exit' to quit.")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            response = self.chat(user_input, external_llm_func)
            print(f"\nAI: {response}")



if __name__ == "__main__":
    # Example usage
    #from transformers import pipeline

    # Initialize the system
    rag_chat = CompactRAGChat(storage_path="./my_chat_store")
    from glob import glob
    files = glob("/media/codemonkeyxl/TBofCode/MainCodingFolder/new_coding/expert_augmenting_system/voice_chat_rag_web/**/*.py")
    for file in files:
        if file.endswith(".py"):
            rag_chat.add_document(file)

    # Add documents for context (optional)
    #rag_chat.add_document("project_documentation.pdf")
    #rag_chat.add_document("meeting_notes.txt")

    # Define an external LLM function (example using Hugging Face pipeline)
    def llm_function(prompt):
        # You can replace this with any LLM API call
        pipe = pipeline("text-generation", model="google/flan-t5-small")
        return pipe(prompt, max_length=512)[0]["generated_text"]

    # Run interactive chat
    rag_chat.run_interactive(llm_function)