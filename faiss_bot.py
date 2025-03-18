import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import gradio as gr
from dotenv import load_dotenv, find_dotenv
import warnings
load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")
token = os.getenv("HF_TOKEN")

class FAISSChatbot:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 document_store=None, vector_dim=384):
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_dim = vector_dim
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.vector_dim)
        
        # Store for documents and their embeddings
        self.documents = []
        
        # Load documents if provided
        if document_store:
            self.load_documents(document_store)
    
    def load_documents(self, document_store):
        """Load documents from a list of strings or a file"""
        if isinstance(document_store, list):
            self.documents = document_store
        elif isinstance(document_store, str) and os.path.isfile(document_store):
            with open(document_store, 'r', encoding='utf-8') as f:
                self.documents = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("document_store must be a list of strings or a valid file path")
        
        # Create embeddings for documents
        self.add_documents(self.documents)
    
    def add_documents(self, documents):
        """Add documents to the FAISS index"""
        if not documents:
            return
            
        # Create embeddings
        embeddings = self.embedding_model.encode(documents)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
    
    def search(self, query, k=3):
        """Search for similar documents"""
        # Create embedding for the query
        query_embedding = self.embedding_model.encode([query])
        
        # Search the FAISS index
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        # Return relevant documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                results.append({
                    'text': self.documents[idx],
                    'score': float(distances[0][i])
                })
        
        return results
    
    def generate_response(self, message, system_message="You are a friendly and helpful assistant.", 
                          context_window=3, max_tokens=512, temperature=0.7):
        """Generate a response based on retrieved context and query"""
        # Get relevant documents as context
        context_docs = self.search(message, k=context_window)
        context = "\n".join([doc['text'] for doc in context_docs])
        
        # Here you'd typically call your LLM API
        # For simplicity, we'll return a template response
        # In production, replace this with an actual LLM call
        if not context_docs:
            return f"I don't have specific information about that in my knowledge base. {system_message}"
        
        return f"Based on my knowledge: {context}\n\nIs there anything specific you'd like to know about this?"

# Create a simple Gradio interface
def create_chatbot_interface(doc_store=None):
    # Initialize the chatbot
    chatbot = FAISSChatbot(document_store=doc_store)
    
    # Sample documents if none provided
    if not doc_store:
        sample_docs = [
            "FAISS is a library for efficient similarity search and clustering of dense vectors.",
            "Vector embeddings represent words or documents as numerical vectors.",
            "Chatbots use natural language processing to simulate conversation with users.",
            "Gradio is a library for building machine learning web interfaces quickly.",
            "Sentence transformers are models that convert sentences into meaningful embeddings."
        ]
        chatbot.add_documents(sample_docs)
    
    # Define the Gradio interface
    def respond(message, system_message, max_tokens, temperature, chat_history):
        response = chatbot.generate_response(
            message, 
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature
        )
        chat_history.append((message, response))
        return "", chat_history
    
    with gr.Blocks() as demo:
        gr.Markdown("# FAISS Vector Store Backed Chatbot")
        
        with gr.Row():
            with gr.Column():
                system_message = gr.Textbox(
                    label="System Message", 
                    placeholder="You are a friendly and helpful assistant.",
                    value="You are a friendly and helpful assistant."
                )
                max_tokens = gr.Slider(
                    label="Max Tokens", 
                    minimum=32, 
                    maximum=1024, 
                    value=512, 
                    step=32
                )
                temperature = gr.Slider(
                    label="Temperature", 
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.7, 
                    step=0.1
                )
        
        chatbot = gr.Chatbot(label="Conversation")
        msg = gr.Textbox(label="Your message")
        
        msg.submit(
            respond, 
            [msg, system_message, max_tokens, temperature, chatbot], 
            [msg, chatbot]
        )
    
    return demo
email="goldenkooy@gmail.com"
password="CodeMoneyXL0o9i8u7y!."
# Run the interface
if __name__ == "__main__":
    from llm_chatbot import LLMChatBot
    llm=LLMChatBot(email, password)
   
    # You can replace None with a path to your document file or a list of documents
    demo = create_chatbot_interface(doc_store="/media/codemonkeyxl/TBofCode/MainCodingFolder/new_coding/expert_augmenting_system/voice_chat_rag_web/README.md")
    demo.launch()