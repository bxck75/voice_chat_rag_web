import os
import faiss
from langchain_community.vectorstores import FAISS

from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter, 
    TextSplitter
)
import requests
import warnings 
from dotenv import  load_dotenv, find_dotenv
from langchain.chains import LLMChain

#The first time you generate the embeddings, it may take a while (approximately 20 seconds) for the API to return them. We use the retry decorator (install with pip install retry) so that if on the first try, output = query(dict(inputs = texts)) doesn't work, wait 10 seconds and try three times again. This happens because, on the first request, the model needs to be downloaded and installed on the server, but subsequent calls are much faster.

def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
model_id = "sentence-transformers/all-MiniLM-L6-v2"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {token}"}


# Load environment variables
load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")
os.environ['FAISS_NO_AVX2'] = '1'
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")
token = os.getenv("HF_TOKEN")
class DocumentProcessor:
    def __init__(self, persist_dir, embeddings_model):
        self.persistant_dir = persist_dir
        self.embeddings = embeddings_model
        self.vectorstore = None
        self.document_ids = {}

        # Define specific splitters
        self.python_splitter = RecursiveCharacterTextSplitter()
        self.markdown_splitter = MarkdownTextSplitter()
        self.text_splitter = TextSplitter()

    def process_folder(self, folder_path: str):
        """
        Process all the scripts and documents from a folder recursively.
        """
        all_documents = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
                doc = Document(page_content=file_content, metadata={'source': file_path})
                all_documents.append(doc)

        # Split documents into chunks based on type
        split_docs = self.split_documents(all_documents)
        return split_docs

    def split_documents(self, documents: list):
        """
        Split documents into smaller chunks based on file type.
        """
        split_docs = []
        for doc in documents:
            ext = self._get_file_extension(doc)
            splitter = self._get_splitter_by_extension(ext)
            split_docs.extend(splitter.split_documents([doc]))
        return split_docs

    def _get_file_extension(self, document):
        """
        Get the file extension from the document's metadata.
        """
        ext = os.path.splitext(document.metadata.get('source', ''))[1].lower()
        print(f"Processing file with extension: {ext}")
        return ext

    def _get_splitter_by_extension(self, ext):
        """
        Return the appropriate splitter based on the file extension.
        """
        if ext == '.py':
            return self.python_splitter
        elif ext in ['.md', '.markdown']:
            return self.markdown_splitter
        else:
            return self.text_splitter

    def create_vectorstore(self, documents: list):
        """
        Create and save the FAISS vector store from the documents.
        """
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.vectorstore.save(self.persistant_dir)

        # Store document IDs for future reference
        for doc in documents:
            doc_id = self.vectorstore._collection.count()  # Get the current count as the new ID
            self.document_ids[doc.metadata.get('source', f'doc_{doc_id}')] = doc_id
            self.vectorstore.add_documents([doc])  # Add each document individually to get its ID
            print(f"Added document {doc.metadata.get('source', f'doc_{doc_id}')}, ID: {doc_id}")

    def load_vectorstore(self):
        """
        Load an existing FAISS vector store.
        """
        print("Loading existing vector store...")
        self.vectorstore = FAISS.load(self.persistant_dir)

    def setup_retriever(self, k: int = 5):
        """
        Set up the retriever for querying the vector store.
        """
        print(f"Setting up retriever with k={k}...")
        self.retriever = self.vectorstore.as_retriever(k=k)
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.76)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=self.retriever)

    def pretty_print_docs(self, docs):
        """
        Print the document contents.
        """
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
            )
        )


# Example Usage
if __name__ == "__main__":
    persist_dir = "vectorstore"  # Directory to save the FAISS vector store
    processor = DocumentProcessor(persist_dir, embeddings_model)

    folder_path = "/media/codemonkeyxl/TBofCode/MainCodingFolder/new_coding/expert_augmenting_system"  # Folder to scan for scripts
    documents = processor.process_folder(folder_path)

    # Create or load the FAISS vector store
    if not os.path.exists(persist_dir):
        processor.create_vectorstore(documents)
    else:
        processor.load_vectorstore()

    # Set up retriever
    processor.setup_retriever(k=5)

    # Pretty print some documents
    processor.pretty_print_docs(documents[:5])  # Print the first 5 documents
