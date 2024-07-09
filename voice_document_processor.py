# document_processing.py
import os
from git import Repo
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from langchain.retrievers import MultiQueryRetriever
from langchain.llms import BaseLLM


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
        self.document_ids = {}  # Add this line to initialize document_ids

    def clone_github_repo(self, repo_url: str, local_path: str = './repo'):
        if os.path.exists(local_path):
            print("Repository already cloned.")
            return local_path
        Repo.clone_from(repo_url, local_path)
        return local_path

    def load_documents_from_github(self, repo_url: str, local_path: str = './repo', file_types: list = ['*.py', '*.md', '*.txt','*.html']):
        local_repo_path = self.clone_github_repo(repo_url, local_path)
        self.loader = DirectoryLoader(repo_path=local_repo_path,  glob=f"**/{{{','.join(file_types)}}}", show_progress=True, recursive=True)
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
        # Assuming the document object has a 'source' or 'filename' attribute
        filename = getattr(document, 'source', '') or getattr(document, 'filename', '')
        ext=os.path.splitext(filename)[1].lower()
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
        
    def embed_documents(self, documents: list):
        return self.embeddings.embed_documents(documents)

    def create_vectorstore(self, documents: list, embeddings: list, collection_name="voice_hugchat", persist_directory: str = './vectorstore'):
        self.vectorstore = Chroma.from_documents(documents, embeddings, collection_name=collection_name, persist_directory=persist_directory)
        # Store document names and their IDs
        for doc in documents:
            doc_id = self.vectorstore._collection.count()  # Get the current count as the new ID
            self.document_ids[doc.metadata.get('source', f'doc_{doc_id}')] = doc_id
            self.vectorstore.add_documents([doc])  # Add each document individually to get its ID

    def setup_retriever(self, k: int = 5):
        self.retriever = MultiQueryRetriever(vectorstore=self.vectorstore, llm=self.llm, search_k=k)

    def retrieve_similar_documents(self, query: str):
        return self.retriever.get_relevant_documents(query)

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    documents = processor.load_documents_from_github('https://github.com/your-repo/your-project.git')
    split_docs = processor.split_documents(documents)
    embeddings = processor.embed_documents(split_docs)
    processor.create_vectorstore(split_docs, embeddings)
    processor.setup_retriever()
    query = "Enter your query here"
    results = processor.retrieve_similar_documents(query)
    for result in results:
        print(result)
