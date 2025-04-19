import os
from typing import Dict, List, Tuple, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

class DocumentStore:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """ドキュメントストアの初期化"""
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vectorstore = None
        self.documents = []
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """ドキュメントをベクトルストアに追加する"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        documents = [Document(page_content=text, metadata=metadata) 
                     for text, metadata in zip(texts, metadatas)]
        
        self.documents.extend(documents)
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)
    
    def search(self, query: str, k: int = 3) -> List[Document]:
        """クエリに関連するドキュメントを検索する"""
        if self.vectorstore is None:
            return []
        
        return self.vectorstore.similarity_search(query, k=k)