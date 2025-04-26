import os
import glob
from typing import Dict, List, Tuple, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter

class DocumentStore:
    def __init__(self, embedding_model_name: str = "pkshatech/GLuCoSE-base-ja-v2", index_dir: str = "indexes"):
        """ドキュメントストアの初期化"""
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.index_dir = index_dir
        self.vectorstores = {}
        self.documents = {}
        self.text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
        
        # インデックスディレクトリが存在しない場合は作成
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        
        # 既存のインデックスを読み込む
        self.load_indexes()
    
    def load_indexes(self) -> None:
        """保存されているインデックスを読み込む"""
        index_files = glob.glob(os.path.join(self.index_dir, "*.faiss"))
        for index_file in index_files:
            # ファイル名からカテゴリ名を取得 (例: indexes/batch_design.faiss -> batch_design)
            category = os.path.basename(index_file).replace(".faiss", "")
            try:
                # インデックスを読み込む
                # 注意: allow_dangerous_deserialization=True は自分自身で作成した信頼できるピックルファイルの場合のみ使用してください
                # 信頼できない外部ソースからのファイルには使用しないでください
                self.vectorstores[category] = FAISS.load_local(
                    self.index_dir, 
                    self.embeddings, 
                    index_name=category,
                    allow_dangerous_deserialization=True  # 自分で作成したピックルファイルの場合のみTrueに設定
                )
                print(f"インデックスを読み込みました: {category}")
            except Exception as e:
                print(f"インデックス読み込みエラー ({category}): {e}")
    
    def load_documents_from_directory(self, directory: str) -> None:
        """指定ディレクトリ内のMarkdownファイルを読み込み、カテゴリごとにインデックスを作成する"""
        if not os.path.exists(directory):
            print(f"ディレクトリが存在しません: {directory}")
            return
        
        # サブディレクトリを取得
        subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        
        for subdir in subdirs:
            category = subdir
            md_files = glob.glob(os.path.join(directory, subdir, "*.md"))
            
            all_documents = []
            
            for md_file in md_files:
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # ファイル情報をメタデータとして保存
                    metadata = {
                        "source": md_file,
                        "filename": os.path.basename(md_file),
                        "category": category
                    }
                    
                    # MarkdownTextSplitterを使用してドキュメントを分割
                    doc = Document(page_content=content, metadata=metadata)
                    split_docs = self.text_splitter.split_documents([doc])
                    all_documents.extend(split_docs)
                    
                except Exception as e:
                    print(f"ファイル読み込みエラー ({md_file}): {e}")
            
            if all_documents:
                if category not in self.documents:
                    self.documents[category] = []
                
                self.documents[category].extend(all_documents)
                
                # カテゴリに応じてベクトルストアを取得または作成
                if category not in self.vectorstores or self.vectorstores[category] is None:
                    self.vectorstores[category] = FAISS.from_documents(all_documents, self.embeddings)
                else:
                    self.vectorstores[category].add_documents(all_documents)
                
                self.save_index(category)
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, category: str = "default") -> None:
        """ドキュメントをベクトルストアに追加する"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        documents = [Document(page_content=text, metadata=metadata) 
                     for text, metadata in zip(texts, metadatas)]
        
        # MarkdownTextSplitterを使用してドキュメントを分割
        split_documents = []
        for doc in documents:
            split_docs = self.text_splitter.split_documents([doc])
            split_documents.extend(split_docs)
        
        # カテゴリに応じてドキュメントを保存
        if category not in self.documents:
            self.documents[category] = []
        
        self.documents[category].extend(split_documents)
        
        # カテゴリに応じてベクトルストアを取得または作成
        if category not in self.vectorstores or self.vectorstores[category] is None:
            self.vectorstores[category] = FAISS.from_documents(split_documents, self.embeddings)
        else:
            self.vectorstores[category].add_documents(split_documents)
    
    def save_index(self, category: str) -> None:
        """カテゴリのインデックスを保存する"""
        if category in self.vectorstores and self.vectorstores[category] is not None:
            try:
                self.vectorstores[category].save_local(self.index_dir, index_name=category)
                print(f"インデックスを保存しました: {category}")
            except Exception as e:
                print(f"インデックス保存エラー ({category}): {e}")
    
    def save_all_indexes(self) -> None:
        """すべてのカテゴリのインデックスを保存する"""
        for category in self.vectorstores:
            self.save_index(category)
    
    def search(self, query: str, category: str = None, k: int = 3) -> List[Document]:
        """クエリに関連するドキュメントを検索する
        
        Args:
            query: 検索クエリ
            category: 検索対象のカテゴリ。Noneの場合はすべてのカテゴリを検索
            k: 返却する最大ドキュメント数
        """
        results = []
        
        if category is not None:
            # 特定のカテゴリのみ検索
            if category in self.vectorstores and self.vectorstores[category] is not None:
                return self.vectorstores[category].similarity_search(query, k=k)
        else:
            # すべてのカテゴリを検索
            for cat, vectorstore in self.vectorstores.items():
                if vectorstore is not None:
                    # 各カテゴリから結果を取得
                    cat_results = vectorstore.similarity_search(query, k=min(k, 2))  # カテゴリあたりの結果数を制限
                    results.extend(cat_results)
        
        # 最大k件までに制限して返却
        return results[:k]
    
    def get_all_categories(self) -> List[str]:
        """利用可能なすべてのカテゴリを取得する"""
        return list(self.vectorstores.keys())
        
    def get_documents(self, category: str) -> List[Document]:
        """指定したカテゴリのドキュメントを取得する
        
        Args:
            category: ドキュメントを取得するカテゴリ名
        
        Returns:
            指定したカテゴリのドキュメントリスト。カテゴリが存在しない場合は空リスト
        """
        if category in self.documents:
            return self.documents[category]
        return []