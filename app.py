import streamlit as st
import openai
import os
from llm_client import get_gpt_response
from agents import get_multi_agent_response
from document_store import DocumentStore

# OpenAI APIキーの設定
openai.api_key = os.getenv("OPENAI_API_KEY")

# ドキュメントストアの初期化
@st.cache_resource
def get_document_store():
    doc_store = DocumentStore()
    # サンプルドキュメントを追加
    sample_texts = [
        "LangGraphは、LangChainエコシステムの一部で、複雑なAIアプリケーションを構築するためのフレームワークです。",
        "LangGraphを使用すると、複数のエージェントを組み合わせたワークフローを定義できます。",
        "RAG（検索拡張生成）は、LLMに外部データを提供するテクニックです。",
        "マルチエージェントシステムでは、異なる役割を持つ複数のAIエージェントが協力して問題を解決します。",
        "エージェントは特定のタスクに最適化された専門家として機能し、より複雑な問題を解決するために協力します。"
    ]
    doc_store.add_documents(sample_texts)
    return doc_store

# Streamlitの設定
st.title("マルチエージェント型RAGサンプル")
st.write("このアプリはLangGraphを使用したマルチエージェント型RAGのサンプル実装です。")

# タブを作成
tab1, tab2 = st.tabs(["シンプルLLM", "マルチエージェントRAG"])

with tab1:
    st.header("シンプルLLM")
    st.write("直接LLMに質問を投げるシンプルな対話です。")
    
    # ユーザーからの入力を受け取る
    user_input_simple = st.text_input("質問を入力してください:", key="simple_input")
    
    if user_input_simple:
        with st.spinner("回答を生成中..."):
            response = get_gpt_response(user_input_simple)
            st.write("応答:", response)

with tab2:
    st.header("マルチエージェントRAG")
    st.write("LangGraphを使用したマルチエージェント型RAGシステムです。複数の専門エージェントが協力して回答を生成します。")
    
    # 思考過程を表示するかどうかのチェックボックス
    show_thoughts = st.checkbox("エージェントの思考過程を表示", value=True)
    
    # ユーザーからの入力を受け取る
    user_input_rag = st.text_input("質問を入力してください:", key="rag_input")
    
    # ユーザー入力があれば
    if user_input_rag:
        with st.spinner("マルチエージェントRAGシステムが回答を生成中..."):
            # ドキュメントストアを取得
            doc_store = get_document_store()
            
            # マルチエージェントRAGで回答を生成
            result = get_multi_agent_response(user_input_rag, doc_store, show_thoughts)
            
            # 思考過程を表示（オプション）
            if show_thoughts and "thoughts" in result:
                st.subheader("エージェントの思考過程:")
                st.text(result["thoughts"])
            
            # 最終的な応答を表示
            st.subheader("最終応答:")
            st.write(result["response"])
    
    # ドキュメント追加機能
    st.divider()
    st.subheader("ドキュメントの追加")
    
    new_doc = st.text_area("新しいドキュメントを追加:", placeholder="ここに新しいドキュメントを入力してください")
    if st.button("追加"):
        if new_doc:
            doc_store = get_document_store()
            doc_store.add_documents([new_doc])
            st.success("ドキュメントが追加されました！")
        else:
            st.error("ドキュメントを入力してください")