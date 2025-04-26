import streamlit as st
import openai
import os
from llm_client import get_gpt_response
from agents import get_multi_agent_response, create_multi_agent_rag
from document_store import DocumentStore
from streamlit_mermaid import st_mermaid

# OpenAI APIキーの設定
openai.api_key = os.getenv("OPENAI_API_KEY")

# ドキュメントストアの初期化
@st.cache_resource
def get_document_store():
    doc_store = DocumentStore()
    
    # 既存のインデックスを使用するか確認
    if os.path.exists("indexes"):
        # インデックスが既に存在する場合はインデックスから読み込む
        return doc_store
    
    # インデックスが存在しない場合は、サンプルディレクトリからドキュメントを読み込む
    if os.path.exists("sample"):
        doc_store.load_documents_from_directory("sample")
    
    # サンプルドキュメントも追加
    sample_texts = [
        "LangGraphは、LangChainエコシステムの一部で、複雑なAIアプリケーションを構築するためのフレームワークです。",
        "LangGraphを使用すると、複数のエージェントを組み合わせたワークフローを定義できます。",
        "RAG（検索拡張生成）は、LLMに外部データを提供するテクニックです。",
        "マルチエージェントシステムでは、異なる役割を持つ複数のAIエージェントが協力して問題を解決します。",
        "エージェントは特定のタスクに最適化された専門家として機能し、より複雑な問題を解決するために協力します。"
    ]
    doc_store.add_documents(sample_texts)
    
    # インデックスを保存
    doc_store.save_all_indexes()
    
    return doc_store

# Streamlitの設定
st.title("マルチエージェント型RAGサンプル")
st.write("このアプリはLangGraphを使用したマルチエージェント型RAGのサンプル実装です。")

# ドキュメントストアの初期化
doc_store = get_document_store()
categories = doc_store.get_all_categories()

# タブの作成
workflow_tab, main_tab, thoughts_tab, documents_tab = st.tabs(["ワークフローグラフ", "メインチャット", "エージェント思考過程", "検索ドキュメント"])

with workflow_tab:
    st.header("エージェントのワークフローグラフ")
    
    # エージェントのワークフローグラフを取得して表示
    if 'workflow_graph' not in st.session_state:
        # エージェントのグラフを生成
        graph = create_multi_agent_rag(doc_store)
        # Mermaidダイアグラムを取得
        mermaid_code = graph.get_graph().draw_mermaid()
        st.session_state.workflow_graph = mermaid_code
    
    # streamlit-mermaidを使用してグラフを視覚化
    st.markdown("### グラフの視覚化")
    st_mermaid(st.session_state.workflow_graph, show_controls=False)
    
    # Mermaidダイアグラムをテキスト形式で表示
    st.markdown("### Mermaidグラフ (テキスト形式)")
    st.code(st.session_state.workflow_graph, language="mermaid")
with main_tab:
    # マルチエージェントRAGセクション
    st.header("マルチエージェントRAG")
    st.write("LangGraphを使用したマルチエージェント型RAGシステムです。複数の専門エージェントが協力して回答を生成します。")

    # カテゴリ選択（もしあれば）
    selected_category = None
    if categories:
        selected_category = st.selectbox(
            "検索対象カテゴリ", 
            ["すべて"] + categories, 
            index=0
        )
        
        if selected_category == "すべて":
            selected_category = None

    # 思考過程を表示するかどうかのチェックボックス
    show_thoughts = st.checkbox("エージェントの思考過程を表示", value=True)

    # ユーザーからの入力を受け取る
    user_input_rag = st.text_input("質問を入力してください:", key="rag_input", value="受注処理の詳細を教えてください")

    # 結果表示用のセッション状態変数
    if 'result' not in st.session_state:
        st.session_state.result = None

    # ユーザー入力があれば
    if user_input_rag:
        with st.spinner("マルチエージェントRAGシステムが回答を生成中..."):
            # マルチエージェントRAGで回答を生成
            st.session_state.result = get_multi_agent_response(user_input_rag, doc_store, show_thoughts)
            
            # 最終的な応答を表示
            st.subheader("最終応答:")
            st.write(st.session_state.result["response"])

with thoughts_tab:
    st.header("エージェントの思考過程")
    
    if st.session_state.get('result') and "thoughts" in st.session_state.result:
        st.text(st.session_state.result["thoughts"])
    else:
        st.info("エージェントの思考過程はまだありません。メインチャットタブで質問を入力してください。")

with documents_tab:
    st.header("検索されたドキュメント一覧")
    
    if st.session_state.get('result') and "documents" in st.session_state.result and st.session_state.result["documents"]:
        documents = st.session_state.result["documents"]
        st.write(f"検索結果: {len(documents)}件のドキュメントが見つかりました")
        
        for i, doc in enumerate(documents):
            # メタデータから情報を取得
            source = doc.metadata.get('source', '不明')
            filename = doc.metadata.get('filename', os.path.basename(source) if source != '不明' else '不明')
            category = doc.metadata.get('category', '不明')
            
            # エキスパンダーを使用してドキュメント内容を表示
            with st.expander(f"ドキュメント {i+1}: {filename}"):
                st.markdown("**カテゴリ**: " + category)
                st.markdown("**ソース**: " + source)
                st.markdown("**内容**:")
                st.markdown(doc.page_content)
    else:
        st.info("検索されたドキュメントはまだありません。メインチャットタブで質問を入力してください。")

    