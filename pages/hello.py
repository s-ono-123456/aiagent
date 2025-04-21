import streamlit as st
from mcp_agent import run_mcp_agent

# Streamlitアプリのタイトル設定
st.title("Playwright MCPエージェント")

# 説明文を追加
st.write("このアプリはPlaywright MCPを使用してウェブブラウザを操作し、質問に答えるAIエージェントです。")
st.write("URLを含む質問を入力すると、AIがブラウザでWebサイトを閲覧し、情報を収集して回答します。")

# サンプル質問
st.subheader("サンプル質問")
st.code("https://www.google.co.jp/にアクセスし、「Google」と入力して検索するテストを行ってください。")

# ユーザー入力
query = st.text_area("質問を入力してください:", height=100)

# 処理中のインジケーター用の変数
if 'processing' not in st.session_state:
    st.session_state.processing = False

# 回答結果用の変数
if 'result' not in st.session_state:
    st.session_state.result = ""

# 送信ボタン
if st.button("送信"):
    if query:
        # 処理中フラグをオン
        st.session_state.processing = True
        
        # 処理中のインジケーターを表示
        with st.spinner("AIがブラウザを操作して回答を生成中..."):
            # MCPエージェントを実行して結果を取得
            result = run_mcp_agent(query)
            st.session_state.result = result
        
        # 処理中フラグをオフ
        st.session_state.processing = False

# 結果の表示
if st.session_state.result:
    st.subheader("回答:")
    st.markdown(st.session_state.result)
