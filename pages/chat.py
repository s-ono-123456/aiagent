import streamlit as st
import os
from llm_client import get_gpt_response

# タイトルとページ説明
st.title("LLMチャット")
st.write("AIアシスタントとの対話ができるシンプルなチャットインターフェースです。")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力の受け取り
if prompt := st.chat_input("メッセージを入力してください"):
    # ユーザーメッセージをチャット履歴に追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # ユーザーメッセージの表示
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AIの応答を生成
    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            response = get_gpt_response(prompt)
            st.markdown(response)
    
    # AIの応答をチャット履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": response})

# チャット履歴のクリアボタン
if st.button("チャット履歴をクリア"):
    st.session_state.messages = []
    st.rerun()