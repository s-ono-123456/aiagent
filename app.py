import streamlit as st
import openai
import os
from llm_client import get_gpt_response

# OpenAI APIキーの設定
openai.api_key = os.getenv("OPENAI_API_KEY")
# Streamlitの設定
st.title("サンプル Streamlit アプリ")
st.write("これはStreamlitを使ったサンプルアプリケーションです。")

# ユーザーからの入力を受け取る
user_input = st.text_input("質問を入力してください:")

if user_input:
    response = get_gpt_response(user_input)
    st.write("応答:", response)