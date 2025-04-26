import streamlit as st
import os
import json
from datetime import datetime
from llm_client import get_gpt_response

# タイトルとページ説明
st.title("拡張チャットインターフェース")
st.write("高度な機能を備えたAIアシスタントとの対話インターフェースです。")

# サイドバーの設定
st.sidebar.header("設定")

# モデル選択（将来的な拡張のため）
model = st.sidebar.selectbox(
    "モデル",
    ["gpt-4.1-nano", "gpt-4o", "gpt-4"],
    index=0
)

# プロンプトテンプレート
prompt_templates = {
    "一般的な会話": "以下は、人間とAIアシスタントの会話です。AIアシスタントは親切で、正確で、無害です。\n\n{input}",
    "コードアシスタント": "あなたは熟練したプログラマーです。以下のプログラミングに関する質問に簡潔に回答してください：\n\n{input}",
    "学習サポート": "あなたは教育者です。以下の学習に関する質問に分かりやすく回答してください：\n\n{input}",
    "創造的な執筆": "あなたは創造的な作家です。以下のお題に基づいて創造的な文章を書いてください：\n\n{input}",
    "カスタム": "{input}"
}

template_key = st.sidebar.selectbox(
    "プロンプトテンプレート",
    list(prompt_templates.keys()),
    index=0
)

# カスタムテンプレートの場合のみテキストエリアを表示
if template_key == "カスタム":
    custom_template = st.sidebar.text_area(
        "カスタムテンプレート", 
        "以下の質問に答えてください：\n\n{input}",
        height=150
    )
    template = custom_template
else:
    template = prompt_templates[template_key]

# チャット履歴の表示/非表示設定
show_history = st.sidebar.checkbox("チャット履歴を表示", value=True)

# チャット履歴の保存と読み込み機能
st.sidebar.header("チャット履歴")

# 履歴の保存
if st.sidebar.button("現在の会話を保存"):
    if "messages" in st.session_state and st.session_state.messages:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"
        
        # チャット履歴をJSONとして保存
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
        
        st.sidebar.success(f"会話を保存しました: {filename}")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# メインチャットエリア
if show_history:
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
    
    # テンプレートを適用
    formatted_prompt = template.format(input=prompt)
    
    # AIの応答を生成
    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            response = get_gpt_response(formatted_prompt)
            st.markdown(response)
    
    # AIの応答をチャット履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": response})

# チャット履歴のクリアボタン
if st.button("チャット履歴をクリア"):
    st.session_state.messages = []
    st.rerun()