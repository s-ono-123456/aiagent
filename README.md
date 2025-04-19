# aiagent

## 概要
このアプリは、Streamlitを用いてOpenAIのGPT-4.1-nanoモデルに質問できるシンプルなWebアプリケーションです。エンターキーで質問を送信すると、AIからの応答が表示されます。

## ファイル構成
```
ragapp/
├── app.py
├── llm_client.py
├── ragapp.code-workspace
├── README.md
├── requirements.txt
└── __pycache__/
    └── llm_client.cpython-310.pyc
```

## 必要条件
- Python 3.7以上
- OpenAI APIキー（環境変数 `OPENAI_API_KEY` に設定）

## セットアップ手順
1. 必要なパッケージのインストール
   ```bash
   pip install -r requirements.txt
   ```
2. OpenAI APIキーを環境変数に設定
   ```bash
   set OPENAI_API_KEY=sk-...   # Windowsの場合
   export OPENAI_API_KEY=sk-... # Mac/Linuxの場合
   ```
3. アプリの起動
   ```bash
   streamlit run app.py --server.fileWatcherType none
   ```


## 使い方
- テキストボックスに質問を入力し、エンターキーを押すとAIの応答が表示されます。
