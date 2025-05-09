# aiagent

## 概要
このアプリは、LangGraphを使用したマルチエージェント型RAG（検索拡張生成）システムを実装したStreamlitアプリケーションです。複数の専門エージェント（検索エージェント、計画立案エージェント、回答生成エージェント）が協力して、質問に対する最適な回答を生成します。また、シンプルなLLM対話機能も搭載しています。

## 主な機能
- シンプルLLM対話: 直接LLMに質問を投げるシンプルな対話機能
- マルチエージェントRAG: 複数のエージェントが協力して質問に回答する高度な機能
- ドキュメント追加: RAGシステムが使用するナレッジベースにドキュメントを追加する機能
- エージェント思考過程の表示: 各エージェントの思考過程を確認できる機能

## ファイル構成
```
aiagent/
├── agents.py          # マルチエージェントRAGシステムの実装
├── app.py             # Streamlitアプリのメイン実装
├── document_store.py  # ドキュメントストアとベクトル検索の実装
├── llm_client.py      # LLMとの通信クライアント
├── aiagent.code-workspace
├── README.md
├── requirements.txt
└── __pycache__/       # コンパイル済みPythonファイル
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
### シンプルLLM対話
- 「シンプルLLM」タブで、テキストボックスに質問を入力し、エンターキーを押すとAIからの応答が表示されます。

### マルチエージェントRAG
- 「マルチエージェントRAG」タブで質問を入力すると、複数のエージェントが協力して回答を生成します。
- 「エージェントの思考過程を表示」チェックボックスをオンにすると、各エージェントの思考過程を確認できます。
- 「ドキュメントの追加」セクションで新しいドキュメントを追加することで、RAGシステムのナレッジベースを拡張できます。

## 技術詳細
- LangGraph: 複数のエージェントを組み合わせたワークフローを定義
- LangChain: LLMアプリケーションの構築フレームワーク
- FAISS: 高速ベクトル検索ライブラリ
- HuggingFace Embeddings: ドキュメントとクエリの埋め込みに使用
- Streamlit: Webインターフェースの実装

## ワークフロー
マルチエージェントRAGシステムは以下のエージェントで構成されています：

1. 検索エージェント（Retriever）: クエリを分析し、関連するドキュメントを検索
2. 計画立案エージェント（Planner）: 検索結果を基に回答計画を立案
3. 回答生成エージェント（Responder）: 計画と検索結果を基に最終回答を生成
