import streamlit as st
import os
import json
import asyncio
from typing import Annotated, TypedDict, List, Dict, Any, Literal
import operator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 環境変数の読み込み
openai_api_key = os.getenv("OPENAI_API_KEY")

# 状態の型定義
class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

def create_graph(state: GraphState, tools, model_chain):
    """LangGraphによるエージェントグラフの作成"""
    
    def should_continue(state):
        """次のノードを決定する関数"""
        messages = state["messages"]
        last_message = messages[-1]
        print(f"Last message: {last_message}")
        print(last_message.tool_calls)
        # ツール呼び出しがある場合はツールノードへ、そうでなければ終了
        return_value = "tools" if last_message.tool_calls else END
        print(f"Return value: {return_value}")
        return return_value
    
    def call_model(state):
        """モデルを呼び出す関数"""
        messages = state["messages"]
        response = model_chain.invoke(messages)
        return {"messages": [response]}
    
    # ツールノードの作成
    tool_node = ToolNode(tools)
    
    # グラフの作成
    workflow = StateGraph(state)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # エッジの追加
    workflow.add_edge(START, "agent")
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        END: END
    })
    
    # メモリの設定
    memory = MemorySaver()
    
    # グラフのコンパイル
    app = workflow.compile(checkpointer=memory)
    return app

async def run_mcp_agent_async(query):
    """MCPエージェントを実行する非同期関数"""
    
    # モデルの定義（OpenAIのgpt-4.1-nanoを使用）
    model = ChatOpenAI(
        model="gpt-4.1-nano",
        openai_api_key=openai_api_key,
        temperature=0.2,
    )
    
    # MCPの設定を読み込む
    with open("mcp_config.json", "r") as f:
        mcp_config = json.load(f)
    
    # システムメッセージとプロンプトの設定
    message = [
        SystemMessage(content="""
あなたは役に立つAIアシスタントです。日本語で回答し、考えた過程を結論より前に出力してください。
あなたは、「Playwright」というブラウザを操作するツールを利用することができます。適切に利用してユーザーからの質問に回答してください。
ツールを利用する場合は、必ずツールから得られた情報のみを利用して回答してください。

ユーザーの質問からツールをどのような意図で何回利用する必要があるかを判断し、必要なら複数回ツールを利用して情報収集をした後、
すべての情報が取得できたら、その情報を元に返答してください。

なお、サイトのアクセスでエラーが出た場合は、もう一度再試行してください。ネットワーク関連のエラーの場合があります。
"""),
        MessagesPlaceholder("messages"),
    ]
    
    # プロンプトの作成
    prompt = ChatPromptTemplate.from_messages(message)
    
    # MCPクライアントの作成と実行
    async with MultiServerMCPClient(mcp_config["mcpServers"]) as mcp_client:
        # ツールの取得
        tools = mcp_client.get_tools()
        
        # モデルにツールをバインド
        model_with_tools = prompt | model.bind_tools(tools)
        
        # グラフの作成
        graph = create_graph(GraphState, tools, model_with_tools)
        
        # 入力クエリの作成
        input_query = [HumanMessage(content=query)]
        
        # グラフの実行
        graph_config = {"configurable": {"thread_id": "12345"}}
        response = await graph.ainvoke({"messages": input_query}, graph_config)
        print(f"Response: {response}")

        # 結果を返す
        return response["messages"][-1].content

def run_mcp_agent(query):
    """ThreadPoolExecutorを使用して非同期関数を同期的に実行するラッパー関数"""
    # 新しいループを作成
    # Windows環境での非同期処理のためにProactorEventLoopを使用
    if os.name == 'nt':
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()

    # ループをセット
    asyncio.set_event_loop(loop)
    
    # ThreadPoolExecutorを使用して別スレッドで非同期関数を実行
    with ThreadPoolExecutor() as executor:
        future = executor.submit(lambda: loop.run_until_complete(run_mcp_agent_async(query)))
        return future.result()

# Streamlitアプリのタイトル設定
st.title("Playwright MCPエージェント")

# 説明文を追加
st.write("このアプリはPlaywright MCPを使用してウェブブラウザを操作し、質問に答えるAIエージェントです。")
st.write("URLを含む質問を入力すると、AIがブラウザでWebサイトを閲覧し、情報を収集して回答します。")

# サンプル質問
st.subheader("サンプル質問")
st.code("https://zenn.dev/asap/articles/59b8dd06d44754を読んで要約して")
st.code("https://github.com/microsoft/playwright-mcpの機能について教えて")

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
            # ThreadPoolExecutorを使用した同期処理に変更
            result = run_mcp_agent(query)
            st.session_state.result = result
        
        # 処理中フラグをオフ
        st.session_state.processing = False

# 結果の表示
if st.session_state.result:
    st.subheader("回答:")
    st.markdown(st.session_state.result)