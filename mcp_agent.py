import os
import json
import asyncio
from typing import Annotated, TypedDict, List, Dict, Any, Literal
import operator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import time
import base64

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
        # print(last_message.tool_calls)
        # ツール呼び出しがある場合はツールノードへ、そうでなければ終了
        return_value = "tools" if last_message.tool_calls else END
        print(f"Return value: {return_value}")
        return return_value
    
    def call_model(state):
        """モデルを呼び出す関数"""
        last_message = state["messages"][-1]
        print(f"Last message: {last_message}")
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
また、操作を行った後は「browser_screen_capture」のツールを利用してスクリーンショットを撮影してください。

ユーザーの質問からツールをどのような意図で何回利用する必要があるかを判断し、必要なら複数回ツールを利用して情報収集をした後、
すべての情報が取得できたら、その情報を元に返答してください。
"""),
        MessagesPlaceholder("messages"),
    ]
    
    # プロンプトの作成
    prompt = ChatPromptTemplate.from_messages(message)
    session_id = f"session_{int(time.time())}"
    
    # MCPクライアントの作成と実行
    async with MultiServerMCPClient(
        mcp_config["mcpServers"]
    ) as mcp_client:
        # Playwrightの設定を構成
        playwright_config = {
            "browser_session_id": session_id,
            "user_data_dir": f"./browser_data_{session_id}",
            "headless": True,
            "timeout": 600000  # タイムアウトを600秒に設定
        }
        
        # 設定をクライアントに適用（必要に応じて）
        if hasattr(mcp_client, "configure"):
            await mcp_client.configure("playwright", playwright_config)
        
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
        # print(f"Response: {response}")
        # レスポンスからスクリーンショットを抽出して保存
        num = 0
        for message in response["messages"]:
            # print(f"Message: {message}")
            print("")
            num += 1
            if isinstance(message, ToolMessage):
                artifacts = message.artifact
                if artifacts:
                    print(f"artifact num: {len(artifacts)}")
                    for i, artifact in enumerate(artifacts):
                        if artifact.data:
                            # 一時的なスクリーンショット保存用ディレクトリの作成
                            os.makedirs("screenshots", exist_ok=True)
                            
                            # ファイル名の作成（タイムスタンプ付き）
                            timestamp = int(time.time())
                            filename = f"screenshots/screenshot_{timestamp}_{num}.png"
                            
                            # Base64デコードしてファイルに保存
                            with open(filename, "wb") as f:
                                f.write(base64.b64decode(artifact.data))
                            
                            print(f"スクリーンショットを保存しました: {filename}")

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