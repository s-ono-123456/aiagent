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

# 状態の型定義
class GraphState(TypedDict):
    messages: List[Any]  # HumanMessageやAIMessageのリスト
    query: str
    thoughts: str
    response: str
    agent_thoughts: List[Dict[str, str]]  # 各エージェントの思考過程を保存するリスト

# エージェントのLLMモデル
def get_agent_llm(model_name="gpt-4.1-nano"):
    return ChatOpenAI(
        model=model_name,
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def run_async_in_sync(async_func, *args, **kwargs):
    """非同期関数を同期的に実行するヘルパー関数"""
    # Windowsの場合はProactorEventLoopを使用
    if os.name == 'nt':
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(async_func(*args, **kwargs))
    finally:
        loop.close()

def create_planner_agent():
    """プランナーエージェントの作成"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは計画立案エージェントです。
        ユーザーのクエリを分析し、適切なテスト実行計画を立ててください。
        複雑なクエリの場合は、複数のステップに分けて計画を立てることができます。"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", """
         本システムは、ユーザの入力に基づいて、検索処理を行うシステムです。
         ユーザのクエリに基づいて、適切なテスト実行計画を立ててください。
         クエリ: 
         {query}
         
         これらの情報をもとに、テスト実行計画を立ててください。""")
    ])
    
    llm = get_agent_llm()
    
    def _planning_chain(state):
        planning_input = {
            "messages": state["messages"],
            "query": state["query"],
        }
        
        plan = prompt | llm | StrOutputParser()
        plan_result = plan.invoke(planning_input)
        
        thoughts = f"計画: {plan_result}"
        
        # 思考過程を状態オブジェクトに保存
        state["agent_thoughts"].append({
            "agent": "planner",
            "thought": thoughts
        })
        print(f"Planning thoughts: {thoughts}")
        return {
            "thoughts": thoughts,
            "agent_thoughts": state.get("agent_thoughts", [])
        }

    return _planning_chain

def create_test_executor_agent(tools):
    """テスト実行エージェントの作成"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたはテスト実行エージェントです。
        あなたは、「Playwright」というブラウザを操作するツールを利用することができます。
        ユーザーのクエリに基づいて、適切なテストを実行してください。
        テストを実行するための具体的な手順を考えてください。"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", """クエリ: 
         {query}
         
         計画：
         {thoughts}

         これまで実施した内容：
         {agent_thoughts}
         
         これらの情報をもとに、テストを実行してください。""")
    ])
    
    llm = get_agent_llm().bind_tools(tools)
    
    def _test_executor_chain(state):
        test_input = {
            "query": state["query"],
            "messages": state["messages"],
            "thoughts": state["thoughts"],
            "agent_thoughts": state.get("agent_thoughts", [])
        }
        
        test_result = prompt | llm | StrOutputParser()
        test_response = test_result.invoke(test_input)
        
        thoughts = f"テスト結果: {test_response}"
        
        # 思考過程を状態オブジェクトに保存
        state["agent_thoughts"].append({
            "agent": "test_executor",
            "thought": thoughts
        })
        print(f"Test executor thoughts: {thoughts}")
        
        return {
            "thoughts": thoughts,
            "agent_thoughts": state.get("agent_thoughts", [])
        }

    return _test_executor_chain

def create_screenshot_agent(tools):
    """スクリーンショットエージェントの作成"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたはスクリーンショットエージェントです。
        あなたは、「Playwright」というブラウザを操作するツールを利用することができます。
        ユーザーのクエリに基づいて、適切なスクリーンショットを取得してください。
        スクリーンショットを取得するための具体的な手順を考えてください。"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", """クエリ: 
         {query}

         いままでの情報：
         {agent_thoughts}
         
         これらの情報をもとに、スクリーンショットを取得してください。""")
    ])
    
    llm = get_agent_llm().bind_tools(tools)
    
    def _screenshot_chain(state, tools):
        screenshot_input = {
            "query": state["query"],
            "agent_thoughts": state["agent_thoughts"],
            "messages": state["messages"],
            "thoughts": state["thoughts"]
        }
        
        screenshot = prompt | llm | StrOutputParser()
        screenshot_result = screenshot.invoke(screenshot_input)
        
        thoughts = f"スクリーンショット: {screenshot_result}"
        
        # 思考過程を状態オブジェクトに保存
        state["agent_thoughts"].append({
            "agent": "screenshot",
            "thought": thoughts
        })
        
        return {
            "thoughts": thoughts,
            "agent_thoughts": state.get("agent_thoughts", []),
        }

    return _screenshot_chain


def create_graph(state: GraphState, tools):
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
    
    # def call_model(state):
    #     """モデルを呼び出す関数"""
    #     messages = state["messages"]
    #     response = model_chain.invoke(messages)
    #     return {"messages": [response]}
    
    # def take_screenshot(state):
    #     """ツールノードでアクセスした画面をスクリーンショットAIに渡す関数"""
    #     messages = state["messages"]
    #     last_message = messages[-1]
    #     print(f"Last message: {last_message}")

    #     query = "現在の画面をスクリーンショットとして保存してください。"
    #     response = model_chain.invoke(query)
    #     return {"messages": [response]}
    
    # ツールノードの作成
    tool_node = ToolNode(tools)
    screenshottool = ToolNode(tools)
    
    # グラフの作成
    workflow = StateGraph(state)
    workflow.add_node("planning", create_planner_agent())
    workflow.add_node("testing", create_test_executor_agent(tools))
    workflow.add_node("tools", tool_node)
    workflow.add_node("screenshotagent", create_screenshot_agent(tools))
    workflow.add_node("screenshottool", screenshottool)
    
    # エッジの追加
    workflow.add_edge(START, "planning")
    workflow.add_edge("planning", "testing")
    workflow.add_edge("tools", "screenshotagent")
    workflow.add_edge("screenshotagent", "screenshottool")
    workflow.add_edge("screenshottool", "testing")
    workflow.add_conditional_edges("testing", should_continue, {
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
    
    # MCPの設定を読み込む
    with open("mcp_config.json", "r") as f:
        mcp_config = json.load(f)
    
    # 初期状態を設定
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "thoughts": "",
        "response": "",
        "agent_thoughts": [],  # 各エージェントの思考過程を保存するリスト
    }

    # プロンプトの作成
    # prompt = ChatPromptTemplate.from_messages(message)
    
    # MCPクライアントの作成と実行
    async with MultiServerMCPClient(mcp_config["mcpServers"]) as mcp_client:
        # ツールの取得
        tools = mcp_client.get_tools()
        
        # モデルにツールをバインド
        # model_with_tools = prompt | model.bind_tools(tools)
        
        # グラフの作成
        graph = create_graph(GraphState, tools)
        
        # グラフの実行
        graph_config = {"configurable": {"thread_id": "12345"}}
        response = await graph.ainvoke(initial_state, config=graph_config)
        # print(f"Response: {response}")
        # レスポンスからスクリーンショットを抽出して保存
        for message in response["messages"]:
            # print(f"Message: {message}")
            print("")
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
                            filename = f"screenshots/screenshot_{timestamp}.png"
                            
                            # Base64デコードしてファイルに保存
                            with open(filename, "wb") as f:
                                f.write(base64.b64decode(artifact.data))
                            
                            print(f"スクリーンショットを保存しました: {filename}")

        # 結果を返す
        return response["messages"][-1].content

def run_mcp_agent(query):
    """ThreadPoolExecutorを使用して非同期関数を同期的に実行するラッパー関数"""
    return run_async_in_sync(run_mcp_agent_async, query)