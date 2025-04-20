import os
import json
from typing import Annotated, TypedDict, List, Dict, Any, Literal
import operator
import asyncio
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters import MultiServerMCPClient
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 環境変数の読み込み
load_dotenv()
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
        
        # ツール呼び出しがある場合はツールノードへ、そうでなければ終了
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
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
    workflow.add_edge("agent", "tools")
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

async def main():
    """メイン関数"""
    
    # モデルの定義（Google AI Studioのgemini-1.5-proを使用）
    model = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY")
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
        
        # 対話ループ
        while True:
            query = input("質問を入力してください（終了するには 'exit' と入力）: ")
            
            if query.lower() in ["exit", "quit"]:
                print("終了します。")
                break
            
            # 入力クエリの作成
            input_query = [HumanMessage(content=query)]
            
            # グラフの実行
            graph_config = {"configurable": {"thread_id": "12345"}}
            response = await graph.ainvoke({"messages": input_query}, graph_config)
            
            # 結果の表示
            print("\n=== 回答 ===")
            print(response["messages"][-1].content)
            print("===========\n")

if __name__ == "__main__":
    asyncio.run(main())