from typing import Dict, List, Annotated, TypedDict, Literal, Any
import os
import operator
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, Runnable
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from document_store import DocumentStore
from IPython.display import Image, display

os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]="pr-virtual-clay-8"
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# 状態の型定義
class AgentState(TypedDict):
    messages: List[Any]  # HumanMessageやAIMessageのリスト
    query: str
    documents: List[Any]  # 検索されたドキュメント
    thoughts: str
    response: str
    agent_thoughts: List[Dict[str, str]]  # 各エージェントの思考過程を保存するリスト
    next: Literal["retriever", "planner", "responder", "end"]

# エージェントのLLMモデル
def get_agent_llm(model_name="gpt-4.1-nano"):
    return ChatOpenAI(
        model=model_name,
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# 検索エージェント（Retriever）
def create_retriever_agent(document_store: DocumentStore):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたはリサーチアシスタントです。
        ユーザーのクエリに関連する情報を検索し、関連するドキュメントを見つける役割を持っています。
        ユーザーのクエリを分析して、検索に最適なキーワードと質問を考えてください。"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "クエリ: {query}\n検索に最適なキーワードや質問はなんですか？")
    ])
    
    llm = get_agent_llm()
    
    @tool
    def search_documents(query: str) -> List[Document]:
        """クエリに関連するドキュメントを検索する"""
        return document_store.search(query)
    
    def _retrieval_chain(state):
        llm_response = prompt | llm | StrOutputParser()
        search_query = llm_response.invoke(state)
        docs = search_documents.invoke(search_query)
        
        thoughts = f"検索クエリ '{search_query}' で{len(docs)}件のドキュメントを検索しました。"
        
        # 思考過程を状態オブジェクトに保存
        state["agent_thoughts"].append({
            "agent": "retriever",
            "thought": thoughts
        })
        
        return {
            "documents": docs,
            "thoughts": thoughts,
            "agent_thoughts": state.get("agent_thoughts", []),
            "next": "planner"
        }
    
    return _retrieval_chain

# プランナーエージェント（Planner）
def create_planner_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは計画立案エージェントです。
        ユーザーのクエリと検索されたドキュメントを分析し、最適な回答計画を立ててください。
        複雑な質問は分解し、回答に必要なステップを考えてください。"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", """クエリ: {query}
        
        検索結果:
        {documents}
        
        これらの情報を基に、どのように回答すべきか計画を立ててください。""")
    ])
    
    llm = get_agent_llm()
    
    def _planning_chain(state):
        # ドキュメントをテキスト形式に変換
        docs_str = "\n".join([f"ドキュメント {i+1}: {doc.page_content}" for i, doc in enumerate(state["documents"])])
        
        planning_input = {
            "messages": state["messages"],
            "query": state["query"],
            "documents": docs_str
        }
        
        plan = prompt | llm | StrOutputParser()
        plan_result = plan.invoke(planning_input)
        
        thoughts = f"計画: {plan_result}"
        
        # 思考過程を状態オブジェクトに保存
        state["agent_thoughts"].append({
            "agent": "planner",
            "thought": thoughts
        })
        
        return {
            "thoughts": thoughts,
            "agent_thoughts": state.get("agent_thoughts", []),
            "next": "responder"
        }
    
    return _planning_chain

# 回答生成エージェント（Responder）
def create_responder_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは高精度な応答を生成するエージェントです。
        ユーザーのクエリ、検索結果、および計画に基づいて、最適な回答を生成してください。
        回答は明確で簡潔、かつ正確でなければなりません。
        関連する情報源からの情報を引用し、事実に基づいた回答を提供してください。"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", """クエリ: {query}
        
        検索結果:
        {documents}
        
        計画:
        {thoughts}
        
        以上の情報に基づいて、最終的な回答を生成してください。""")
    ])
    
    llm = get_agent_llm()
    
    def _response_chain(state):
        # ドキュメントをテキスト形式に変換
        docs_str = "\n".join([f"ドキュメント {i+1}: {doc.page_content}" for i, doc in enumerate(state["documents"])])
        
        response_input = {
            "messages": state["messages"],
            "query": state["query"],
            "documents": docs_str,
            "thoughts": state["thoughts"]
        }
        
        response = prompt | llm | StrOutputParser()
        final_response = response.invoke(response_input)
        
        # 思考過程を状態オブジェクトに保存
        state["agent_thoughts"].append({
            "agent": "responder",
            "thought": f"最終回答を生成しました。"
        })
        
        return {
            "response": final_response,
            "agent_thoughts": state.get("agent_thoughts", []),
            "next": "end"
        }
    
    return _response_chain

# 状態の遷移ルートを決定する関数
def route(state: AgentState) -> Literal["retriever", "planner", "responder", "endagent"]:
    print(state["messages"])
    return state["next"]

# マルチエージェントRAGシステムを作成
def create_multi_agent_rag(document_store: DocumentStore = None):
    if document_store is None:
        document_store = DocumentStore()
        
        # サンプルドキュメントを追加
        sample_texts = [
            "LangGraphは、LangChainエコシステムの一部で、複雑なAIアプリケーションを構築するためのフレームワークです。",
            "LangGraphを使用すると、複数のエージェントを組み合わせたワークフローを定義できます。",
            "RAG（検索拡張生成）は、LLMに外部データを提供するテクニックです。",
            "マルチエージェントシステムでは、異なる役割を持つ複数のAIエージェントが協力して問題を解決します。",
            "エージェントは特定のタスクに最適化された専門家として機能し、より複雑な問題を解決するために協力します。"
        ]
        document_store.add_documents(sample_texts)
    
    # ワークフローグラフを作成
    workflow = StateGraph(AgentState)
    
    # 各エージェントをノードとして追加
    workflow.add_node("retriever", create_retriever_agent(document_store))
    workflow.add_node("planner", create_planner_agent())
    workflow.add_node("responder", create_responder_agent())
    workflow.add_node("endagent", lambda state: { "response": state["response"], "next": "end"})
    
    # エッジを定義（エージェント間の遷移）
    workflow.add_conditional_edges("retriever", route)
    workflow.add_conditional_edges("planner", route)
    workflow.add_conditional_edges("responder", route)
    
    # 開始ノードを設定
    workflow.set_entry_point("retriever")
    workflow.set_finish_point("endagent")

    graph = workflow.compile()
    
    # Mermaidダイアグラムを取得（コメントアウトを解除）
    mermaid_diagram = graph.get_graph().draw_mermaid()
    print("Workflow Graph (Mermaid):")
    print(mermaid_diagram)
        
    # グラフをコンパイル
    return graph

# マルチエージェントRAGを使って応答を生成する関数
def get_multi_agent_response(query: str, document_store: DocumentStore = None, show_thoughts: bool = True):
    graph = create_multi_agent_rag(document_store)
    
    # 初期状態を設定
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "documents": [],
        "thoughts": "",
        "response": "",
        "agent_thoughts": [],  # 各エージェントの思考過程を保存するリスト
        "next": "retriever"
    }
    
    # グラフを実行
    result = graph.invoke(initial_state)
    
    if show_thoughts and "agent_thoughts" in result:
        # 各エージェントの思考過程を結合
        thought_process = []
        for thought_entry in result["agent_thoughts"]:
            agent_name = thought_entry["agent"]
            thought = thought_entry["thought"]
            thought_process.append(f"【{agent_name}】: {thought}")
            
        thoughts_text = "\n\n".join(thought_process)
        
        return {
            "thoughts": thoughts_text,
            "response": result["response"]
        }
    else:
        # 思考過程なしで応答のみ返す
        return {
            "response": result["response"]
        }