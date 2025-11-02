from langgraph.graph import StateGraph, START ,END
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage,SystemMessage

class AgentState(TypedDict):
    messages : list

def chat_node(state:AgentState)->AgentState:
    llm:ChatOllama=ChatOllama(model="qwen3:1.7b")

    messages: list =state["messages"]
    chat_response = llm.invoke(messages)

    state["messages"].append(chat_response)
    return state

graph_builder = StateGraph(AgentState)

graph_builder.add_node("chat_node",chat_node)
graph_builder.add_edge(START,"chat_node")
graph_builder.add_edge("chat_node",END)

graph = graph_builder.compile()

graph_image: bytes= graph.get_graph().draw_mermaid_png()

with open("graph.png","wb") as f:
    f.write(graph_image)

initial_state = AgentState(messages=[])
initial_state["messages"].append(HumanMessage(content="Hello, how are you?"))

response = graph.invoke(initial_state)
print(response)


