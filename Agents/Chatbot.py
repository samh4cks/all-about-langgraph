## Memory-Agent.py

import os
from typing import List, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # Used to store secret stuff like credentials, API keys, etc. in a .env file

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union [HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4.1-nano")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}\n")
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter the message: ")
while user_input != "exit" or user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]   
    user_input = input("Enter the message: ")

with open("Conversation_History.txt", "w") as file:
    file.write("Your Conversation Log:\n")

    for messsage in conversation_history:
        if isinstance(messsage, HumanMessage):
            file.write(f"User: {messsage.content}\n")
        elif isinstance(messsage, AIMessage):
            file.write(f"AI: {messsage.content}\n")
    file.write("End of Conversation\n")

print("Conversation history saved to Conversation_History.txt")