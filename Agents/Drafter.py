from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core. messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# This is the global variable to store document content

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """This is a tool function that updates the document content with the provided content"""

    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is: \n{document_content}"

@tool
def save(filename: str) -> str:
    """Saves the current document content to a file with the provided filename

    Args:
        filename: Name for the text file
    """
    
    global document_content

    if not filename.endswith(".txt"):
        filename += ".txt"
    
    try:
        with open(filename, "w") as file:
            file.write(document_content)
            print(f"Document content has been saved to {filename}")
            return f"Document content has been saved successfully to ''{filename}'."
        
    except Exception as e:
        return f"An error occurred while saving the document: {str(e)}"
    
tools = [update, save]

model = ChatOpenAI(model="gpt-4.1-nano").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

    - If the user wnats to update or modify content, use the 'update' tool with the completed updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
                                  
    The current document content is: {document_content}
    """
    )

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("What would you like to do with the document? ")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f'\nDrafter: {response.content}\n')
    if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}\n")

    return {"messages" : list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Continue the loop until the user says 'exit'"""

    messages = state["messages"]

    if not messages:
        return "continue"
    
    # This looks for the most recent tool message......
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower():
            return "end"
    
    return "continue"
    
def print_messages(messages):
    """Function I made to print the messages in a more readable format"""

    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\nTOOL MESSAGE: {message.content}\n")

graph = StateGraph(AgentState)

graph.add_node("our_agent", model_call)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("our_agent")
graph.add_edge("our_agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "our_agent",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():
    print("======== DRAFTER ========\n")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
        
    print("\nThank you for using Drafter! Goodbye!\n")

if __name__ == "__main__":
    run_document_agent()