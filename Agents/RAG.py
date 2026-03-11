from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI 
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# Our Embedding Model - has to be compatible with the LLM

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pdf_path = "7_Day_Time_Based_Vegetarian_Diet_Walk_and_Routine_Plan.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path) # This loads the PDF

# Checks if the PDF is there
try:
    pages = pdf_loader.load() # This splits the PDF into pages
    print(f"Successfully loaded PDF with {len(pages)} pages.")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise 

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
    )

pages_split = text_splitter.split_documents(pages) # We now apply this to our all pages


persist_directory = "chroma_db"

collection_name = "diet-plan"


if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # Here, we actually create the chroma database using our embeddings model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromDB vector store!")

except Exception as e:
    print(f"Error creating vector store: {e}")
    raise

# Now we create our retriever
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the 7 Day Time Based Vegetarian Diet Walk and Routine Plan PDF based on the user's query.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "Sorry, I couldn't find any relevant information in the document for your query."

    results = []

    for i, doc in enumerate(docs):
        results.append(f"Result {i+1}:\n{doc.page_content}\n")

    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> str:
    """Check if the last message contains tool calls"""

    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

system_prompt = SystemMessage(content=f"""
You are an intelligent fitness AI assistant who makes the diet plan and workout plan for the user based on the information
from the 7 Day Time Based Vegetarian Diet Walk and Routine Plan PDF. You need to understand the current diet plan and workout 
plan of the user and then according to the current diet, suggest a plan which is comfortable for the user. If you don't find
any relevant details regarding user query, please feel free to make a diet plan and workout plan for the user based on the information you have.
""")

tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictonary of our tools

def call_llm(state: AgentState):
    """Function to call the LLM with the current state."""

    messages = list(state["messages"])
    messages = [system_prompt] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}

graph = StateGraph(AgentState)

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
graph.add_edge("llm", END)

agent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()