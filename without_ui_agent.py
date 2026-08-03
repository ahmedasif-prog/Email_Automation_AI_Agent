from typing import List, Dict, TypedDict, Sequence, Annotated
import groq
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import os
import smtplib
from email.mime.text import MIMEText

document_content = ""
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASS = os.environ.get("EMAIL_PASSWORD")
class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
@tool
def update_text(content:str)->str:
    """Updates the document with provided content"""
    global document_content
    document_content = content
    return f"Document has been updated successfully!\nThe current content is:\n{document_content}"
@tool
def save_content(file_name:str)->str:
    """Saves the current document to a text file and finish the process
    Args:
        file_name:Name for text file
    """
    global document_content
    try:
        with open(file_name,'w') as file:
            file.write(document_content)
        print(f"Document has been saved to {file_name}")
        return f"Document has been saved to {file_name}"
    except Exception as e:
        print(f"Error in saving the file:{str(e)}")
    if not file_name.endswith('.txt'):
        file_name = f"{file_name}.txt"
@tool
def send_email(to:str,subject:str,body:str)->str:
    """Sends the email to the address the user provided"""
    try:
        msg=MIMEText(body)
        msg["Subject"]=subject
        msg["From"]=EMAIL_ADDRESS
        msg["To"]=to
        
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as server:
            server.login(EMAIL_ADDRESS,EMAIL_PASS)
            server.send_message(msg)
        return f"Email successfully sent to {to}"
    except Exception as e:
        return f"Error in sending Email:{e}"
tools = [update_text,save_content,send_email]
llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model="llama-3.3-70b-versatile", temperature=0.7).bind_tools(tools)
def our_agent(state:AgentState)->AgentState:
    system_prompt = f"""
You are an email, ahelpful writing assistant, you are going to help save update and modify emails.
-If the user wants to update or modify email use the 'update_text' tool with complete updated content.
-If the user wants to save  use the 'save_content' tool.
-Make sure to show the current document state after modification
-After saving when the user tells tou to send the email first use 'save_content' 'send_email' tool
The current document content is:{document_content}
"""
    if not state["messages"]:
        user_input = "I am ready to help you update a document. What would you like to do with the document"
        user_message = HumanMessage(content = user_input)
    else:
        user_input = input("What would you like me to do with the document? ")
        print(f"User:{user_input}")
        user_message = HumanMessage(content=user_input)
    all_messages = [SystemMessage(content=system_prompt)] + list(state["messages"]) + [user_message]
    response = llm.invoke(all_messages)
    print(f"\nAI:{response.content}")
    # if hasattr(response,"tool_calls") and response.tool_calls:
    #     print(f"Using Tools:{[tc['name']]}")
    return {"messages":list(state["messages"])+[user_message,response]}
def should_continue(state:AgentState)->AgentState:
    messages = state["messages"]
    if not messages:
        return "continue"
    for message in reversed(messages):
        if(isinstance(message,ToolMessage)) and "saved" in message.content.lower() and "document" in message.content.lower():
            return "end"
    return "continue"
def print_messages(messages):
    if not messages:
        return
    for message in messages[-3:]:
        if(isinstance(message,ToolMessage)):
            print(f"\nTool Result:{message.content}")
graph = StateGraph(AgentState)
graph.add_node("Agent",our_agent)
graph.set_entry_point("Agent")
graph.add_node("tools",ToolNode(tools))
graph.add_edge("Agent","tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue":"Agent",
        "end":END
    }
)
agent = graph.compile()
def run_agent():
    print("\n======Drafter======")
    state = {"messages":[]}
    for step in agent.stream(state,stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n======Drafter Finished======")
if __name__ == "__main__":
    run_agent()
        
