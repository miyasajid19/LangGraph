# -------------
# importing liraries
# -------------

import os
import json
from pyexpat.errors import messages
from dotenv import load_dotenv
from typing import TypedDict
from imap_tools import MailBox, AND
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph,START,END
from operator import add
import smtplib
from email.message import EmailMessage
# -------------
# loading environment variables
# -------------
load_dotenv()


IMAP_HOST=os.getenv("IMAP_SERVER")
IMAP_PORT=int(os.getenv("IMAP_PORT"))
EMAIL=os.getenv("EMAIL")
APP_PASSWORD=os.getenv("APP_PASSWORD")
IMAP_FOLDER="INBOX"


SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))


CHAT_OLLAMA_MODEL="gpt-oss:120b-cloud"

class ChatState(TypedDict):
    messages: list[str,add]


# -------------
# connector to mailbox
# -------------
def connect_mailbox():
    mail_box=MailBox(IMAP_HOST)
    mail_box.login(EMAIL,APP_PASSWORD,initial_folder=IMAP_FOLDER)
    return mail_box


# -------------
# Tool 1 : Fetch Latest Unread Emails
# -------------
@tool
def fetch_latest_unread_emails_tool() -> str:
    """
    Docstring for fetch_latest_unread_emails_tool
    
    :return: Description
    :rtype: str
    """
    print("Fetching latest unread emails...")
    with connect_mailbox() as mail_box:
        messages=list(mail_box.fetch(AND(seen=False),limit=5,reverse=True))
        print(f"Fetched {len(messages)} unread emails.")
    
    if not messages:
        return "No unread emails found."
    
    response=json.dumps(
        [
            {
                "subject":msg.subject,
                "from":msg.from_,
                "date":str(msg.date),
                "body":msg.text
            }
            for msg in messages
        ],
        indent=4
    )
    print("Formatted fetched emails as JSON.")
    return response


# -------------
# Tool 2 : Summarizer
# -------------
@tool
def summarize_emails_tool(uid) -> str:
    """
    Docstring for summarize_emails_tool
    
    :param messages: Description
    :type messages: list[dict]
    :return: Description
    :rtype: str
    """
    print(f"Summarizing {uid} emails...")
    with connect_mailbox() as mail_box:
        mail=next(mail_box.fetch(AND(uid=uid),limit=1),None)
        
        if not mail:
            return f"No email found with UID {uid}."
        
        prompt=f"""
        Summarize the following email content in concise bullet points:
        Subject: {mail.subject}
        From: {mail.from_}
        Date: {mail.date}
        Body: {mail.text or mail.html}
        Note: Focus on key information and main points.
        """
        
        return raw_model.invoke(prompt)


#-------------
# Tool 3: Tavily Search Results
#-------------
TAVILY_API_KEY = os.getenv("tavily_api_key")
@tool
def tavily_search_tool(query: str) -> str:
    """
    Docstring for tavily_search_tool
    
    :param query: Description
    :type query: str
    :return: Description
    :rtype: str
    """
    print("Performing Tavily search...")
    tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
    result_list = tavily_tool.invoke(query)  # Returns LIST of dicts

    formatted_results = ""
    for r in result_list:
        formatted_results += f"Title: {r['title']}\n"
        formatted_results += f"Content: {r['content'][:200]}...\n"
        formatted_results += f"URL: {r['url']}\n"
        formatted_results += "-----\n"
    print("Tavily search completed.")
    return formatted_results 


# -------------
# Tool 4 : Mail Sender (imported from mail_sender.py)
# -------------
@tool
def send_email_tool(to_email: str, subject: str, body: str) -> str:
    """
    Docstring for send_email_tool
    
    :param to_email: Description
    :type to_email: str
    :param subject: Description
    :type subject: str
    :param body: Description
    :type body: str
    :return: Description
    :rtype: str
    """
    print(f"Sending email to {to_email}...")
    message=EmailMessage()
    message["From"]=EMAIL
    message["To"]=to_email
    message["Subject"]=subject
    message.set_content(body)
    try:
        with smtplib.SMTP(SMTP_SERVER,SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL,APP_PASSWORD)
            server.send_message(message)
        print(f"Email sent successfully to {to_email}.")
        return f"Email sent successfully to {to_email}."
    except Exception as e:
        return f"Failed to send email. Error: {str(e)}"
# -------------
# Setting up the Language Model
# -------------

raw_model=ChatOllama(model=CHAT_OLLAMA_MODEL)
model_with_tools=raw_model.bind_tools([fetch_latest_unread_emails_tool,summarize_emails_tool,tavily_search_tool,send_email_tool])


# -------------
# Building the Graph
# -------------

def llm_node(ChatState):
    response=model_with_tools.invoke(ChatState["messages"])
    return {"messages":ChatState["messages"] + [response]}

def router_node(ChatState):
    last_message=ChatState["messages"][-1]
    return 'tools' if getattr(last_message,'tool_calls',None) else 'end'

tool_node=ToolNode([fetch_latest_unread_emails_tool,summarize_emails_tool,tavily_search_tool,send_email_tool])

builder=StateGraph(ChatState)

builder.add_node("llm",llm_node)
builder.add_node("router",router_node)
builder.add_node("tools",tool_node)

builder.add_edge(START,"llm")
builder.add_edge("tools","llm")
builder.add_conditional_edges('llm',router_node,{'tools':'tools','end':END})


graph=builder.compile()


if __name__=="__main__":
    state= {'messages':[]}
    
    while True:
        user_input=input("User: ")
        if user_input.lower() in ['exit','quit']:
            break
        state['messages'].append(user_input)
        state=graph.invoke(state)
        print(f"Agent: {state['messages'][-1].content}")
    