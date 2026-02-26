# -------------
# importing liraries
# -------------

import keyword
import os
import json
from pyexpat.errors import messages
from dotenv import load_dotenv
from typing import TypedDict, List
from imap_tools import MailBox, AND
from imap_tools.consts import MailMessageFlags
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph,START,END
from operator import add
import smtplib
from email.message import EmailMessage
import re
import string
from nltk.stem import PorterStemmer
import joblib
import json
import pymysql
import dotenv
from urllib.parse import urlparse
import cloudinary
import cloudinary.uploader
import time
import threading
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

DATABASE=os.getenv("DATABASE")
HOST=os.getenv("HOST")
PASSWORD=os.getenv("PASSWORD")
TIMEOUT=os.getenv("TIMEOUT")
DB_PORT=os.getenv("DB_PORT")
USER=os.getenv("USER")
CLOUD_NAME=os.getenv("CLOUD_NAME")
API_KEY=os.getenv("API_KEY")
API_SECRET=os.getenv("API_SECRET")




def get_database_connection():
    timeout = int(os.getenv("TIMEOUT", 10))
    connection = pymysql.connect(
    charset="utf8mb4",
    connect_timeout=timeout,
    cursorclass=pymysql.cursors.DictCursor,
    database=os.getenv("DATABASE", "defaultdb"),
    host=os.getenv("HOST", "localhost"),
    password=os.getenv("PASSWORD", ""),
    read_timeout=timeout,
    port=int(os.getenv("DB_PORT", os.getenv("MYSQL_PORT", 3306))),
    user=os.getenv("USER", "root"),
    write_timeout=timeout,
    )
    return connection



def get_cloudinary_configuratin():
    return cloudinary.config(
        cloud_name=os.getenv("CLOUD_NAME"),
        api_key=os.getenv("API_KEY"),
        api_secret=os.getenv("API_SECRET"),
        secure=True
    )


class ChatState(TypedDict):
    messages: list[str,add]


# -------------
# connector to mailbox
# -------------
def connect_mailbox():
    mail_box=MailBox(IMAP_HOST)
    mail_box.login(EMAIL,APP_PASSWORD,initial_folder=IMAP_FOLDER)
    return mail_box


def _format_email_list(messages) -> str:
    return json.dumps(
        [
            {
                "uid": msg.uid,
                "subject": msg.subject,
                "from": msg.from_,
                "date": str(msg.date),
                "body": msg.text,
            }
            for msg in messages
        ],
        indent=4,
    )


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
        messages=list(mail_box.fetch(AND(seen=False),limit=5,reverse=True,mark_seen=False))
        print(f"Fetched {len(messages)} unread emails.")
    
    if not messages:
        return "No unread emails found."
    
    response=_format_email_list(messages)
    print("Formatted fetched emails as JSON.")
    return response


# -------------
# Tool 2 : Summarizer
# -------------
@tool
def summarize_emails_tool(uid) -> str:
    """
    Docstring for summarize_emails_tool
    
    :param uid: Unique identifier of the email to summarize
    :type uid: str
    :return: Summary of the email
    :rtype: str
    """
    print(f"Summarizing {uid} emails...")
    with connect_mailbox() as mail_box:
        mail = next(mail_box.fetch(AND(uid=uid), limit=1), None)
        
        if not mail:
            return f"No email found with UID {uid}."
        
        prompt = f"""
        Summarize the following email content in concise bullet points:
        Subject: {mail.subject}
        From: {mail.from_}
        Date: {mail.date}
        Body: {mail.text or mail.html}.
        summarize individually each email in bullet points.
        """
        
        # Mark the email as seen only if summarizing
        mail_box.flag(mail.uid, MailMessageFlags.SEEN, True)
        
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
    
    
    
    
#-------------
# Tool 5 : get mails
# -------------
@tool
def get_mails_tool() -> str:
    """
    Docstring for get_mails_tool
    
    :return: Description
    :rtype: str
    """
    print("Getting mails...")
    with connect_mailbox() as mail_box:
        messages=list(mail_box.fetch(AND(seen=False),limit=5,reverse=True))
        print(f"Fetched {len(messages)} unread emails.")
    
    if not messages:
        return "No unread emails found."
    
    response=_format_email_list(messages)
    print("Formatted fetched emails as JSON.")
    return response


# ------------ 
# Tool 6 : Search mail by keyword
# -------------
@tool
def search_mail_tool(keyword: str) -> str:
    """
    Docstring for search_mail_tool
    
    :param keyword: Description
    :type keyword: str
    :return: Description
    :rtype: str
    """
    print(f"Searching mails with keyword: {keyword}...")
    with connect_mailbox() as mail_box:
        messages=list(mail_box.fetch(AND(subject=keyword),limit=5,reverse=True,charset='UTF-8'))
        print(f"Found {len(messages)} emails with keyword '{keyword}'.")
    
    if not messages:
        return f"No emails found with keyword '{keyword}'."
    
    response=_format_email_list(messages)
    print("Formatted searched emails as JSON.")
    return response



# -------------
# Tool 7 : mails recieved from
# -------------
@tool
def mails_received_from_tool(sender_email: str) -> str:
    """
    Docstring for mails_received_from_tool
    
    :param sender_email: Description
    :type sender_email: str
    :return: Description
    :rtype: str
    """
    print(f"Fetching mails from: {sender_email}...")
    with connect_mailbox() as mail_box:
        messages=list(mail_box.fetch(AND(from_=sender_email),limit=5,reverse=True))
        print(f"Fetched {len(messages)} emails from {sender_email}.")
    
    if not messages:
        return f"No emails found from {sender_email}."
    
    response=_format_email_list(messages)
    print("Formatted fetched emails as JSON.")
    return response


# -------------
# Tool 8 : List recent emails (optionally including read)
# -------------
@tool
def list_recent_emails_tool(include_seen: bool = True, limit: int = 5) -> str:
    """
    Docstring for list_recent_emails_tool

    :param include_seen: Whether to include read emails
    :type include_seen: bool
    :param limit: Number of emails to return
    :type limit: int
    :return: Recent emails list with UID/subject/from/date/body
    :rtype: str
    """
    print(f"Listing recent emails (include_seen={include_seen}, limit={limit})...")
    with connect_mailbox() as mail_box:
        if include_seen:
            messages = list(mail_box.fetch("ALL", limit=limit, reverse=True, mark_seen=False))
        else:
            messages = list(mail_box.fetch(AND(seen=False), limit=limit, reverse=True, mark_seen=False))
        print(f"Fetched {len(messages)} emails.")

    if not messages:
        return "No emails found."

    response = _format_email_list(messages)
    print("Formatted recent emails as JSON.")
    return response


# -------------
# Tool 9 : Reply to latest email (auto-fetch + reply)
# -------------
@tool
def reply_to_latest_email_tool(reply_body: str, include_seen: bool = False) -> str:
    """
    Docstring for reply_to_latest_email_tool

    :param reply_body: Body of the reply email
    :type reply_body: str
    :param include_seen: Whether to include read emails when selecting latest
    :type include_seen: bool
    :return: Status of the reply action
    :rtype: str
    """
    print(f"Replying to latest email (include_seen={include_seen})...")
    with connect_mailbox() as mail_box:
        if include_seen:
            mail = next(mail_box.fetch("ALL", limit=1, reverse=True, mark_seen=False), None)
        else:
            mail = next(mail_box.fetch(AND(seen=False), limit=1, reverse=True, mark_seen=False), None)

        if not mail:
            return "No email found to reply to."

        to_email = mail.from_
        subject = f"Re: {mail.subject}"

        send_status = send_email_tool.invoke(
            {"to_email": to_email, "subject": subject, "body": reply_body}
        )
        return send_status


# -------------
# Tool 10 : Delete multiple emails by UID
# -------------
@tool
def delete_emails_tool(uids: List[str]) -> str:
    """
    Docstring for delete_emails_tool

    :param uids: List of email UIDs to delete
    :type uids: list[str]
    :return: Status of delete action
    :rtype: str
    """
    print(f"Deleting emails with UIDs: {uids}...")
    if not uids:
        return "No UIDs provided."

    deleted = []
    not_found = []

    with connect_mailbox() as mail_box:
        for uid in uids:
            mail = next(mail_box.fetch(AND(uid=uid), limit=1), None)
            if not mail:
                not_found.append(uid)
                continue
            mail_box.delete(mail.uid)
            deleted.append(uid)

    return json.dumps(
        {
            "deleted": deleted,
            "not_found": not_found,
        },
        indent=4,
    )
    

# -------------
# Tool 11 : Auto reply to email 
# -------------
@tool
def auto_reply_tool(uid,reply_body) -> str:
    """
    Docstring for auto_reply_tool
    
    :param uid: Unique identifier of the email to reply to
    :type uid: str
    :param reply_body: Body of the reply email
    :type reply_body: str
    :return: Status of the reply action
    :rtype: str
    """
    print(f"Auto replying to email UID {uid}...")
    with connect_mailbox() as mail_box:
        mail = next(mail_box.fetch(AND(uid=uid), limit=1), None)
        
        if not mail:
            return f"No email found with UID {uid}."
        
        to_email = mail.from_
        subject = f"Re: {mail.subject}"
        
        # Send the reply using the send_email_tool
        send_status = send_email_tool.invoke(
            {"to_email": to_email, "subject": subject, "body": reply_body}
        )
        
        return send_status
    

# -------------
# Tool 12 : Delete email by UID
# -------------
@tool
def delete_email_tool(uid) -> str:
    """
    Docstring for delete_email_tool
    
    :param uid: Unique identifier of the email to delete
    :type uid: str
    :return: Status of the delete action
    :rtype: str
    """
    print(f"Deleting email UID {uid}...")
    with connect_mailbox() as mail_box:
        mail = next(mail_box.fetch(AND(uid=uid), limit=1), None)
        
        if not mail:
            return f"No email found with UID {uid}."
        
        mail_box.delete(mail.uid)
        
        return f"Email with UID {uid} has been deleted."
    

# -------------
# Tool 13 : Mark email as read by UID
# -------------
@tool
def mark_email_as_read_tool(uid) -> str:
    """
    Docstring for mark_email_as_read_tool
    
    :param uid: Unique identifier of the email to mark as read
    :type uid: str
    :return: Status of the action
    :rtype: str
    """
    print(f"Marking email UID {uid} as read...")
    with connect_mailbox() as mail_box:
        mail = next(mail_box.fetch(AND(uid=uid), limit=1), None)
        
        if not mail:
            return f"No email found with UID {uid}."
        
        mail_box.flag(mail.uid, MailMessageFlags.SEEN, True)
        
        return f"Email with UID {uid} has been marked as read."


# -------------
# Tool 14 : Mark email as unread by UID
# -------------
@tool
def mark_email_as_unread_tool(uid) -> str:
    """
    Docstring for mark_email_as_unread_tool
    
    :param uid: Unique identifier of the email to mark as unread
    :type uid: str
    :return: Status of the action
    :rtype: str
    """
    print(f"Marking email UID {uid} as unread...")
    with connect_mailbox() as mail_box:
        mail = next(mail_box.fetch(AND(uid=uid), limit=1), None)
        
        if not mail:
            return f"No email found with UID {uid}."
        
        mail_box.flag(mail.uid, MailMessageFlags.SEEN, False)
        
        return f"Email with UID {uid} has been marked as unread."
    
    
# -------------
# Tool 15: Take Note
# -------------
@tool
def take_note_tool(note: str) -> str:
    """
    Docstring for take_note_tool
    
    :param note: Description
    :type note: str
    :return: Description
    :rtype: str
    """
    print("Taking note...")
    with open("notes.txt", "a", encoding="utf-8") as f:
        f.write(note + "\n")
    print("Note saved.")
    return "Note saved successfully."


# -------------
# Tool 16: Send Email with Attachments
# -------------
@tool
def send_email_with_attachments_tool(to_email: str, subject: str, body: str, attachment_paths: List[str]) -> str:
    """
    Send an email with file attachments
    
    :param to_email: Recipient email address
    :type to_email: str
    :param subject: Email subject
    :type subject: str
    :param body: Email body
    :type body: str
    :param attachment_paths: List of file paths to attach
    :type attachment_paths: List[str]
    :return: Status of the email sending
    :rtype: str
    """
    print(f"Sending email with attachments to {to_email}...")
    message = EmailMessage()
    message["From"] = EMAIL
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)
    
    # Attach files
    attached_files = []
    for file_path in attachment_paths:
        try:
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
                file_name = os.path.basename(file_path)
                
                # Determine the maintype and subtype based on file extension
                import mimetypes
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type:
                    maintype, subtype = mime_type.split('/', 1)
                else:
                    maintype, subtype = 'application', 'octet-stream'
                
                message.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=file_name)
                attached_files.append(file_name)
        except Exception as e:
            return f"Failed to attach file {file_path}. Error: {str(e)}"
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL, APP_PASSWORD)
            server.send_message(message)
        print(f"Email with {len(attachment_paths)} attachment(s) sent successfully to {to_email}.")
        return f"Email with {len(attached_files)} attachment(s) ({', '.join(attached_files)}) sent successfully to {to_email}."
    except Exception as e:
        return f"Failed to send email. Error: {str(e)}"

# -------------
# Tool 17: fetch messages from database
# -------------
@tool
def fetch_messages_from_db_tool() -> str:
    """
    Docstring for fetch_messages_from_db_tool
    
    :return: Description
    :rtype: str
    """
    print("Fetching messages from database...")
    try:
        connection = get_database_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, name, email, message, created_at FROM contacts WHERE replied=FALSE ORDER BY created_at DESC LIMIT 10")
            messages = cursor.fetchall()
        connection.close()
        print(f"Fetched {len(messages)} messages from database.")
        for msg in messages:
            print(f"ID: {msg['id']}, Name: {msg['name']}, Email: {msg['email']}, Message: {msg['message']}, Created At: {msg['created_at']}")
            print("-----"*10)
        # Convert datetime objects to strings for JSON serialization
        for msg in messages:
            if 'created_at' in msg and msg['created_at']:
                msg['created_at'] = str(msg['created_at'])
        return json.dumps(messages, indent=4, default=str)
    except Exception as e:
        print(f"Error fetching messages from database: {str(e)}")
        return f"Failed to fetch messages from database. Error: {str(e)}"
    
    

# -------------
# Tool 18: Getting email details by UID from database
# -------------
@tool
def get_email_details_from_db_tool(uid) -> str:
    """
    This tool fetches email details from the database based on the provided UID. It retrieves information such as the sender's name, email address, message content, and creation date for the specified email entry in the database. The results are returned in a JSON format for easy consumption by other components of the system. 
    """
    print(f"Fetching email details for UID: {uid}")
    try:
        connection = get_database_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, name, email, message, created_at FROM contacts WHERE id=%s", (uid,))
            email_details = cursor.fetchone()
        connection.close()
        
        if not email_details:
            print(f"No email found in database with UID: {uid}")
            return f"No email found in database with UID: {uid}"
        
        # Convert datetime object to string for JSON serialization
        if 'created_at' in email_details and email_details['created_at']:
            email_details['created_at'] = str(email_details['created_at'])
        
        print(f"Fetched email details for UID {uid}: {email_details}")
        return json.dumps(email_details, indent=4, default=str)
    except Exception as e:
        print(f"Error fetching email details from database for UID {uid}: {str(e)}")
        return f"Failed to fetch email details from database for UID {uid}. Error: {str(e)}"
    
# -------------
# Tool 19: Reply to email by UID from database
# -------------
#  CREATE TABLE IF NOT EXISTS contacts (
#             id INT AUTO_INCREMENT PRIMARY KEY,
#             name VARCHAR(100) NOT NULL,
#             email VARCHAR(100) NOT NULL,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             message TEXT NOT NULL,
#             replied BOOLEAN DEFAULT FALSE,
#             reply TEXT DEFAULT NULL,
#             replied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             attachments JSON DEFAULT NULL
#         );
@tool
def reply_to_email_from_db_tool(uid, reply_body) -> str:
    """
    This tool allows you to reply to an email entry in the database based on its UID. It retrieves the email details from the database using the provided UID, extracts the sender's email address, and sends a reply email with the specified body content. The tool returns a status message indicating whether the reply was sent successfully or if there were any issues during the process.
    """
    print(f"Replying to email from database with UID: {uid}")
    try:
        connection = get_database_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, name, email, message, created_at FROM contacts WHERE id=%s", (uid,))
            email_entry = cursor.fetchone()
        
        if not email_entry:
            connection.close()
            print(f"No email found in database with UID: {uid}")
            return f"No email found in database with UID: {uid}"
        
        to_email = email_entry['email']
        subject = f"Re: Your message received on {email_entry['created_at']}"
        
        send_status = send_email_tool.invoke(
            {"to_email": to_email, "subject": subject, "body": reply_body}
        )
        import datetime
        replied_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Reply sent at {replied_at} with content: {reply_body}", end="\n"*4)
        print(f"Reply status for UID {uid}: {send_status}")
        print(f"Updating database to mark email with UID {uid} as replied...")
        
        with connection.cursor() as cursor:
            cursor.execute("UPDATE contacts SET replied=TRUE, reply=%s, replied_at=%s WHERE id=%s", (reply_body, replied_at, uid))
            connection.commit()
        
        connection.close()
        print(f"Database updated successfully for UID {uid}.")
        return send_status
    except Exception as e:
        print(f"Error replying to email from database for UID {uid}: {str(e)}", end="\n"*4)
        return f"Failed to reply to email from database for UID {uid}. Error: {str(e)}"
    
# -------------
# Setting up the Language Model
# -------------

raw_model=ChatOllama(model=CHAT_OLLAMA_MODEL)
model_with_tools=raw_model.bind_tools([
    fetch_latest_unread_emails_tool,
    summarize_emails_tool,
    tavily_search_tool,
    send_email_tool,
    get_mails_tool,
    search_mail_tool,
    mails_received_from_tool,
    auto_reply_tool,
    delete_email_tool,
    mark_email_as_read_tool,
    mark_email_as_unread_tool,
    take_note_tool,
    list_recent_emails_tool,
    reply_to_latest_email_tool,
    delete_emails_tool,
    send_email_with_attachments_tool,
    fetch_messages_from_db_tool,
    get_email_details_from_db_tool,
    reply_to_email_from_db_tool
])


# -------------
# Building the Graph
# -------------

def llm_node(ChatState):
    response=model_with_tools.invoke(ChatState["messages"])
    return {"messages":ChatState["messages"] + [response]}

def router_node(ChatState):
    last_message=ChatState["messages"][-1]
    return 'tools' if getattr(last_message,'tool_calls',None) else 'end'

tool_node=ToolNode([
    fetch_latest_unread_emails_tool,
    summarize_emails_tool,
    tavily_search_tool,
    send_email_tool,
    get_mails_tool,
    search_mail_tool,
    mails_received_from_tool,
    auto_reply_tool,
    delete_email_tool,
    mark_email_as_read_tool,
    mark_email_as_unread_tool,
    take_note_tool,
    list_recent_emails_tool,
    reply_to_latest_email_tool,
    delete_emails_tool,
    send_email_with_attachments_tool,
    fetch_messages_from_db_tool,
    get_email_details_from_db_tool,
    reply_to_email_from_db_tool
])

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
    