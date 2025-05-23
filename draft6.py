import random
from typing import Literal
from IPython.display import Image, display
import os
from typing import TypedDict, List, Dict, Any, Optional

from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END

import mlflow  # to run: mlflow server --host 127.0.0.1 --port 8080
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv
load_dotenv()


# GLOBALS
HF_TOKEN = os.getenv('HF_TOKEN')
PHOENIX_API_KEY = os.getenv('PHOENIX_API_KEY')
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY')
LANGFUSE_SECRET_KEY= os.getenv('LANGFUSE_SECRET_KEY')
LANGFUSE_HOST= os.getenv('LANGFUSE_HOST')

# langfuse_handler = CallbackHandler(
#     public_key=LANGFUSE_PUBLIC_KEY,
#     secret_key=LANGFUSE_SECRET_KEY,
#     host=LANGFUSE_HOST
# )

# Optional: Set an experiment to organize your traces + Enable tracing
mlflow.set_experiment("LangChain MLflow Integration")
mlflow.langchain.autolog()

# model_name = "qwen3:1.7b"
model_name = "mistral:v0.3"

chat_model = ChatOllama(model=model_name)


# CLASSES AND FUNCS
class EmailState(TypedDict):
    email: Dict[str, Any]
    email_category: Optional[str]
    spam_reason: Optional[str]
    is_spam: Optional[bool]
    email_draft: Optional[str]
    messages: List[Dict[str, Any]]


def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]
    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")
    return {}  # no state changes needed here


def classify_email(state: EmailState):
    """Alfred uses an LLM to determine if the email is spam or legit"""
    email = state['email']
    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    First, determine if this email is spam as following:
    "This email is spam/not spam."
    If it is spam, explain why as following:
    "reason: <your explanation>"
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """
    messages = [HumanMessage(content=prompt)]
    response = chat_model.invoke(messages)  # call the LLM !

    response_text = response.content.lower()
    is_spam = 'spam' in response_text and 'not spam' not in response_text

    spam_reason = None
    if is_spam:
        spam_reason = 'kaha'
    if is_spam and 'reason:' in response_text:
        spam_reason = response_text.split('reason:')[1].strip()


    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break

    new_messages = state.get('messages', []) + [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': response.content}
    ]

    return {
        'is_spam': is_spam,
        'spam_reason': spam_reason,
        'email_category': email_category,
        'messages': new_messages
    }

def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    print(f'Alfred marked it as spam because {state['spam_reason']}')
    return {}

def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legit emails"""
    email = state['email']
    category = state['email_category'] or 'general'

    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    This email has been categorized as: {category}

    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """

    messages = [HumanMessage(content=prompt)]
    # response = chat_model.invoke(messages)  # LLM call !

    print("\n" + "=" * 50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-" * 50)
    response = chat_model.stream(messages)  # LLM call !
    email_draft = ''
    for c in response:
        print(c.content, end='')
        email_draft += c.content
    print("\n" + "=" * 50 + "\n")

    new_messages = state.get('messages', []) + [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': email_draft},
    ]

    return {
        'email_draft': email_draft,
        'messages': new_messages
    }


def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state['email']

    print("\n" + "="*50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-"*50)
    print(state["email_draft"])
    print("="*50 + "\n")

    return {}


def route_email(state: EmailState) -> str | List[str]:
    """Determine the next step based on spam classification"""
    if state['is_spam']:
        return 'spam'
    else:
        return 'legit'


# MAIN
def main():
    email_graph = StateGraph(EmailState)

    # nodes
    email_graph.add_node('read_email', read_email)
    email_graph.add_node('classify_email', classify_email)
    email_graph.add_node('handle_spam', handle_spam)
    email_graph.add_node('draft_response', draft_response)
    # email_graph.add_node('notify_mr_hugg', notify_mr_hugg)

    # edges
    email_graph.add_edge(START, 'read_email')
    email_graph.add_edge('read_email', 'classify_email')
    email_graph.add_conditional_edges('classify_email', route_email, {
        'spam': 'handle_spam',
        'legit': 'draft_response'
    })
    email_graph.add_edge('handle_spam', END)
    email_graph.add_edge('draft_response', END)
    # email_graph.add_edge('notify_mr_hugg', END)

    # compile
    compiled_graph = email_graph.compile()


    # display(Image(compiled_graph.get_graph().draw_mermaid_png()))


    # Example legitimate email
    legitimate_email = {
        "sender": "john.smith@example.com",
        "subject": "Question about your services",
        "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
    }

    # Example spam email
    spam_email = {
        "sender": "winner@lottery-intl.com",
        "subject": "YOU HAVE WON $5,000,000!!!",
        "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100."
    }


    # Process the legitimate email
    print("\nProcessing legitimate email...")
    legitimate_result = compiled_graph.invoke({
        "email": legitimate_email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "email_draft": None,
        "messages": []
    },
        # config={"callbacks": [langfuse_handler]}
    )
    print(legitimate_result)


    # Process the spam email
    print("\nProcessing spam email...")
    spam_result = compiled_graph.invoke({
        "email": spam_email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "email_draft": None,
        "messages": []
    },
        # config={"callbacks": [langfuse_handler]}
    )
    print(spam_result)

    compiled_graph.get_graph().draw_ascii()

if __name__ == '__main__':
    main()