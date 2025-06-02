import base64
from typing import List, TypedDict, Annotated, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
import mlflow  # to run: mlflow server --host 127.0.0.1 --port 8080
from langchain_ollama import ChatOllama

import matplotlib.pyplot as plt
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv
import os
load_dotenv()

# GLOBALS
HF_TOKEN = os.getenv('HF_TOKEN')
PHOENIX_API_KEY = os.getenv('PHOENIX_API_KEY')
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY')
LANGFUSE_SECRET_KEY= os.getenv('LANGFUSE_SECRET_KEY')
LANGFUSE_HOST= os.getenv('LANGFUSE_HOST')

langfuse_handler = CallbackHandler(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)

# mlflow.set_experiment("Doc Analysis")
# mlflow.langchain.autolog()


class AgentState(TypedDict):
    # The doc provided
    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]

vlm_name = "moondream:latest"
# model_name = "mistral:v0.3"
model_name = "qwen3:1.7b"

vision_llm = ChatOllama(model=vlm_name)

def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.

    Master Wayne often leaves notes with his training regimen or meal plans.
    This allows me to properly analyze the contents.
    """
    all_text = ''
    try:
        # read image and encode as base64
        with open(img_path, 'rb') as image_file:
            image_bytes = image_file.read()
        image_base64 = base64.b64decode(image_bytes).decode('utf-8')

        # prepare the prompt including the base64 image data
        message = [
            HumanMessage(
                content=[
                    {
                        'type': 'text',
                        'text': (
                            'Extract all the text from this image.'
                            'Return only the extracted text, no explanations.'
                        ),
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{image_base64}'
                        },
                    },
                ]
            )
        ]

        # call the vision-capable model
        response = vision_llm.invoke(message)

        # append extracted text
        all_text += response.content + '\n\n'

        return all_text.strip()

    except Exception as e:
        error_msg = f'Error extracting text: {str(e)}'
        print(error_msg)
        return ""


def divide(a: int, b: int) -> float:
    """Divide a and b - for Master Wayne's occasional calculations."""
    return a / b


tools = [
    divide,
    extract_text
]


llm = ChatOllama(model=model_name)
llm_with_tools = llm.bind_tools(tools)


# --- THE NODES ---


def assistant(state: AgentState):
    # system message
    textual_description_of_tool = """
    extract_text(img_path: str) -> str:
        Extract text from an image file using a multimodal model.
    
        Args:
            img_path: A local image file path (strings).
    
        Returns:
            A single string containing the concatenated text extracted from each image.
    divide(a: int, b: int) -> float:
        Divide a and b
    """
    image = state['input_file']
    sys_msg = SystemMessage(content=f'You are a helpful butler named Alfred that serves Mr. Wayne and Batman. You can analyse documents and run computations with provided tools:\n{textual_description_of_tool}.'
                                    f'You should use tools where needed.'
                                    f' \n You have access to some optional images. Currently the loaded image is: {image}')
    return {
        'massages': [llm_with_tools.invoke([sys_msg] + state['messages'])],
        'input_file': state['input_file']
    }


builder = StateGraph(AgentState)

builder.add_node('assistant', assistant)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, 'assistant')
builder.add_conditional_edges(
    'assistant',
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition
)
builder.add_edge('tools', 'assistant')
react_graph = builder.compile()

# display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
print(react_graph.get_graph(xray=True).draw_ascii())

messages = [HumanMessage(content='Divide 6790 by 5')]
out_result = react_graph.invoke({'messages': messages, 'input_file': None}, config={"callbacks": [langfuse_handler]})
for m in out_result['messages']:
    m.pretty_print()
