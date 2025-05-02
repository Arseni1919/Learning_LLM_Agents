# from langchain_ollama import OllamaLLM
# from langchain_ollama import ChatOllama
#
# # llm = OllamaLLM(model="llama3.1:8b")
# # print(llm.invoke("The first man on the moon was ..."))
#
#
# # chat_model = ChatOllama(model="llama3.1:8b")
# chat_model = ChatOllama(model="gemma3:1b")
#
# # print(chat_model.invoke("Say the following input exactly as I tell you, letter by letter. Do not change anything. The input: ").content)
# print(chat_model.invoke("Create an image of a cat").content)



from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Retrieve HF_TOKEN from the environment variables
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
)

response = llm.complete("Hello, how are you?")
print(response)








