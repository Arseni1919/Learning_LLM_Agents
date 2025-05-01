from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama

# llm = OllamaLLM(model="llama3.1:8b")
# print(llm.invoke("The first man on the moon was ..."))


# chat_model = ChatOllama(model="llama3.1:8b")
chat_model = ChatOllama(model="gemma3:1b")

# print(chat_model.invoke("Say the following input exactly as I tell you, letter by letter. Do not change anything. The input: ").content)
print(chat_model.invoke("Create an image of a cat").content)