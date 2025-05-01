from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama

# llm = OllamaLLM(model="llama3.1:8b")
# print(llm.invoke("The first man on the moon was ..."))


# chat_model = ChatOllama(model="llama3.1:8b")
chat_model = ChatOllama(model="gemma3:1b")

print(chat_model.invoke("Who was the first man on the moon?"))