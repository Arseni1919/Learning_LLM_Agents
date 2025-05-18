from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama

model_name = "qwen3:1.7b"

# llm = OllamaLLM(model=model_name)
# print(llm.invoke("The first man on the moon was ..."))

chat_model = ChatOllama(model=model_name)
print(chat_model.invoke("Who was the first man on the moon?"))