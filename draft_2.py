# import datasets
# from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, ToolCallingAgent
# from smolagents import TransformersModel, tool
# import os
# from smolagents import OpenAIServerModel
# from smolagents import VLLMModel
# from transformers import BitsAndBytesConfig
# import numpy as np
# import time
# import datetime
# from smolagents import MLXModel
# from transformers import AutoTokenizer

# agent = CodeAgent(tools=[], model=model, additional_authorized_imports=['datetime'])
# system_prompt = '''
# You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
# To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
# To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.
#
# At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
# Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
# During each intermediate step, you can use 'print()' to save whatever important information you will then need.
# These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
# In the end you have to return a final answer using the `final_answer` tool.
#
# Here are a few examples using notional tools:
# ---
# Task: "Generate an image of the oldest person in this document."
#
# Thought: I will proceed step by step and use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.
# Code:
# ```py
# answer = document_qa(document=document, question="Who is the oldest person mentioned?")
# print(answer)
# ```<end_code>
# Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."
#
# Thought: I will now generate an image showcasing the oldest person.
# Code:
# ```py
# image = image_generator("A portrait of John Doe, a 55-year-old man living in Canada.")
# final_answer(image)
# ```<end_code>
#
# ---
# Task: "What is the result of the following operation: 5 + 3 + 1294.678?"
#
# Thought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool
# Code:
# ```py
# result = 5 + 3 + 1294.678
# final_answer(result)
# ```<end_code>
#
# ---
# Task:
# "Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French.
# You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
# {'question': 'Quel est l'animal sur l'image?', 'image': 'path/to/image.jpg'}"
#
# Thought: I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.
# Code:
# ```py
# translated_question = translator(question=question, src_lang="French", tgt_lang="English")
# print(f"The translated question is {translated_question}.")
# answer = image_qa(image=image, question=translated_question)
# final_answer(f"The answer is {answer}")
# ```<end_code>
#
#
# Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools, behaving like regular python functions:
# ```python
# {%- for tool in tools.values() %}
# def {{ tool.name }}({% for arg_name, arg_info in tool.inputs.items() %}{{ arg_name }}: {{ arg_info.type }}{% if not loop.last %}, {% endif %}{% endfor %}) -> {{tool.output_type}}:
#     """{{ tool.description }}
#
#     Args:
#     {%- for arg_name, arg_info in tool.inputs.items() %}
#         {{ arg_name }}: {{ arg_info.description }}
#     {%- endfor %}
#     """
# {% endfor %}
# ```
#
# {%- if managed_agents and managed_agents.values() | list %}
# You can also give tasks to team members.
# Calling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'task'.
# Given that this team member is a real human, you should be very verbose in your task, it should be a long string providing informations as detailed as necessary.
# Here is a list of the team members that you can call:
# ```python
# {%- for agent in managed_agents.values() %}
# def {{ agent.name }}("Your query goes here.") -> str:
#     """{{ agent.description }}"""
# {% endfor %}
# ```
# {%- endif %}
#
# Here are the rules you should always follow to solve your task:
# 1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
# 2. Use only variables that you have defined!
# 3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
# 4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
# 5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
# 6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
# 7. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
# 8. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
# 9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
# 10. Don't give up! You're in charge of solving the task, not providing directions to solve it.
#
# Now Begin!
# '''
# Access the default prompt templates
# default_templates = agent.prompt_templates
# default_templates['system_prompt'] = system_prompt
# agent = CodeAgent(tools=[], model=model, prompt_templates=default_templates, additional_authorized_imports=['datetime'])
# agent = ToolCallingAgent(tools=[], model=model)
# model = MLXModel(model_id="mistralai/Mistral-7B-Instruct-v0.3")


# from smolagents import DuckDuckGoSearchTool, ToolCallingAgent, FinalAnswerTool
# from smolagents import MLXModel
# from phoenix.otel import register
# from openinference.instrumentation.smolagents import SmolagentsInstrumentor
#
# register()
# SmolagentsInstrumentor().instrument()
#
# model = MLXModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct", max_tokens=10000)
# agent = ToolCallingAgent(tools=[FinalAnswerTool(), DuckDuckGoSearchTool()], model=model)
# agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

# messages = [
#     {
#       "role": "system",
#     "content": [{"type": "text", "text":  "You are an expert assistant who can solve any task using tool calls. You will be given a task to solve as best you can.\nTo do so, you have been given access to some tools.\n\nThe tool call you write is an action: after the tool is executed, you will get the result of the tool call as an \"observation\".\nThis Action/Observation can repeat N times, you should take several steps when needed.\n\nYou can use the result of the previous action as input for the next action.\nThe observation will always be a string: it can represent a file, like \"image_1.jpg\".\nThen you can use it as input for the next action. You can do it for instance as follows:\n\nObservation: \"image_1.jpg\"\n\nAction:\n{\n  \"name\": \"image_transformer\",\n  \"arguments\": {\"image\": \"image_1.jpg\"}\n}\n\nTo provide the final answer to the task, use an action blob with \"name\": \"final_answer\" tool. It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:\nAction:\n{\n  \"name\": \"final_answer\",\n  \"arguments\": {\"answer\": \"insert your final answer here\"}\n}\n\n\nHere are a few examples using notional tools:\n---\nTask: \"Generate an image of the oldest person in this document.\"\n\nAction:\n{\n  \"name\": \"document_qa\",\n  \"arguments\": {\"document\": \"document.pdf\", \"question\": \"Who is the oldest person mentioned?\"}\n}\nObservation: \"The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland.\"\n\nAction:\n{\n  \"name\": \"image_generator\",\n  \"arguments\": {\"prompt\": \"A portrait of John Doe, a 55-year-old man living in Canada.\"}\n}\nObservation: \"image.png\"\n\nAction:\n{\n  \"name\": \"final_answer\",\n  \"arguments\": \"image.png\"\n}\n\n---\nTask: \"What is the result of the following operation: 5 + 3 + 1294.678?\"\n\nAction:\n{\n    \"name\": \"python_interpreter\",\n    \"arguments\": {\"code\": \"5 + 3 + 1294.678\"}\n}\nObservation: 1302.678\n\nAction:\n{\n  \"name\": \"final_answer\",\n  \"arguments\": \"1302.678\"\n}\n\n---\nTask: \"Which city has the highest population , Guangzhou or Shanghai?\"\n\nAction:\n{\n    \"name\": \"search\",\n    \"arguments\": \"Population Guangzhou\"\n}\nObservation: ['Guangzhou has a population of 15 million inhabitants as of 2021.']\n\n\nAction:\n{\n    \"name\": \"search\",\n    \"arguments\": \"Population Shanghai\"\n}\nObservation: '26 million (2019)'\n\nAction:\n{\n  \"name\": \"final_answer\",\n  \"arguments\": \"Shanghai\"\n}\n\nAbove example were using notional tools that might not exist for you. You only have access to these tools:\n- final_answer: Provides a final answer to the given problem.\n    Takes inputs: {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}\n    Returns an output of type: any\n- web_search: Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results.\n    Takes inputs: {'query': {'type': 'string', 'description': 'The search query to perform.'}}\n    Returns an output of type: string\n\nHere are the rules you should always follow to solve your task:\n1. ALWAYS provide a tool call, else you will fail.\n2. Always use the right arguments for the tools. Never use variable names as the action arguments, use the value instead.\n3. Call a tool only when needed: do not call the search agent if you do not need information, try to solve the task yourself.\nIf no tool call is needed, use final_answer tool to return your answer.\n4. Never re-do a tool call that you previously did with the exact same parameters.\n\nNow Begin!"}]
#
#     },
#     {
#       "role": "user",
#       "content":[
#             {"type": "text", "text": "New task:\nSearch for the best music recommendations for a party at the Wayne's mansion."}
#         ]
#     }
# ]

# response = model(messages, stop_sequences=["great"])
# response = model(messages, stop_sequences=["END"])
# print(response)


#

#
# agent.run(
#     """
#     what color is the sky?
#     """
# )

# from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
#
# agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())
#
# agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")


from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-Coder-32B-Instruct-4bit")

prompt="hello"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
print(response)


# from smolagents import CodeAgent, MLXModel
#
# model = MLXModel(
#     "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
#     {
#         "temperature": 0.7,
#         "top_k": 20,
#         "top_p": 0.8,
#         "min_p": 0.05,
#         "num_ctx": 32768,
#     },
# )
# agent = CodeAgent(tools=[], model=model, add_base_tools=True)
# agent.run(
#     "Could you give me the 40th number in the Fibonacci sequence?",
# )