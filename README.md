# Learning LLM Agents

<img src="pics/whiteboard-no-check.jpg" width="700">

## What is an agent?

_Agent_ - is an AI model capable of reasoning, planning, and interacting with its environment. More officially:
> "An Agent is a system that leverages an AI model to interact with its environment in order to achieve a user-defined objective. It combines reasoning, planning, and the execution of actions (often via external tools) to fulfill tasks."

_Agency_ - the ability to interact with the environment.

_Brain (AI Model)_ - this is where all the thinking happens. The AI model handles reasoning and planning. Decides on actions to take. 

_Body (Capabilities and Tools)_ - this is everything the agent is equipped to do.

Agency levels:

<img src="pics/a_levels.png" width="700">

The most common AI model found in agents is an LLM, which takes Text as an input and outputs Text as well.

## What are LLMs?

_LLMs_ are the most common type of _AI models_ for Agents.

There are encoders (text -> embedding), decoders (text -> next token), seq2seq or encoder-decoder (text -> embedding -> another text).

The underline principle: **its objective is to predict the next token, given a sequence of previous tokens**. 

Each model has its own end-of-sequence (EOS) token.


## Messages and Special Tokens

- there are special message tokens that are added to the input
- system message - is the first message the LLM sees, and it instructs it how to behave
- each model has its own conversation tokens style

An example:
```python
messages = [
    {"role": "system", "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
]
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```


## What are Tools?

A Tool is a function given to the LLM. This funciton should fulfill a clear objective.

A tool should contain:
- A textual description of what the function does 
- A _Callable_ with typings
- _Arguments_ with typings
- (Optional) Outputs with typings

For example, instead of retrieving the weather data itself, the LLM will generate text that reoresents a tool call, such as call weather_tool('Paris').

We describe all the tools an agent can use inside the system prompt.

**Model Context Protocol (MCP)** is an open protocol that standardizes how applications _provide tools to LLMs_. 
MCP provides:
- A growing list of pre-built integrations that your LLM can directly plug into
- The flexibility to switch between LLM providers and vendors
- Best practice for securing your data within your infrastructure

Any framework implementing MPC can leverage tools defined within the protocol.

## Agent workflow: Thought-Action-Observation Cycle

<img src="pics/agents_2.png" width="700">

The agent can go through the cycle again and again until reaching the solution. This principle is a part of a core concept _ReAct cycle_.

In the thought step, the agent breaks down the task to a set of sub-problems.

In the action step, it executes the tools via structured prompts. Or generates code to solve the task (Code agents).

In the observation step, the agent processes the output of actions. An important idea here is that this process works in loops, so that the agent has a chance to reiterate if needed.

## Create your first Agent

```python
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

model_name = "microsoft/phi-3-mini-4k-instruct"
client = InferenceClient(model_name)
#%%
output = client.text_generation(
    "The capitals of France were",
    max_new_tokens=100,
)
print(output)
#%%
output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capitals of France were"},
    ],
    stream=False,
    max_tokens=1024,
)
print(output.choices[0].message.content)
#%%
output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of Israel is"},
    ],
    stream=False,
    max_tokens=1024,
)
print(output.choices[0].message.content)
#%%
# This system prompt is a bit more complex and actually contains the function description already appended.
# Here we suppose that the textual description of the tools has already been appended.

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use :

{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}


ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """
#%%
messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London ?"},
    ]
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
#%%
prompt = tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
output = client.text_generation(
    prompt,
    max_new_tokens=200,
)

print(output)
#%%
output = client.text_generation(
    prompt,
    max_new_tokens=200,
    stop=["Observation:"] # Let's stop before any actual function is called
)

print(output)
#%%
# Dummy function
def get_weather(location):
    return f"the weather in {location} is sunny with low temperatures. \n"

get_weather('London')
#%%
new_prompt = prompt + output + get_weather('London')
final_output = client.text_generation(
    new_prompt,
    max_new_tokens=200,
)

print(final_output)
```

Tedious, yea? Let's learn smolagents and see how this (and other) framework helps us to save some time with technical code for agents.

## Create an Agent with `smolagents`





## Credits

- [🤗 Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)