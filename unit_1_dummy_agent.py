import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

## You need a token from https://hf.co/settings/tokens, ensure that you select 'read' as the token type. If you run this on Google Colab, you can set it up in the "settings" tab under "secrets". Make sure to call it "HF_TOKEN"
# os.environ["HF_TOKEN"]="hf_xxxxxxxxxxxxxx"

# client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")
# client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8")

client = InferenceClient("microsoft/phi-3-mini-4k-instruct")
# if the outputs for next cells are wrong, the free model may be overloaded. You can also use this public endpoint that contains Llama-3.2-3B-Instruct
# client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")

# output = client.text_generation(
#     "The capitals of France were",
#     max_new_tokens=100,
# )
#
# print(output)

output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of France is"},
    ],
    stream=False,
    max_tokens=1024,
)
print(output.choices[0].message.content)










