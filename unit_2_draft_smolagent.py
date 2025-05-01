from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from smolagents import TransformersModel, tool, FinalAnswerTool
import os
from smolagents import OpenAIServerModel
from smolagents import VLLMModel
from transformers import BitsAndBytesConfig


@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

def main():

    # Initialize a local, quantized LLM using model_config
    model = TransformersModel(
        # model_id="Qwen/Qwen2.5-3B-Instruct",    # 3B parameter instruction Qwen2.5
        model_id="Qwen/Qwen2.5-0.5B-Instruct",    # 3B parameter instruction Qwen2.5
        device_map="auto",                       # auto device placement
        trust_remote_code=True,                   # allow custom model code
        # model_config={"quantization_config": bnb_config}
    )

    # Tool to suggest a menu based on the occasion



    # Alfred, the butler, preparing the menu for the party
    agent = CodeAgent(tools=[suggest_menu, FinalAnswerTool()], model=model)


    # print(agent.prompt_templates)
    # Preparing the menu for the party
    response = agent.run("Prepare a formal menu for the party.")
    # print(f"Response:\n{response}\n")


if __name__ == "__main__":
    main()
