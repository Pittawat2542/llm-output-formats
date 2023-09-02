import os
from dotenv import load_dotenv

import openai
import google.generativeai as palm
from transformers import pipeline, Conversation

from src.utils import sleep

# Initialize API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
palm.configure(api_key=os.getenv("PALM_API_KEY"))
hf_auth_token = os.getenv("HF_AUTH_TOKEN")

# Initialize local models only once
# TODO: Change back to 13B
llama2_chat = pipeline("conversational", model="meta-llama/Llama-2-7b-chat-hf", token=hf_auth_token,
                       temperature=0.5, max_length=4096)


# stable_beluga_chat = pipeline("conversational", model="stabilityai/StableBeluga-7B", token=hf_auth_token,
#                               temperature=0.5, max_length=4096)
# stable_platypus2_chat = pipeline("conversational", model="garage-bAInd/Platypus2-7B",
#                                  token=hf_auth_token,
#                                  temperature=0.5, max_length=4096)  # "garage-bAInd/Stable-Platypus2-13B"


def chatgpt_model(prompt: str, temperature: float = 1):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except openai.error.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        sleep(3)
        return chatgpt_model(prompt)
    except openai.error.Timeout as e:
        print(f"OpenAI API request timed out: {e}")
        sleep(3)
        return chatgpt_model(prompt)
    except openai.error.APIConnectionError as e:
        print(f"OpenAI API request failed to connect: {e}")
        sleep(3)
        return chatgpt_model(prompt)
    except openai.error.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
        sleep(5)
        return chatgpt_model(prompt)
    except Exception as e:
        print(f"Unexpected error: {e}")


def gpt4_model(prompt: str, temperature: float = 1):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except openai.error.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        sleep(3)
        return gpt4_model(prompt)
    except openai.error.Timeout as e:
        print(f"OpenAI API request timed out: {e}")
        sleep(3)
        return gpt4_model(prompt)
    except openai.error.APIConnectionError as e:
        print(f"OpenAI API request failed to connect: {e}")
        sleep(3)
        return gpt4_model(prompt)
    except openai.error.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
        sleep(5)
        return gpt4_model(prompt)
    except Exception as e:
        print(f"Unexpected error: {e}")


def palm_model(prompt: str, temperature: float = 0.5):
    response = palm.chat(messages=prompt, temperature=temperature)
    return response.last


def llama2_chat_model(prompt: str):
    conversation = Conversation(prompt)
    return llama2_chat(conversation).generated_responses[-1]


def stable_beluga_chat_model(prompt: str):
    conversation = Conversation(prompt)
    return stable_beluga_chat(conversation).generated_responses[-1]


def stable_platypus2_chat_model(prompt: str):
    conversation = Conversation(prompt)
    return stable_platypus2_chat(conversation).generated_responses[-1]
