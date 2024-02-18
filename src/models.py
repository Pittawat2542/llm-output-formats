import os
from abc import ABC, abstractmethod

import google.generativeai as palm
import openai
from openai import Client
from dotenv import load_dotenv
from transformers import pipeline, Conversation, AutoTokenizer

from src.utils import sleep

# Initialize API keys
load_dotenv()
openai_client = Client()
palm.configure(api_key=os.getenv("PALM_API_KEY"))
hf_auth_token = os.getenv("HF_AUTH_TOKEN")


class IModel(ABC):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IModel, cls).__new__(cls)
        return cls._instance

    @abstractmethod
    def inference(self, prompt: str, temperature: float = 1):
        pass

    def __call__(self, *args, **kwargs):
        return self.inference(*args, **kwargs)

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class ChatGPT(IModel):
    def inference(self, prompt: str, temperature: float = 1):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except openai.APIConnectionError as e:
            print(f"OpenAI API request failed to connect: {e}")
            sleep(3)
            return self.inference(prompt)
        except openai.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            sleep(3)
            return self.inference(prompt)
        except openai.Timeout as e:
            print(f"OpenAI API request timed out: {e}")
            sleep(3)
            return self.inference(prompt)
        except openai.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            sleep(5)
            return self.inference(prompt)
        except Exception as e:
            print(f"Unexpected error: {e}")

    def __str__(self):
        return "ChatGPT"

    def __repr__(self):
        return "ChatGPT"


class GPT4(IModel):
    def inference(self, prompt: str, temperature: float = 1):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4-0613",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except openai.APIConnectionError as e:
            print(f"OpenAI API request failed to connect: {e}")
            sleep(3)
            return self.inference(prompt)
        except openai.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            sleep(3)
            return self.inference(prompt)
        except openai.Timeout as e:
            print(f"OpenAI API request timed out: {e}")
            sleep(3)
            return self.inference(prompt)
        except openai.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            sleep(5)
            return self.inference(prompt)
        except Exception as e:
            print(f"Unexpected error: {e}")

    def __str__(self):
        return "GPT4"

    def __repr__(self):
        return "GPT4"


class PaLM(IModel):
    def inference(self, prompt: str, temperature: float = 1):
        response = palm.chat(messages=prompt, temperature=temperature)
        return response.last

    def __str__(self):
        return "PaLM"

    def __repr__(self):
        return "PaLM"


class GeminiPro(IModel):
    def inference(self, prompt: str, temperature: float = 1):
        model = palm.GenerativeModel('gemini-pro')
        chat = model.start_chat()
        response = chat.send_message(prompt, generation_config=palm.types.GenerationConfig(temperature=temperature))
        return response.text

    def __str__(self):
        return "GeminiPro"

    def __repr__(self):
        return "GeminiPro"


class Llama2(IModel):
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IModel, cls).__new__(cls)
            cls._instance.model = pipeline(
                "conversational",
                model="meta-llama/Llama-2-7b-hf",
                token=hf_auth_token,
                temperature=0.5,
                max_length=4096,
                device_map="auto",
            )
        return cls._instance

    def inference(self, prompt: str, temperature: float = 1):
        conversation = Conversation(prompt)
        return self.model(conversation).generated_responses[-1]

    def __str__(self):
        return "Llama2"

    def __repr__(self):
        return "Llama2"


class Falcon(IModel):
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IModel, cls).__new__(cls)
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(
                "tiiuae/falcon-7b-instruct"
            )
            cls._instance.model = pipeline(
                "text-generation",
                model="tiiuae/falcon-7b-instruct",
                tokenizer=cls._instance.tokenizer,
                trust_remote_code=True,
                device_map="auto",
            )
        return cls._instance

    def inference(self, prompt: str, temperature: float = 1):
        sequences = self._instance.model(
            prompt,
            max_length=4096,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=self._instance.tokenizer.eos_token_id,
        )
        return sequences[-1]["generated_text"].replace(prompt, "").strip()

    def __str__(self):
        return "Falcon"

    def __repr__(self):
        return "Falcon"


class MPT(IModel):
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IModel, cls).__new__(cls)
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(
                "mosaicml/mpt-7b-instruct"
            )
            cls._instance.model = pipeline(
                "text-generation",
                model="mosaicml/mpt-7b-instruct",
                tokenizer=cls._instance.tokenizer,
                trust_remote_code=True,
                device_map="auto",
            )
        return cls._instance

    def inference(self, prompt: str, temperature: float = 1):
        instruction_key = "### Instruction:"
        response_key = "### Response:"
        intro_blurb = ("Below is an instruction that describes a task. Write a response that appropriately completes "
                       "the request.")
        prompt_for_generation_format = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
            intro=intro_blurb,
            instruction_key=instruction_key,
            instruction="{instruction}",
            response_key=response_key,
        )

        formatted_prompt = prompt_for_generation_format.format(instruction=prompt)

        sequences = self._instance.model(
            formatted_prompt,
            max_length=4096,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=self._instance.tokenizer.eos_token_id,
        )

        return sequences[-1]["generated_text"].replace(formatted_prompt, "").strip()

    def __str__(self):
        return "MPT"

    def __repr__(self):
        return "MPT"


def get_model(model_name: str) -> IModel:
    match model_name:
        case "gpt-3.5-turbo":
            return ChatGPT()
        case "gpt-4":
            return GPT4()
        case "palm":
            return PaLM()
        case "gemini-pro":
            return GeminiPro()
        case "llama-2":
            return Llama2()
        case "falcon":
            return Falcon()
        case "mpt":
            return MPT()
        case _:
            raise ValueError("Invalid model name.")
