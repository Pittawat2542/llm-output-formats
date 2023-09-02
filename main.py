from src.models import llama2_chat_model

# TODO: Add logging and raw output saving and printing
# TODO: Make it resumable
# TODO: Add try-catch for OpenAI error

if __name__ == '__main__':
    print(llama2_chat_model("Hello, how are you?"))
