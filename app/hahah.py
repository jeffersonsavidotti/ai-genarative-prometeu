import openai
import os

# Configurar a chave de API da OpenAI
api_key = 'sk-my-service-key-23P8DMcGXwX9Q7vnkSgNT3BlbkFJchUwETdknPyflpfWshBl'
openai.api_key = api_key

model_name = "gpt-3.5-turbo"

# Exemplo de chamada à API da OpenAI
response = openai.ChatCompletion.create(
    model=model_name,
    messages=[
        # {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=1,
    max_tokens=32,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# Imprimir a resposta
print(response['choices'][0]['message']['content'])
