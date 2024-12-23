### this code is a modified template from https://github.com/ollama/ollama-python ###

from ollama import Client

client = Client(
  host='http://localhost:11434'
)
response = client.chat(model='qwen2:7b', messages=[
  {
    'role': 'system',
    'content': 'You are a professor at the Machine Learning and Computer Vision Lab.'
  },
  {
    'role': 'user',
    'content': 'Give me the definition of "Advanced Deep Learning" in one short sentence!',
  },
])

print(response["message"]["content"])


# input prompt:
# Give me the definition of "Advanced Deep Learning" in one short sentence!
#
# output text:
# Advanced Deep Learning refers to sophisticated neural network architectures and algorithms that utilize deep learning techniques to solve complex problems with high performance, often involving large datasets, specialized architectures like Transformers, and innovative optimization methods.