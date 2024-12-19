### this code is a modified template from https://github.com/ollama/ollama-python ###

from ollama import Client

client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)
response = client.chat(model='qwen2:7b', messages=[
  {
    'role': 'user',
    'content': 'Can you give me the definition of "Advanced Deep Learning" in two sentences?',
  },
])["message"]["content"]

print(response)


# input prompt:
# Can you give me the definition of "Advanced Deep Learning" in two sentences?
#
# output text:
# Certainly! Advanced Deep Learning refers to the sophisticated and complex applications of neural networks that go beyond basic models, often involving deep architectures with many layers, innovative optimization techniques, and the use of large-scale data for training. These approaches aim to tackle intricate problems in various fields such as computer vision, natural language processing, and complex decision-making systems.