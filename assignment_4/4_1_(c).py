### this code is a modified template from https://github.com/ollama/ollama-python ###

from ollama import Client

client = Client(
  host='http://localhost:11434'
)
response = client.chat(model='qwen2:7b', messages=[
  {
    'role': 'system',
    'content': 'You are a professor at the Machine Learning and Computer Vision Lab'
  },
  {
    'role': 'user',
    'content': 'Can you give me the definition of "Advanced Deep Learning" in two sentences?',
  },
])

print(response["message"]["content"])


# input prompt:
# Can you give me the definition of "Advanced Deep Learning" in two sentences?
#
# output text:
# Advanced Deep Learning refers to sophisticated neural network architectures and methodologies that push the boundaries of traditional deep learning techniques, often involving complex models such as Transformers, Generative Adversarial Networks (GANs), or autoencoders, which tackle challenging tasks like natural language understanding, image generation, and unsupervised feature learning with high performance.