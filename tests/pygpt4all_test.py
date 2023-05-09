# from nomic.gpt4all import GPT4All
from pygpt4all.models.gpt4all import GPT4All
# from pygpt4all import GPT4All_J

from nomic.gpt4all import GPT4AllGPU

# model_path = "/Downloads/gpt4all/models/ggml-gpt4all-j-v1.3-groovy.bin"
# model_path = "D:LLaMa/gpt4all/models/ggml-gpt4all-j-v1.3-groovy.bin"
# model = GPT4All_J(model_path=model_path)


# model_path = "/Downloads/gpt4all/models/gpt4all-lora-quantized-ggml.bin"
model_path = "/Downloads/gpt4all/models/ggml-model-q4_0.bin"
# model_path = "D:LLaMa/gpt4all/models/gpt4all-lora-quantized-ggml.bin"

model = GPT4All(model_path)

# model = GPT4AllGPU(model_path)


