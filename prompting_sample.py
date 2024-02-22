from google.cloud import aiplatform
aiplatform.init(project=projectid, location=Location)

from vertexai.preview.language_models import TextGenerationModel, TextEmbeddingModel

model = TextGenerationModel.from_pretrained("text-bison@001")

response = model.predict(''' Explain the Bayes theorem ''', temperature=0.2, max_output_tokens=256, top_k=40, top_p=0.8)
print(response)


#####################################################################################################3333

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="cuda", torch_dtype=torch.blfoat16)

chat = [{
    "role":"user", "content": "Explain the Bayes theorem"
}]
prompt = tokenizer.apply_chat_template(chat, tokenizer=False, add_generation_prompt=True)
inputs = tokenizer.encode(prompt, add_special_token=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0]))