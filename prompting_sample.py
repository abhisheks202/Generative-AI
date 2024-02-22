from google.cloud import aiplatform
aiplatform.init(project=projectid, location=Location)

from vertexai.preview.language_models import TextGenerationModel, TextEmbeddingModel

model = TextGenerationModel.from_pretrained("text-bison@001")

response = model.predict(''' Explain the Bayes theorem ''', temperature=0.2, max_output_tokens=256, top_k=40, top_p=0.8)
print(response)