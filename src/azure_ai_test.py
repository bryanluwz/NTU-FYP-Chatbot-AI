import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.ai.vision.imageanalysis.models._models import ImageAnalysisResult
from azure.core.credentials import AzureKeyCredential

# Set the values of your computer vision endpoint and computer vision key
# as environment variables:

endpoint = "https://ntu-fyp-chatbot-azure-ai-services.cognitiveservices.azure.com/"
key = "Fd4y4zPXPpAyu1YZGVBo8ZZnzDnGDLdqDb4c9RRIiZmaIs7Dq7KxJQQJ99BCACYeBjFXJ3w3AAAEACOGGXdd"

# Create an Image Analysis client for synchronous operations,
# using API key authentication
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

with open("./src/test.png", "rb") as image_file:
    image_bytes = image_file.read()

response: ImageAnalysisResult = client.analyze(
    image_bytes,
    visual_features=[VisualFeatures.CAPTION]
)
try:
    print(response.get('captionResult').get('text'))
except Exception as e:
    print(e)
    print(response.as_dict())
    print(response.get('captionResult'))
    print(response.get('captionResult').get('text'))
