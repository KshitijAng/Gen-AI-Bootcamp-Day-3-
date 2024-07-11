import boto3
import json
import base64
from PIL import Image
from io import BytesIO
import gradio as gr

def generate_image(prompt):
    bedrock = boto3.client(service_name="bedrock-runtime")

    payload = {
        "textToImageParams": {
            "text": prompt
        },
        "taskType": "TEXT_IMAGE",
        "imageGenerationConfig": {
            "cfgScale": 8,
            "seed": 0,
            "quality": "standard",
            "width": 1024,
            "height": 1024,
            "numberOfImages": 1
        }
    }

    body = json.dumps(payload)
    model_id = "amazon.titan-image-generator-v1"

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    base64_image = response_body.get("images")[0]
    image = Image.open(BytesIO(base64.b64decode(base64_image)))

    return image

iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=3, placeholder="Enter your prompt here..."),
    outputs="image",
    title="Image Generator Tool",
    description="Generate images from text prompts using Amazon Bedrock"
)

iface.launch()
