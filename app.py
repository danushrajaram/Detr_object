import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import ImageDraw, ImageFont, Image
import requests
import os

# Load the DETR processor and model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Define the inference function
def inference(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Convert outputs (bounding boxes and class logits) to COCO API
    # Only keep detections with a score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Draw bounding boxes and labels on the image
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Alternate font path
    font = ImageFont.truetype(font_path, 15)

    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=2)
        label_text = f"{model.config.id2label[label.item()]} ({round(score.item() * 100, 1)}%)"
        draw.text((box[0], box[1]), label_text, fill="red", font=font)
        detected_objects.append(f"{label_text} at {box}")

    return image, "\n".join(detected_objects)

# Create the Gradio interface
demo = gr.Interface(
    fn=inference,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Textbox()],
    title="DETR Object Detection",
    description=(
        "Upload an image, and this app will detect objects using the DETR model. "
        "The detected objects are displayed with bounding boxes and confidence scores."
    ),
)

# Launch the app
demo.launch()
