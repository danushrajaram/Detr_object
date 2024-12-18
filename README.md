

# 🖼 DETR Object Detection with Gradio 🖼

This project uses the **DETR model** (DEtection TRansformer) from Facebook to perform object detection on images. The application is built with Gradio and can be deployed seamlessly on Hugging Face Spaces.

[Detr Object App](https://huggingface.co/spaces/danushrajaram/ObjectsLabel) - Live Gradio to detect the object present in the image.🌟

[Danush Rajaram](https://www.linkedin.com/in/danushrajaram) - I am active on LinkedIn! Follow me to connect, share, and exchange ideas 🌟


---

## 🚀 Project Overview

The app performs object detection on uploaded images:
- Objects are detected using the **DETR ResNet-50** model.
- Bounding boxes and confidence scores are drawn on the image.
- Results are displayed interactively via a **Gradio interface**.

---

## 📋 Requirements

1. **Python 3.8+**
2. Required Libraries:
   - `torch`
   - `transformers`
   - `diffusers`
   - `Pillow`
   - `gradio`

---

## 📂 Project Structure

```plaintext
Project Root
├── app.py                # Main Gradio application
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## 🛠️ Installation

Follow these steps to set up the project:

1. **Clone the Repository**

```bash
git clone https://github.com/<your-username>/detr-object-detection.git
cd detr-object-detection
```

2. **Set Up Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Gradio App Locally**

```bash
python app.py
```

---

## 🚀 Deploying on Hugging Face Spaces

Follow these steps to deploy your Gradio app to Hugging Face Spaces:

1. **Create a New Space**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
   - Choose **Gradio** as the SDK.

2. **Add Project Files**
   - Upload the following files to the Space:
     - `app.py` (Main code file)
     - `requirements.txt` (Dependencies file)

3. **Add Dependencies**
   - Ensure the `requirements.txt` includes:

     ```plaintext
     torch
     transformers
     diffusers
     Pillow
     gradio
     ```

4. **Run the Space**
   - Once files are uploaded, Hugging Face will automatically install dependencies and run the app.

---

## 📸 Usage

1. Open the Gradio interface.
2. Upload an image.
3. The app will:
   - Detect objects in the image.
   - Draw bounding boxes with confidence scores.
   - List detected objects in text format.
