import gradio as gr
from src.predict import predict
from PIL import Image
import os


def predict_image(image: Image.Image):
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "temp_upload.jpg")
    image.save(temp_path)
    result = predict(temp_path)
    # Return label and raw probability/confidence
    return result['Label'], result['Probability'], result['Confidence']


def main():
    title = "Chicken Type Classifier"
    description = "Upload a hen image to predict probability of being Local or Broiler."

    iface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Upload Hen Image"),
        outputs=[gr.Label(num_top_classes=2, label="Predicted Label"), gr.Number(label="Probability"), gr.Number(label="Confidence (%)")],
        title=title,
        description=description,
        allow_flagging="never",
    )

    iface.launch()


if __name__ == '__main__':
    main()
