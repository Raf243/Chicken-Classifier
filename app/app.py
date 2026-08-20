import gradio as gr
import sys
sys.path.append("../src")
from predict import predict

def classify_image(image_path):
    # Get the prediction
    result = predict(image_path)

    # Formal for the output
    probability = result['Probability']
    label = result["Label"]
    confidence = result["Confidence"]

    labels = {
        "Local Breed": round(1-probability, 4),
        "Broiler": round(probability, 4)
    }
    summary = f"Predicted  Breed: {label}\nConfidence: {confidence}"

    return labels, summary

app = gr.Interface(
    fn = classify_image,
    inputs= gr.Image(type="filepath", label="Upload Chicken Photo"),
    outputs= [
        gr.Label(label= "Breed Prediction"),
        gr.Text(label= "Detailed Summary")],
    title= "Chicken Breed Classifier",
    description= "Upload a photo of a chicken to classify it as a Local Breed or a Broiler"
)

app.launch()