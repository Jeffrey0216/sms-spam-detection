import gradio as gr
import spaces
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# Your fine-tuned model
MODEL_NAME = "Jeffrey0216/sms-spam-distilbert"

# Accuracy from your test set
MODEL_ACCURACY = 99.28

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# Load fine-tuned model
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME
)

model.eval()


@spaces.GPU
def predict_spam(message):

    if not message.strip():
        return "Please enter an SMS message."

    # Tokenize input
    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)

    # Get predicted class
    prediction = torch.argmax(probabilities, dim=1).item()

    # Get confidence
    confidence = probabilities[0][prediction].item() * 100

    # Convert class to label
    if prediction == 1:
        result = "🚨 SPAM"
    else:
        result = "✅ NOT SPAM"

    return (
        f"{result}\n\n"
        f"Confidence: {confidence:.2f}%\n"
        f"Model Accuracy: {MODEL_ACCURACY:.2f}%"
    )


# Gradio interface
demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(
        label="Enter SMS",
        placeholder="Example: Congratulations! You have won a free prize!"
    ),
    outputs=gr.Textbox(
        label="Prediction"
    ),
    title="📱 SMS Spam Detector",
    description="Fine-tuned DistilBERT model for SMS spam classification."
)

demo.launch()