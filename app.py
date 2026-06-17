import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Loads directly from HuggingFace Hub — no local model files needed
model_path = "Anamali153/financial-sentiment-distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_sentiment(text):
    if not text.strip():
        return "Please enter a sentence.", ""

    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128, padding="max_length"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()

    label_names = ["negative", "neutral", "positive"]
    emoji_map = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    return emoji_map[label_names[pred]], f"{confidence:.1%}"

examples = [
    ["The company reported record profits for Q3."],
    ["Operating losses widened amid rising costs."],
    ["The board approved the annual dividend payment."],
]

with gr.Blocks(title="Financial Sentiment Analyser") as demo:
    gr.Markdown("# Financial Sentiment Analyser")
    text_input = gr.Textbox(label="Financial Sentence", lines=2)
    submit_btn = gr.Button("Analyse", variant="primary")
    sentiment_output = gr.Textbox(label="Sentiment")
    confidence_output = gr.Textbox(label="Confidence")
    submit_btn.click(predict_sentiment, inputs=text_input, outputs=[sentiment_output, confidence_output])
    gr.Examples(examples=examples, inputs=text_input)

if __name__ == "__main__":
    demo.launch()
