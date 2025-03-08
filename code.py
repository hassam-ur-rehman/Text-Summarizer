!pip install gradio
import torch
import gradio as gr

from transformers import pipeline

text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

def summary (input):
    output = text_summary(input)
    return output[0]['summary_text']
gr.close_all()


demo = gr.Interface(fn=summary,        
                    inputs=[gr.Textbox(label="Input text to summarize",lines=6)],
                    outputs=[gr.Textbox(label="Summarized text",lines=4)],
                    title="Personal Text Summarizer",
                    description="")
demo.launch()
