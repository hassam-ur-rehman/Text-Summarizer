Personal Text Summarizer:
A simple real-time text summarizer built using Hugging Face Transformers, PyTorch, and Gradio. This project leverages the sshleifer/distilbart-cnn-12-6 model to generate high-quality text summaries through a user-friendly web interface.

Features:
Uses a pretrained DistilBART CNN model for text summarization
Provides real-time summarization via a web-based UI
Built with PyTorch & Hugging Face Transformers
Lightweight & easy to use with Gradio
Installation
To run the project locally, install the required dependencies:

How to run it? 

pip install gradio torch transformers

Run the Python script to launch the summarizer:

python:

import torch
import gradio as gr
from transformers import pipeline

# Load the model
text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

# Define the summarization function
def summary(input):
    output = text_summary(input)
    return output[0]['summary_text']

# Launch the Gradio interface
demo = gr.Interface(fn=summary,
                    inputs=[gr.Textbox(label="Input text to summarize", lines=6)],
                    outputs=[gr.Textbox(label="Summarized text", lines=4)],
                    title="Personal Text Summarizer",
                    description="")
demo.launch()


How It Works:
  Enter text into the input box.
  Click submit, and the model generates a concise summary.
  View the summarized text in real-time.


Technologies Used:
  Hugging Face Transformers – Model and pipeline integration
  PyTorch – Deep learning framework
  Gradio – Easy-to-use web interface for AI applications
  
Future Improvements:
  Add support for multiple summarization models
  Improve UI with custom styling
  Deploy as a web app (Hugging Face Spaces or Streamlit)
License:
This project is open-source under the MIT License. Feel free to contribute and improve it.
