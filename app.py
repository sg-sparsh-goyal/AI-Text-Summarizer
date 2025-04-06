import gradio as gr
from transformers import pipeline
from newspaper import Article
import fitz  # PyMuPDF
from summarizer import Summarizer

# --------- UTILITY FUNCTIONS ---------

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
extractive_summarizer = Summarizer()

def generate_abstractive_summary(text, max_length=130, min_length=30):
    summary = abstractive_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def generate_extractive_summary(text, ratio=0.3):
    return extractive_summarizer(text, ratio=ratio)

def summarize_text(source_type, text, pdf, url, max_length, min_length, ratio):
    input_text = ""

    try:
        if source_type == "Text" and text:
            input_text = text
        elif source_type == "PDF" and pdf is not None:
            input_text = extract_text_from_pdf(pdf)
        elif source_type == "URL" and url:
            input_text = extract_text_from_url(url)
        else:
            return "❗Please provide a valid input.", ""

        if len(input_text.strip()) == 0:
            return "❗Input is empty after extraction.", ""

        # Bart/T5 models handle ~1024 tokens (~2000 characters)
        input_text = input_text[:2000]

        abstractive = generate_abstractive_summary(input_text, max_length, min_length)
        extractive = generate_extractive_summary(input_text, ratio)

        return abstractive, extractive

    except Exception as e:
        return f"⚠️ Error: {str(e)}", ""

# --------- GRADIO UI ---------

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 AI Text Summarizer\nChoose input type and get both **abstractive** and **extractive** summaries.")

    source_type = gr.Radio(["Text", "PDF", "URL"], label="Select Input Source")

    text_input = gr.Textbox(lines=8, label="Enter Text", visible=False)
    pdf_input = gr.File(label="Upload PDF", type="binary", visible=False)
    url_input = gr.Textbox(label="Enter URL", visible=False)

    max_length = gr.Slider(50, 300, step=10, value=130, label="Max Length (Abstractive)")
    min_length = gr.Slider(20, 100, step=10, value=30, label="Min Length (Abstractive)")
    ratio = gr.Slider(0.1, 1.0, step=0.1, value=0.3, label="Summary Ratio (Extractive)")

    btn = gr.Button("Generate Summaries")

    output_ab = gr.Textbox(label="Abstractive Summary")
    output_ex = gr.Textbox(label="Extractive Summary")

    def toggle_inputs(src):
        return {
            text_input: gr.update(visible=(src == "Text")),
            pdf_input: gr.update(visible=(src == "PDF")),
            url_input: gr.update(visible=(src == "URL"))
        }

    source_type.change(fn=toggle_inputs, inputs=source_type, outputs=[text_input, pdf_input, url_input])

    btn.click(
        summarize_text,
        inputs=[source_type, text_input, pdf_input, url_input, max_length, min_length, ratio],
        outputs=[output_ab, output_ex]
    )

if __name__ == "__main__":
    demo.launch()
