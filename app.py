import gradio as gr
from model import *

def predict(text):
    return test_model(model, text)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter a Human-readable Date", max_length=30),
    outputs=gr.Textbox(label="Translated Date"),
    title="Date Recognition with Attention Model",
    description="""### Example User Inputs

    - `May 1, 2025 -> 2025-05-01`
    - `3 May 1979 -> 1979-05-03`
    - `Tue 10 Jul 2007 -> 2007-07-10`
    - `Monday November 13 2000 -> 2000-11-13`
    - `Aug 30 2013 -> 2013-08-30`

    **Format:** `Human-readable date within 30 characters` into `Date format [yyyy-mm-dd]`.""",
    article="See on GitHub Repo via [GitHub Link](https://github.com/ATreep/DateRecogAttentionModel)")

if __name__ == "__main__":
    demo.launch()

