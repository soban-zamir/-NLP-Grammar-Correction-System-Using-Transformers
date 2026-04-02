# ==========================================
# Transformer-Based Grammar Aid (Single File)
# ==========================================

# Install dependencies
!pip install -q transformers sentencepiece torch spacy gradio
!python -m spacy download en_core_web_sm

import torch
import gradio as gr
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------
# Load Models
# ----------------------------
nlp = spacy.load("en_core_web_sm")

MODEL_NAME = "prithivida/grammar_error_correcter_v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------------------
# Grammar Correction
# ----------------------------
def correct_grammar(sentence):
    inputs = tokenizer.encode(
        "gec: " + sentence,
        return_tensors="pt",
        max_length=128,
        truncation=True
    ).to(device)

    outputs = model.generate(
        inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected


# ----------------------------
# Explanation Generator
# ----------------------------
def generate_explanations(original, corrected):
    explanations = []

    orig_doc = nlp(original)
    corr_doc = nlp(corrected)

    # Article errors
    articles = {"a", "an", "the"}
    if any(t.text.lower() in articles for t in corr_doc) and not any(
        t.text.lower() in articles for t in orig_doc
    ):
        explanations.append("Article usage corrected")

    # Verb tense/form
    orig_verbs = [t.lemma_ for t in orig_doc if t.pos_ == "VERB"]
    corr_verbs = [t.lemma_ for t in corr_doc if t.pos_ == "VERB"]
    if orig_verbs != corr_verbs:
        explanations.append("Verb tense or form corrected")

    # Prepositions
    orig_preps = [t.text for t in orig_doc if t.pos_ == "ADP"]
    corr_preps = [t.text for t in corr_doc if t.pos_ == "ADP"]
    if orig_preps != corr_preps:
        explanations.append("Preposition usage corrected")

    # Subject–verb agreement
    for o, c in zip(orig_doc, corr_doc):
        if o.pos_ == "VERB" and o.text != c.text:
            explanations.append("Subject–verb agreement corrected")
            break

    if not explanations:
        explanations.append("Minor grammatical or stylistic improvement")

    return explanations


# ----------------------------
# Main Function
# ----------------------------
def grammar_aid(sentence):
    if not sentence.strip():
        return "Please enter a sentence.", ""

    corrected = correct_grammar(sentence)

    explanations = generate_explanations(sentence, corrected)
    explanation_text = "\n".join([f"- {e}" for e in explanations])

    return corrected, explanation_text


# ----------------------------
# Gradio Interface
# ----------------------------
interface = gr.Interface(
    fn=grammar_aid,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter an English sentence...",
        label="Input Sentence"
    ),
    outputs=[
        gr.Textbox(label="Corrected Sentence"),
        gr.Textbox(label="Explanation")
    ],
    title="Transformer-Based Grammar Aid",
    description="Grammar correction using a Transformer model with educational explanations.",
)

interface.launch()
