import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Avoid TensorFlow import warnings
os.environ["TRANSFORMERS_NO_TF"] = "1"

@st.cache_resource
def load_model():
    model_name = "flax-community/t5-recipe-generation"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def build_prompt(user_input):
    ingredients = ", ".join(user_input["ingredients"])
    prompt = f"items: {ingredients}\n"
    prompt += f"Make a {user_input['meal_type']} recipe for {user_input['servings']} people."

    if user_input["diet"]:
        prompt += f" It should follow these dietary restrictions: {user_input['diet']}."

    return prompt


def clean_output(text, tokenizer):
    tokens_map = {"<sep>": "--", "<section>": "\n"}
    for token in tokenizer.all_special_tokens:
        text = text.replace(token, "")
    for k, v in tokens_map.items():
        text = text.replace(k, v)
    return text.strip()

def generate_recipe(tokenizer, model, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=512,
            min_length=64,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            no_repeat_ngram_size=3
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated = generated.split("Make a")[0]  # crude stop marker

    return clean_output(generated, tokenizer)

def format_as_markdown(text):
    sections = text.strip().split("\n")
    markdown = ""
    for section in sections:
        section = section.strip()
        if section.startswith("title:"):
            title = section.replace("title:", "").strip().capitalize()
            markdown += f"# {title}\n\n"
        elif section.startswith("ingredients:"):
            items = section.replace("ingredients:", "").split("--")
            markdown += "## Ingredients\n"
            for item in items:
                cleaned = item.strip().capitalize()
                if cleaned:
                    markdown += f"- {cleaned}\n"
            markdown += "\n"
        elif section.startswith("directions:"):
            steps = section.replace("directions:", "").split("--")
            markdown += "## Instructions\n"
            for i, step in enumerate(steps):
                cleaned = step.strip().capitalize()
                if cleaned:
                    markdown += f"{i+1}. {cleaned}\n"
    return markdown

# --- Streamlit UI ---
st.title("🍽️ AI Recipe Generator")

with st.form("recipe_form"):
    ingredients_input = st.text_input("Ingredients (comma-separated)", placeholder="chicken, lemon, garlic, spinach, pasta")
    diet = st.text_input("Dietary restrictions (optional)")
    meal_type = st.selectbox("Meal type", ["breakfast", "lunch", "dinner"])
    servings = st.slider("Number of servings", 1, 8, 2)
    submitted = st.form_submit_button("Generate Recipe")

if submitted:
    ingredients = [item.strip() for item in ingredients_input.split(",") if item.strip()]
    if ingredients:
        st.write("Generating recipe...")
        tokenizer, model = load_model()
        user_input = {
            "ingredients": ingredients,
            "diet": diet,
            "meal_type": meal_type,
            "servings": servings
        }
        prompt = build_prompt(user_input)
        raw_output = generate_recipe(tokenizer, model, prompt)
        recipe_md = format_as_markdown(raw_output)
        st.markdown(recipe_md)
    else:
        st.warning("Please enter at least one ingredient.")
