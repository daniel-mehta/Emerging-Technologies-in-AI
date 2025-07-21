import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Avoid TensorFlow import warnings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_model():
    model_name = "flax-community/t5-recipe-generation"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def collect_user_input():
    ingredients = input("What ingredients do you have? (comma separated): ").split(",")
    ingredients = [item.strip() for item in ingredients if item.strip()]

    diet = input("Any dietary restrictions? (press Enter if none): ")
    meal_type = input("What meal is this for? (breakfast/lunch/dinner): ")
    servings = input("How many servings do you want?: ")

    return {
        "ingredients": ingredients,
        "diet": diet,
        "meal_type": meal_type,
        "servings": servings
    }

def build_prompt(user_input):
    ingredients = ", ".join(user_input["ingredients"])
    prompt = f"items: {ingredients}. "
    prompt += f"Create a unique, creative title and full recipe for a {user_input['meal_type']} dish that serves {user_input['servings']} people. "
    if user_input["diet"]:
        prompt += f"Make sure it follows these dietary restrictions: {user_input['diet']}. "
    prompt += "Respond in this format: title: <title> <section> ingredients: <ingredients separated by <sep>> <section> directions: <steps separated by <sep>>"
    return prompt

def clean_output(text, tokenizer):
    tokens_map = {
        "<sep>": "--",
        "<section>": "\n"
    }

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
            top_k=60,
            top_p=0.95,
            no_repeat_ngram_size=3
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return clean_output(generated, tokenizer)

def format_recipe_sections(text):
    sections = text.strip().split("\n")
    for section in sections:
        section = section.strip()
        if section.startswith("title:"):
            title = section.replace("title:", "").strip().title()
            print(f"\n# {title}")
        elif section.startswith("ingredients:"):
            print("\n## Ingredients")
            items = section.replace("ingredients:", "").split("--")
            for item in items:
                item = item.strip()
                if item:
                    print(f"- {item.capitalize()}")
        elif section.startswith("directions:"):
            print("\n## Instructions")
            steps = section.replace("directions:", "").split("--")
            for i, step in enumerate(steps, 1):
                step = step.strip()
                if step:
                    print(f"{i}. {step.capitalize()}")

def main():
    print("Welcome to the AI Recipe Generator!\n")
    tokenizer, model = load_model()
    user_input = collect_user_input()
    prompt = build_prompt(user_input)
    recipe = generate_recipe(tokenizer, model, prompt)

    print("\n--- Generated Recipe ---")
    format_recipe_sections(recipe)

if __name__ == "__main__":
    main()
