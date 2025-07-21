from llama_cpp import Llama

def load_model():
    return Llama(model_path="model/dut-recipe-generator.i1-Q4_K_M.gguf", n_ctx=512)

def collect_user_input():
    ingredients = input("What ingredients do you have? (comma separated): ")

    diet = input("Any dietary restrictions? (press Enter if none): ")
    meal_type = input("What meal is this for? (breakfast/lunch/dinner): ")
    servings = input("How many servings do you want?: ")

    return {
        "ingredients": ingredients,
        "diet": diet,
        "meal_type": meal_type,
        "servings": servings
    }

def generate_prompt(user_input):
    prompt = f"Create a recipe using the following ingredients: {user_input['ingredients']}.\n"
    if user_input['diet']:
        prompt += f"Make sure it's suitable for someone with these dietary restrictions: {user_input['diet']}.\n"
    if user_input['meal_type']:
        prompt += f"This recipe should be for {user_input['meal_type']}.\n"
    if user_input['servings']:
        prompt += f"It should serve {user_input['servings']} people.\n"

    prompt += "Provide step-by-step instructions."

    return prompt

def generate_recipe(model, prompt):
    output = model(prompt, max_tokens=512, temperature=0.7)
    return output["choices"][0]["text"]

def main():
    model = load_model()
    user_input = collect_user_input()
    prompt = generate_prompt(user_input)
    recipe = generate_recipe(model, prompt)

    print("\n--- Generated Recipe ---\n")
    print(recipe)

if __name__ == "__main__":
    main()
