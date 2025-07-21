# AI Recipe Generator

This is a simple Streamlit app that generates recipes using a T5-based model from Hugging Face (`flax-community/t5-recipe-generation`). You provide ingredients, dietary restrictions, meal type, and number of servings, and it returns a full recipe.

## Features
- Input form for ingredients and preferences
- Generates creative recipes with instructions
- Uses Hugging Face Transformers and PyTorch

## Requirements
Install dependencies using:

```
pip install -r requirements.txt
```

## Run the App

```
streamlit run app.py
```

## Notes
- Model used: `flax-community/t5-recipe-generation`
- Torch and Transformers are required
- You can modify the formatting and behavior in `app.py`
