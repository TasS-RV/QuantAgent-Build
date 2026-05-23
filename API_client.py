import os # Added the missing import
import google.genai as genai
from google.genai import types

key_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'Gemini_API.txt'))

try:
    with open(key_path, "r") as file:
        API_KEY = file.read().strip()
    
    client = genai.Client(api_key=API_KEY)
    print(f"Successfully authenticated. Loaded key from: {key_path}")
except FileNotFoundError:
    print(f"Error: api_key.txt not found at {key_path}.")
    print("Please ensure the file is saved exactly one folder above this repository.")
    exit()