from google import genai
import nltk
import yaml
import sys

# Method to load the config file which contains the API key
def load_config(config_path='/home/ssever/rag-llm-demo/config/config.yaml'):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Config error: {e}")
        sys.exit(1)

def client_response(prompt, api_key, model="gemini-2.0-flash"):
    client = genai.Client(api_key=api_key)
    try:
        return client.models.generate_content_stream(
            model=model, 
            contents=[prompt]
        )
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

# Display the response in a more accurate stream
def display_response(response):
    if not response:
        return
    
    for chunk in response:
        words = nltk.word_tokenize(chunk.text)
        for word in words:
            print(word, end=" ")
    print()  # Add a newline at the end

def main():
    # Initialize NLTK
    nltk.download('punkt')
    
    # Load configuration and generate response
    config = load_config()
    api_key = config['api']['key']
    prompt = "What's the weather like?"
    response = client_response(prompt, api_key)
    display_response(response)

if __name__ == "__main__":
    main()

