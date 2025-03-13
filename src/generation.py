from google import genai
import nltk
import yaml
import sys

# Method to load the config file which contains the API key
def load_config(config_path='/home/ssever/rag-llm-demo/config/config.yaml'):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError:
        print(f"Error: Invalid YAML in config file {config_path}")
        sys.exit(1)

def create_genai_client(api_key):
    return genai.Client(api_key=api_key)

def generate_response(client, prompt, model="gemini-2.0-flash"):
    try:
        return client.models.generate_content_stream(
            model=model, 
            contents=[prompt]
        )
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

# Display the response in a more accurate stream
def process_response(response):
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
    
    # Load configuration
    config = load_config()
    
    # Access API key
    api_key = config['api']['key']
    
    # Create client
    client = create_genai_client(api_key)
    
    # Generate and process response
    prompt = "What's the weather like?"
    response = generate_response(client, prompt)
    process_response(response)

if __name__ == "__main__":
    main()

