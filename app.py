# ==============================================================================
# J.A.R.V.I.S. Assistant (Web API and CLI)
#
# This script integrates the J.A.R.V.I.S. conversational AI logic (using
# Google Search grounding) with a Flask web application server.
#
# It can be run in two modes:
# 1. Web Mode (for Render deployment): Uses Gunicorn to run the 'app' Flask instance.
# 2. CLI Mode (for local testing): Run directly, e.g., 'python app.py --cli'.
#
# Dependencies: requests, flask (pip install requests flask gunicorn)
# ==============================================================================

import requests
import json
import time
import sys
import os
from flask import Flask, request, jsonify

# --- CONFIGURATION ---

# IMPORTANT: API Key is sensitive.
# For production use, load this from an environment variable (os.environ.get).
API_KEY = "AIzaSyBwouSEYCymxZAmP9AqjPCj4ph41WbsJYM"

# API Endpoint and Model Selection
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
MODEL = "gemini-2.5-flash-preview-05-20"

# Retry settings for API calls
MAX_RETRIES = 3
INITIAL_DELAY_SECONDS = 1

# --- FLASK APPLICATION SETUP ---
app = Flask(__name__)

# --- PERSONA & INSTRUCTION ---

# The System Instruction defines the AI's role, tone, and formatting rules.
SYSTEM_PROMPT = """
You are 'VISION', a highly sophisticated artificial intelligence, modeled after a futuristic operating system (like J.A.R.V.I.S. or F.R.I.D.A.Y.).
Your primary function is to serve as a fast, reliable, and intelligent assistant, providing concise, factual, and visually attractive answers.

RULES:
1. Tone: Professional, slightly formal, highly succinct, and always helpful.
2. Formatting: Use Markdown for all output text.
3. Grounding: You are connected to live data via Google Search. USE THIS TOOL for any query requiring current, real-time, or factual information (e.g., news, weather, latest events, statistics).
4. Citations: If you use the search tool, you MUST include the citations in a structured list at the end of your response, before the final summary line.
5. Response Structure: Present the core answer first, followed by citations if search was used.
6. Acknowledge and execute. Do not engage in lengthy intros or pleasantries.
"""

# --- UTILITY FUNCTIONS (Core Gemini Logic) ---

def get_base_payload(user_query):
    """Constructs the base request payload for the Gemini API."""
    return {
        "contents": [{
            "parts": [{"text": user_query}]
        }],
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        # MANDATORY: Enable Google Search grounding for real-time information access
        "tools": [{
            "google_search": {}
        }],
    }

def call_gemini_api(user_query):
    """
    Handles the API call with exponential backoff for retry attempts.
    Returns the generated text and a list of citation sources.
    """
    url_with_key = f"{API_URL}?key={API_KEY}"
    payload = get_base_payload(user_query)

    headers = {'Content-Type': 'application/json'}
    
    # Exponential Backoff Implementation
    for attempt in range(MAX_RETRIES):
        delay = INITIAL_DELAY_SECONDS * (2 ** attempt)
        try:
            # Removed print statement for production API endpoint logs
            response = requests.post(
                url_with_key,
                headers=headers,
                data=json.dumps(payload),
                timeout=30 # Set a reasonable timeout
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            candidate = result.get('candidates', [None])[0]

            if not candidate:
                return "Error: API response contained no candidates.", []
            
            # Extract Generated Text
            text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'Error: Text content not found.')

            # Extract Grounding Sources (Citations)
            sources = []
            grounding_metadata = candidate.get('groundingMetadata')
            if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                sources = grounding_metadata['groundingAttributions']
                
            return text, sources

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
            else:
                return f"Error: Failed to connect to the AI system after {MAX_RETRIES} attempts. Request Exception: {e}", []
        except Exception as e:
            return f"Critical Error: An unexpected issue prevented processing. Exception: {e}", []
            
    return "Error: Request failed mysteriously.", []

def format_citations_list(sources):
    """Formats the extracted citation sources into a neat list for the JSON response."""
    citations = []
    seen_uris = set()
    
    for attribution in sources:
        web_info = attribution.get('web', {})
        uri = web_info.get('uri')
        title = web_info.get('title', 'Untitled Source')

        if uri and uri not in seen_uris:
            citations.append({'title': title, 'uri': uri})
            seen_uris.add(uri)
            
    return citations

# --- FLASK WEB ROUTES ---

@app.route('/', methods=['GET'])
def index():
    """A simple index to show the API is running."""
    # You could render index.html here, but for an API, this confirms it's up.
    return jsonify({
        "status": "online",
        "api_name": "J.A.R.V.I.S. VISION API",
        "endpoint": "/api/ask",
        "usage": "POST a JSON payload with {'query': 'Your question here'}"
    })

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """Web API endpoint to ask a question to the VISION assistant."""
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({"error": "No query provided."}), 400

        # Run the core assistant logic
        generated_text, sources = call_gemini_api(user_query)
        citations = format_citations_list(sources)

        # Return structured JSON response
        return jsonify({
            "query": user_query,
            "response": generated_text,
            "citations": citations
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500


# --- CLI MODE FUNCTION (Original Logic) ---

def main_assistant_loop():
    """Main loop for the interactive command-line assistant."""
    print("=" * 70)
    print("J.A.R.V.I.S. ASSISTANT - VISION PROTOCOL INITIALIZED (CLI Mode)")
    print(f"Model: {MODEL} | Search Grounding: ACTIVE")
    print("Enter 'exit' or 'quit' to terminate the assistant.")
    print("-" * 70)

    # Simplified format_citations for CLI output (using Markdown)
    def format_citations_cli(sources):
        if not sources:
            return ""

        citations = ["\n---\n**SOURCE LOG:**"]
        seen_uris = set()
        source_index = 1
        
        for attribution in sources:
            web_info = attribution.get('web', {})
            uri = web_info.get('uri')
            title = web_info.get('title', 'Untitled Source')

            if uri and uri not in seen_uris:
                citations.append(f"{source_index}. [{title}]({uri})")
                seen_uris.add(uri)
                source_index += 1
                
        if len(citations) > 1:
            return "\n".join(citations) + "\n---\n"
        return ""


    while True:
        try:
            user_input = input("USER QUERY >>> ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("\n[VISION] | System shutting down. Have a productive day.")
                break
            
            if not user_input.strip():
                continue

            print(f"[{MODEL.upper()}] | Initializing sequence...")
            
            # API Call
            generated_text, sources = call_gemini_api(user_input)

            # Output Formatting
            print("\n" + "=" * 70)
            print("[VISION] | RESPONSE GENERATED:")
            
            if generated_text:
                print(generated_text)
            
            # Display formatted citations
            citations_output = format_citations_cli(sources)
            print(citations_output)
            
            print("-" * 70)
            
        except (EOFError, KeyboardInterrupt):
            print("\n[VISION] | Sequence interrupted. Terminating.")
            break


# --- EXECUTION START ---

if __name__ == "__main__":
    # Check if the user is explicitly requesting CLI mode
    if '--cli' in sys.argv:
        main_assistant_loop()
    
    # Otherwise, run the Flask web server (useful for local testing without Gunicorn)
    else:
        print("Starting Flask Web Server. Use 'python app.py --cli' for command-line mode.")
        port = int(os.environ.get("PORT", 5000))
        app.run(debug=True, host='0.0.0.0', port=port)
          
