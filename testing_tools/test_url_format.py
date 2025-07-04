#!/usr/bin/env python3

def format_llm_api_url(url):
    """Ensures the LLM API URL is properly formatted."""
    # Add http:// if no scheme is present
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    
    # Make sure the URL ends with /v1/chat/completions
    if not url.endswith("/v1/chat/completions"):
        # Remove trailing slash if present
        if url.endswith("/"):
            url = url[:-1]
        
        # Check if URL already contains part of the endpoint
        if "/v1/chat" in url:
            # Extract base path up to /v1/chat
            base_path = url.split("/v1/chat")[0]
            url = f"{base_path}/v1/chat/completions"
        elif "/v1" in url:
            # Extract base path up to /v1
            base_path = url.split("/v1")[0]
            url = f"{base_path}/v1/chat/completions"
        else:
            # Just append the full endpoint path
            url = f"{url}/v1/chat/completions"
    
    return url

# Test cases
test_cases = [
    "localhost:8000",
    "api.example.com/v1",
    "https://openai.com",
    "http://localhost:8000/v1/chat/completions",
    "localhost:8000/v1/chat",
    "https://api.example.com/v1",
    "api.openai.com",
    "http://localhost:8000/",
    "localhost:8000/v1/chat/something/else",
]

print("Testing URL formatting function:")
for url in test_cases:
    formatted = format_llm_api_url(url)
    print(f"\nInput:  {url}")
    print(f"Output: {formatted}")
