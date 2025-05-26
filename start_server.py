#!/usr/bin/env python3
import uvicorn
import sys
import os

def main():
    print("Starting GPT API server...")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Environment variables:")
    for key in ['OPENAI_API_KEY', 'PORT', 'HOST']:
        value = os.environ.get(key, 'Not set')
        if 'API_KEY' in key and value != 'Not set':
            value = f"{value[:8]}..." if len(value) > 8 else "***"
        print(f"  {key}: {value}")
    
    # Get port from environment or default to 8000
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print(f"Starting server on {host}:{port}")
    
    try:
        uvicorn.run(
            "gpt_deploy:app",
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 