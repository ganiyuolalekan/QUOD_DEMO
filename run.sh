#!/bin/bash

# QUOD Task - Documentation Processor
# Run script for starting the Streamlit application

echo "🚀 Starting QUOD Task - Documentation Processor"
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Set environment variables if not already set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set. AI features will use fallback processing."
    echo "   To enable AI processing, set your OpenAI API key:"
    echo "   export OPENAI_API_KEY='your-api-key-here'"
    echo ""
fi

# Start Streamlit app
echo "🌐 Starting Streamlit application..."
echo "   Access the app at: http://localhost:8501"
echo "   Press Ctrl+C to stop the application"
echo ""

streamlit run app.py
