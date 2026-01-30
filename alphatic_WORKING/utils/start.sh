#!/bin/bash

# Alphatic Portfolio Analyzer - Quick Start Script

echo "================================================"
echo "Alphatic Portfolio Analyzer - Quick Start"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 found"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo ""
echo "📥 Installing dependencies..."
echo "This may take a few minutes..."
echo ""
pip install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All dependencies installed successfully!"
    echo ""
else
    echo ""
    echo "❌ Error installing dependencies. Please check requirements.txt"
    exit 1
fi

# Run the app
echo "================================================"
echo "🚀 Starting Alphatic Portfolio Analyzer..."
echo "================================================"
echo ""
echo "📱 The app will open in your browser at:"
echo "   http://localhost:8501"
echo ""
echo "💡 Tip: Press Ctrl+C to stop the server"
echo ""

streamlit run alphatic_portfolio_app.py
