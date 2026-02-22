#!/bin/bash

# MDKeyChunker Setup Script
# This script helps you set up the MDKeyChunker package

set -e

echo "=================================================="
echo "MDKeyChunker Setup"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install package
echo ""
echo "Installing MDKeyChunker..."
pip install -e .

# Ask about optional dependencies
echo ""
read -p "Install spaCy model for entity extraction? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
    echo "✓ spaCy model installed"
fi

# Ask about dev dependencies
echo ""
read -p "Install development dependencies (for testing)? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing dev dependencies..."
    pip install -e ".[dev]"
    echo "✓ Dev dependencies installed"
fi

# Set up .env file
echo ""
if [ -f ".env" ]; then
    echo ".env file already exists. Skipping..."
else
    read -p "Create .env file from template? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp .env.sample .env
        echo "✓ Created .env file"
        echo ""
        echo "⚠️  IMPORTANT: Edit .env and add your LLM API key!"
    fi
fi

# Test installation
echo ""
echo "Testing installation..."
python -c "from mdkeychunker import DocumentProcessor; print('✓ Import successful')"

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run example: python examples.py"
echo "3. Process a document: mdkeychunker demo.md"
echo ""
echo "For more information, see README.md"
echo ""
