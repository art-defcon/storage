#!/bin/bash
# Setup script for the Storage Segmentation App

# Exit on error
set -e

# Print commands
set -x

# Create directories
mkdir -p data/models
mkdir -p data/test
mkdir -p data/samples

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Python version $python_version is less than the required version $required_version"
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download sample images
echo "Downloading sample images..."
# TODO

# Create a simple README in the samples directory
cat > data/samples/README.txt << EOL
Sample Images for Storage Segmentation App

These images are downloaded from Unsplash and are free to use.
They are provided as examples to test the Storage Segmentation App.

Sources:
- bookshelf.jpg: Photo by Brina Blum on Unsplash
- cabinet.jpg: Photo by Kam Idris on Unsplash
- dresser.jpg: Photo by Spacejoy on Unsplash
EOL

echo "Setup complete!"
echo ""
echo "To run the application, use the following command:"
echo "source venv/bin/activate && streamlit run app.py"
echo ""
echo "To run the example script, use:"
echo "source venv/bin/activate && python example.py --image data/samples/bookshelf.jpg --output_dir output --compartments"
echo ""
echo "To run the tests, use:"
echo "source venv/bin/activate && python test.py"