#!/bin/bash
# Setup script for Lambda Cloud environment

set -e  # Exit on error

echo "========================================="
echo "Pattern Segmentation Training Setup"
echo "========================================="
echo ""

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Not in a virtual environment. Creating one..."

    # Create virtual environment
    python3 -m venv venv

    echo "✓ Virtual environment created"
    echo ""
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
    echo ""
else
    echo "✓ Already in virtual environment: $VIRTUAL_ENV"
    echo ""
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo ""

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt
echo ""

# Verify installation
echo "========================================="
echo "Verifying installation..."
echo "========================================="
echo ""

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "CUDA version: N/A"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import albumentations; print(f'Albumentations version: {albumentations.__version__}')"

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "To test your setup:"
echo "  python test_setup.py --coco-dir /path/to/data --images-dir /path/to/data"
echo ""
echo "To start training:"
echo "  ./run_training.sh"
echo ""
