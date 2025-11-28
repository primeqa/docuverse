#!/bin/bash

echo "=================================================="
echo "BioHash Installation Check"
echo "=================================================="
echo ""

# Check Python version
echo "Python Version:"
python --version
echo ""

# Check required packages
echo "Required Dependencies:"
echo "------------------------------------------------"

check_package() {
    python -c "import $1; print('✓ $1 installed:', $1.__version__)" 2>/dev/null || echo "✗ $1 NOT installed"
}

check_package "torch"
check_package "numpy"
check_package "sklearn"

echo ""
echo "Optional Dependencies:"
echo "------------------------------------------------"
check_package "sentence_transformers"
check_package "torchvision"
check_package "matplotlib"

echo ""
echo "=================================================="
echo "Installation Instructions"
echo "=================================================="
echo ""
echo "To install minimal dependencies (required):"
echo "  pip install torch numpy scikit-learn"
echo ""
echo "To install all dependencies (recommended):"
echo "  pip install torch numpy scikit-learn sentence-transformers torchvision matplotlib"
echo ""
echo "Or use conda:"
echo "  conda install pytorch numpy scikit-learn -c pytorch"
echo "  pip install sentence-transformers"
echo ""

# Check if files exist
echo "=================================================="
echo "BioHash Files Check"
echo "=================================================="
echo ""

files=(
    "biohash_implementation.py"
    "biohash_text.py"
    "biohash_docuverse.py"
    "test_text_hashing.py"
    "README.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "✓ $file ($lines lines, $size)"
    else
        echo "✗ $file MISSING"
    fi
done

echo ""
echo "=================================================="
echo "Total:"
ls -1 *.py *.md 2>/dev/null | wc -l | xargs echo "  Files:"
du -sh . | awk '{print "  Size: "$1}'
echo "=================================================="
