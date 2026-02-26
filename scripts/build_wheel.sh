#!/bin/bash
# Build and optionally upload docuverse wheel to PyPI
# Usage:
#   ./build_wheel.sh                    # Build only
#   ./build_wheel.sh --test             # Build and upload to TestPyPI
#   ./build_wheel.sh --upload           # Build and upload to PyPI
#   ./build_wheel.sh --clean            # Clean build artifacts only

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to clean build artifacts
clean_build() {
    print_info "Cleaning build artifacts..."
    rm -rf dist/ build/ *.egg-info
    print_success "Build artifacts cleaned"
}

# Function to build the wheel
build_wheel() {
    print_info "Building wheel..."

    # Check if build module is installed
    if ! python -m build --version &>/dev/null; then
        print_warning "build module not found, installing..."
        pip install build
    fi

    # Build the wheel
    python -m build --wheel

    if [ $? -eq 0 ]; then
        print_success "Wheel built successfully"

        # Show the built wheel
        WHEEL_FILE=$(ls -t dist/*.whl | head -1)
        if [ -f "$WHEEL_FILE" ]; then
            print_info "Built wheel: $WHEEL_FILE"
            ls -lh "$WHEEL_FILE"
        fi
    else
        print_error "Wheel build failed"
        exit 1
    fi
}

# Function to check if twine is installed
check_twine() {
    if ! python -m twine --version &>/dev/null; then
        print_warning "twine not found, installing..."
        pip install twine
    fi
}

# Function to upload to TestPyPI
upload_test() {
    print_info "Uploading to TestPyPI..."
    check_twine

    print_warning "You will need TestPyPI credentials."
    print_info "If you don't have an account, create one at: https://test.pypi.org/account/register/"

    python -m twine upload --repository testpypi dist/*

    if [ $? -eq 0 ]; then
        print_success "Successfully uploaded to TestPyPI"
        VERSION=$(cat VERSION)
        print_info "Test installation with:"
        echo -e "  ${GREEN}pip install -i https://test.pypi.org/simple/ docuverse==${VERSION}${NC}"
    else
        print_error "Upload to TestPyPI failed"
        exit 1
    fi
}

# Function to upload to PyPI
upload_pypi() {
    print_info "Uploading to PyPI..."
    check_twine

    print_warning "You are about to upload to production PyPI!"
    read -p "Are you sure you want to continue? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        print_warning "Upload cancelled"
        exit 0
    fi

    print_warning "You will need PyPI credentials."
    print_info "If you don't have an account, create one at: https://pypi.org/account/register/"

    python -m twine upload dist/*

    if [ $? -eq 0 ]; then
        print_success "Successfully uploaded to PyPI"
        VERSION=$(cat VERSION)
        print_info "Install with:"
        echo -e "  ${GREEN}pip install docuverse==${VERSION}${NC}"
    else
        print_error "Upload to PyPI failed"
        exit 1
    fi
}

# Function to verify wheel contents
verify_wheel() {
    print_info "Verifying wheel contents..."

    WHEEL_FILE=$(ls -t dist/*.whl | head -1)
    if [ ! -f "$WHEEL_FILE" ]; then
        print_error "No wheel file found"
        exit 1
    fi

    print_info "Checking for required packages..."

    # Check if docuverse package is included
    if python -m zipfile -l "$WHEEL_FILE" | grep -q "docuverse/__init__.py"; then
        print_success "✓ docuverse package found"
    else
        print_error "✗ docuverse package not found"
        exit 1
    fi

    # Check if faiss engine is included
    if python -m zipfile -l "$WHEEL_FILE" | grep -q "docuverse/engines/retrieval/faiss/"; then
        print_success "✓ FAISS engine included"
    else
        print_warning "✗ FAISS engine not found"
    fi

    # Show package count
    FILE_COUNT=$(python -m zipfile -l "$WHEEL_FILE" | grep -c "\.py$" || true)
    print_info "Total Python files: $FILE_COUNT"

    # Show metadata
    print_info "Package metadata:"
    python -m zipfile -l "$WHEEL_FILE" | grep "METADATA" | head -1
    unzip -p "$WHEEL_FILE" "*/METADATA" | grep -E "^(Name|Version|Summary):" || true
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build docuverse wheel and optionally upload to PyPI.

OPTIONS:
    --clean         Clean build artifacts only (no build)
    --test          Build and upload to TestPyPI
    --upload        Build and upload to production PyPI
    --verify        Build and verify wheel contents
    --help          Show this help message

EXAMPLES:
    $0                  # Build wheel only
    $0 --verify         # Build and verify wheel contents
    $0 --test           # Build and upload to TestPyPI
    $0 --upload         # Build and upload to PyPI
    $0 --clean          # Clean build artifacts

NOTES:
    - For TestPyPI upload, set TWINE_USERNAME and TWINE_PASSWORD env vars
      or use API token via TWINE_USERNAME=__token__ and TWINE_PASSWORD=<token>
    - For PyPI upload, same authentication applies
    - API tokens can be generated at:
      * TestPyPI: https://test.pypi.org/manage/account/token/
      * PyPI: https://pypi.org/manage/account/token/

EOF
}

# Main script logic
main() {
    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found. Please run this script from the project root."
        exit 1
    fi

    if [ ! -f "VERSION" ]; then
        print_error "VERSION file not found."
        exit 1
    fi

    VERSION=$(cat VERSION)
    print_info "Building docuverse version: $VERSION"
    echo ""

    # Parse command line arguments
    ACTION="build"

    case "${1:-}" in
        --clean)
            clean_build
            exit 0
            ;;
        --test)
            ACTION="test"
            ;;
        --upload)
            ACTION="upload"
            ;;
        --verify)
            ACTION="verify"
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        "")
            ACTION="build"
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac

    # Always clean and build first
    clean_build
    echo ""
    build_wheel
    echo ""

    # Execute the requested action
    case "$ACTION" in
        verify)
            verify_wheel
            ;;
        test)
            verify_wheel
            echo ""
            upload_test
            ;;
        upload)
            verify_wheel
            echo ""
            upload_pypi
            ;;
        build)
            verify_wheel
            echo ""
            print_success "Build complete! Wheel is ready in dist/"
            print_info "To upload to TestPyPI: $0 --test"
            print_info "To upload to PyPI: $0 --upload"
            ;;
    esac
}

# Run main function
main "$@"
