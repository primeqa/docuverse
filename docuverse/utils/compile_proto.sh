#!/bin/bash
# Script to compile protobuf files for the Milvus gRPC service
#
# Usage:
#   From project root: bash src/retrievers/compile_proto.sh
#   From src/retrievers: bash compile_proto.sh
#   From anywhere: bash /path/to/src/retrievers/compile_proto.sh
#
# This script will:
#   1. Check for grpcio-tools and install if needed
#   2. Compile protos/milvus_service.proto
#   3. Generate milvus_service_pb2.py and milvus_service_pb2_grpc.py
#   4. Fix import statements in generated files

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="${SCRIPT_DIR}/protos"
OUTPUT_DIR="${SCRIPT_DIR}"

echo -e "${GREEN}=== Milvus gRPC Proto Compilation Script ===${NC}\n"

# Check if grpcio-tools is installed
echo "Checking for grpcio-tools..."
if ! python -c "import grpc_tools.protoc" 2>/dev/null; then
    echo -e "${RED}Error: grpcio-tools is not installed${NC}"
    echo -e "${YELLOW}Installing grpcio-tools...${NC}"
    pip install grpcio-tools grpcio
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install grpcio-tools${NC}"
        exit 1
    fi
    echo -e "${GREEN}grpcio-tools installed successfully${NC}\n"
else
    echo -e "${GREEN}grpcio-tools is installed${NC}\n"
fi

# Check if proto file exists
PROTO_FILE="${PROTO_DIR}/milvus_service.proto"
if [ ! -f "${PROTO_FILE}" ]; then
    echo -e "${RED}Error: Proto file not found at ${PROTO_FILE}${NC}"
    exit 1
fi

echo "Proto file: ${PROTO_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Compile the proto file
echo -e "${YELLOW}Compiling proto file...${NC}"
python -m grpc_tools.protoc \
    --proto_path="${PROTO_DIR}" \
    --python_out="${OUTPUT_DIR}" \
    --grpc_python_out="${OUTPUT_DIR}" \
    "${PROTO_FILE}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Proto compilation successful!${NC}\n"

    # List generated files
    echo "Generated files:"
    ls -lh "${OUTPUT_DIR}"/milvus_service_pb2*.py 2>/dev/null || echo -e "${YELLOW}No generated files found (this might be an error)${NC}"
    echo ""

    # Fix imports in generated files (common issue with protobuf)
    echo -e "${YELLOW}Checking generated files for import issues...${NC}"

    # Check if _pb2_grpc.py exists and fix its imports
    GRPC_FILE="${OUTPUT_DIR}/milvus_service_pb2_grpc.py"
    if [ -f "${GRPC_FILE}" ]; then
        # Replace absolute import with relative import
        sed -i 's/^import milvus_service_pb2/from . import milvus_service_pb2/' "${GRPC_FILE}" 2>/dev/null || \
        sed -i '' 's/^import milvus_service_pb2/from . import milvus_service_pb2/' "${GRPC_FILE}" 2>/dev/null || \
        echo -e "${YELLOW}Note: Could not automatically fix imports (sed may not be available)${NC}"

        echo -e "${GREEN}✓ Import fixes applied${NC}"
    fi

    echo ""
    echo -e "${GREEN}=== Compilation Complete ===${NC}"
    echo ""
    echo "You can now use the generated files:"
    echo "  - milvus_service_pb2.py (message definitions)"
    echo "  - milvus_service_pb2_grpc.py (service definitions)"

else
    echo -e "${RED}✗ Proto compilation failed!${NC}"
    exit 1
fi
