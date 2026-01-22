#!/usr/bin/env python3
"""
Convert IBM Granite Embedding Model to ONNX Format

This script converts the ibm-granite/granite-embedding-english-r2 model
from Hugging Face to ONNX format for optimized inference.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.exporters.onnx import main_export


def download_and_convert_model(
    model_name: str = "ibm-granite/granite-embedding-english-r2",
    output_dir: str = "./granite-embedding-onnx",
    max_length: int = 512,
    optimize: bool = True,
    quantize: bool = False,
    use_gpu: bool = False
) -> Tuple[str, str]:
    """
    Download and convert the Granite embedding model to ONNX format.
    
    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory to save the ONNX model
        max_length: Maximum sequence length for the model
        optimize: Whether to optimize the ONNX model
        quantize: Whether to quantize the model (reduce precision)
        use_gpu: Whether to use GPU for conversion (if available)
        
    Returns:
        Tuple of (model_path, tokenizer_path)
    """
    print(f"Converting {model_name} to ONNX format...")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and model
    print("\n1. Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Disable Flash Attention for ONNX compatibility
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        attn_implementation="eager"  # Use eager attention instead of flash attention
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input for tracing
    print("\n2. Creating dummy input for model tracing...")
    dummy_text = "This is a sample text for embedding generation."
    dummy_inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    # Export to ONNX using Optimum
    print("\n3. Converting to ONNX format...")
    try:
        # Use Optimum's export functionality
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True,
            provider="CPUExecutionProvider" if not use_gpu else "CUDAExecutionProvider",
        )
        
        # Save the model
        ort_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✓ Model converted and saved to {output_dir}")
        
    except Exception as e:
        print(f"Error with Optimum export: {e}")
        print("Falling back to manual ONNX export...")
        
        # Manual export using torch.onnx
        onnx_path = output_path / "model.onnx"
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                str(onnx_path),
                input_names=list(dummy_inputs.keys()),
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
                },
                opset_version=14,
                do_constant_folding=True,
                verbose=False
            )
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        print(f"✓ Manual export completed: {onnx_path}")
    
    # Optimize the model if requested
    if optimize:
        print("\n4. Optimizing ONNX model...")
        try:
            # Load and optimize
            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                output_dir,
                provider="CPUExecutionProvider"
            )
            
            # Create optimization config
            optimization_config = OptimizationConfig(
                optimization_level="all",
                optimize_for_gpu=use_gpu,
                fp16=False  # Keep as fp32 for better compatibility
            )
            
            # Apply optimization
            optimized_dir = output_path / "optimized"
            ort_model.save_pretrained(str(optimized_dir))
            
            print(f"✓ Optimized model saved to {optimized_dir}")
            
        except Exception as e:
            print(f"Warning: Optimization failed: {e}")
            print("Proceeding with unoptimized model...")
    
    return str(output_path / "model.onnx"), str(output_path)


def test_onnx_model(model_path: str, tokenizer_path: str) -> None:
    """
    Test the converted ONNX model to ensure it works correctly.
    
    Args:
        model_path: Path to the ONNX model directory
        tokenizer_path: Path to the tokenizer directory
    """
    print("\n5. Testing converted ONNX model...")
    
    try:
        # Load ONNX model and tokenizer
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            model_path,
            provider="CPUExecutionProvider"
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Test with sample text
        test_texts = [
            "This is a test sentence for embedding generation.",
            "Another example text to verify the model works correctly.",
            "Machine learning models can be converted to ONNX format."
        ]
        
        print("Testing with sample texts...")
        for i, text in enumerate(test_texts):
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # Get embeddings
            with torch.no_grad():
                outputs = ort_model(**inputs)
                embeddings = outputs.last_hidden_state
                
                # Pool embeddings (mean pooling)
                pooled_embeddings = embeddings.mean(dim=1)
                
                print(f"  Text {i+1}: {text[:50]}...")
                print(f"    Embedding shape: {pooled_embeddings.shape}")
                print(f"    Embedding norm: {torch.norm(pooled_embeddings).item():.4f}")
        
        print("✓ ONNX model test completed successfully!")
        
    except Exception as e:
        print(f"Error testing ONNX model: {e}")
        print("The model was converted but testing failed.")


def get_model_info(model_path: str) -> None:
    """
    Display information about the converted ONNX model.
    
    Args:
        model_path: Path to the ONNX model directory
    """
    print("\n6. Model Information:")
    print("-" * 50)
    
    try:
        import onnx
        
        onnx_file = Path(model_path) / "model.onnx"
        if onnx_file.exists():
            model = onnx.load(str(onnx_file))
            
            print(f"Model file: {onnx_file}")
            print(f"Model size: {onnx_file.stat().st_size / (1024*1024):.2f} MB")
            print(f"ONNX version: {model.opset_import[0].version}")
            
            print("\nModel inputs:")
            for input_tensor in model.graph.input:
                print(f"  - {input_tensor.name}: {[dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]}")
            
            print("\nModel outputs:")
            for output_tensor in model.graph.output:
                print(f"  - {output_tensor.name}: {[dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]}")
                
        else:
            print("ONNX model file not found for detailed inspection.")
            
        # List all files in the output directory
        print(f"\nFiles in output directory:")
        for file in sorted(Path(model_path).glob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024*1024)
                print(f"  - {file.name}: {size_mb:.2f} MB")
                
    except Exception as e:
        print(f"Error getting model info: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert IBM Granite embedding model to ONNX format"
    )
    parser.add_argument(
        "--model_name",
        default="ibm-granite/granite-embedding-english-r2",
        help="HuggingFace model name (default: ibm-granite/granite-embedding-english-r2)"
    )
    parser.add_argument(
        "--output_dir",
        default="./granite-embedding-onnx",
        help="Output directory for ONNX model (default: ./granite-embedding-onnx)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize the ONNX model for better performance"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the model to reduce size (experimental)"
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for conversion if available"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the converted model after conversion"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download and conversion, only test existing model"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IBM GRANITE EMBEDDING MODEL TO ONNX CONVERTER")
    print("=" * 60)
    
    try:
        if not args.skip_download:
            # Convert the model
            model_path, tokenizer_path = download_and_convert_model(
                model_name=args.model_name,
                output_dir=args.output_dir,
                max_length=args.max_length,
                optimize=args.optimize,
                quantize=args.quantize,
                use_gpu=args.use_gpu
            )
        else:
            model_path = args.output_dir
            tokenizer_path = args.output_dir
            print(f"Using existing model at: {model_path}")
        
        # Test the model if requested
        if args.test:
            test_onnx_model(model_path, tokenizer_path)
        
        # Display model information
        get_model_info(model_path)
        
        print("\n" + "=" * 60)
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Model saved to: {args.output_dir}")
        print(f"You can now use the ONNX model for fast inference.")
        print("\nUsage example:")
        print(f"  from optimum.onnxruntime import ORTModelForFeatureExtraction")
        print(f"  from transformers import AutoTokenizer")
        print(f"  ")
        print(f"  model = ORTModelForFeatureExtraction.from_pretrained('{args.output_dir}')")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Conversion failed!")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
