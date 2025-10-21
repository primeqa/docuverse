#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

def convert_json_to_jsonl(input_file, output_file=None):
    """
    Convert a JSON file to JSONL format with UTF-8 encoding.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str, optional): Path to the output JSONL file. If None, 
                                     creates a file with the same name but .jsonl extension.
    
    Returns:
        str: Path to the created JSONL file
    """
    # Determine output file path if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix('.jsonl'))
    
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to JSONL format
        with open(output_file, 'w', encoding='utf-8') as f:
            if isinstance(data, list):
                # Handle array/list of objects
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            elif isinstance(data, dict):
                # Handle single object
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
            else:
                raise ValueError("Input JSON must be either an object or an array")
                
        print(f"Successfully converted {input_file} to {output_file}")
        return output_file
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Convert JSON file to JSONL format with UTF-8 encoding')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('-o', '--output-file', help='Path to the output JSONL file (default: input filename with .jsonl extension)')
    
    args = parser.parse_args()
    
    convert_json_to_jsonl(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
