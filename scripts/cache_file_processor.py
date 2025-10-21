#!/usr/bin/env python3

import os
import re
import sys
import argparse
from docuverse.engines.search_data import SearchData

def process_cache_files(cache_file_list):
    """
    Process a list of cache files and print info about each dataset.
    
    Args:
        cache_file_list: List of cache file paths to process
    """
    print(f"{'Dataset':<60} | {'Context size':<60} | {'Num tiles':<12}")
    print("-" * 75)

    format = re.compile(r"benchmark__(.*?)__corpus.jsonl_(\d+)_")
    
    for cache_file in cache_file_list:
        # Get the corresponding input file (assuming same name with different extension)
        # You may need to adjust this logic based on your specific naming convention
        # input_file = os.path.splitext(cache_file)[0] + ".json"
        input_file = cache_file
        
        # Skip if cache file doesn't exist
        if not os.path.exists(cache_file):
            print(f"{cache_file:<60} | File not found")
            continue
            
        # Read the dataset from cache
        try:
            dataset = SearchData.read_cache_file_if_needed(cache_file, None)
            
            # Determine dataset size
            if dataset is None:
                size = 0
            elif isinstance(dataset, list) or isinstance(dataset, SearchData):
                size = len(dataset)
            else:
                size = 1  # Object that's not a list

            m = format.search(cache_file)
            print(f"{m.group(1):<60} | {m.group(2):<60} | {size:<12}")
            
        except Exception as e:
            print(f"{os.path.basename(cache_file):<60} | Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Process cache files and display dataset sizes")
    parser.add_argument("files", nargs="+", help="Cache files to process")
    parser.add_argument("--input-dir", help="Directory containing input files if different from cache files")
    
    args = parser.parse_args()
    
    # Process the cache files
    process_cache_files(args.files)

if __name__ == "__main__":
    main()
