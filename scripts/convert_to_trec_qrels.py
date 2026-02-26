#!/usr/bin/env python3
"""
Convert retrieval system output to TREC qrels format.

This script takes JSON/JSONL output from a retrieval system and produces two TREC-style qrels files:
1. Gold standard file (from question.relevant)
2. System output file (from retrieved_passages[])

TREC qrels format:
query_id  iteration  doc_id  relevance

Author: Claude Code
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set


def remove_last_two_numbers(doc_id: str) -> str:
    """
    Remove the last two dash-separated numbers from a document ID.

    Example: 818165744_68-964-0-848 -> 818165744_68-964

    Args:
        doc_id: Document ID string

    Returns:
        Document ID with last two numbers removed
    """
    parts = doc_id.rsplit('-', 2)
    if len(parts) == 3:
        return parts[0]
    return doc_id


def write_trec_qrels(query_ids, output_file: Path, qrels: Dict[str, Dict[str, int]], iteration: int = 0):
    """
    Writes TREC-style qrels data to a specified output file.

    This function processes query relevance judgments (qrels) for a set of queries and
    writes them in TREC qrels format. Qrels data includes information about which
    documents map to which queries and their respective relevance scores. The output
    follows the format: `query_id iteration doc_id relevance`.

    Args:
        query_ids (List[str]): A list of query IDs for which relevance data is to be written.
        output_file (Path): Path to the file where the qrels data will be saved.
        qrels (Dict[str, Dict[str, int]]): A dictionary containing the relevance judgments
            where the outer key is the query ID, inner key is the document ID, and the
            value is the relevance score.
        iteration (int): The iteration number to include in the output. Defaults to 0.
    """
    with open(output_file, 'w') as f:
        for query_id in query_ids: # sorted(qrels.keys()):
            for doc_id in sorted(qrels[query_id].keys(), key=lambda x: qrels[query_id][x], reverse=True):
                relevance = qrels[query_id][doc_id]
                f.write(f"{query_id}\t{iteration}\t{doc_id}\t{relevance}\n")


def process_retrieval_output(input_file: Path, gold_output: Path, system_output: Path,
                             top_k: int = None, iteration: int = 0,
                             question_id_field: str = "task_id",
                             relevant_field: str = "relevant"):
    """
    Process retrieval system output and create TREC qrels files.

    Args:
        input_file: Input JSON/JSONL file
        gold_output: Output file for gold standard qrels
        system_output: Output file for system output qrels
        top_k: Only include top-k retrieved passages (default: all)
        iteration: Iteration number for TREC format (default: 0)
        question_id_field: Field name for question ID (default: "task_id")
        relevant_field: Field name for relevant docs (default: "relevant")
    """
    gold_qrels = {}
    system_qrels = {}
    query_ids = []

    # Determine if file is JSON or JSONL
    is_jsonl = input_file.suffix == '.jsonl' or input_file.name.endswith('.jsonl')

    if is_jsonl:
        # Process JSONL (one JSON object per line)
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    process_entry(entry, query_ids, gold_qrels, system_qrels, top_k,
                                question_id_field, relevant_field, line_num)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)
    else:
        # Process regular JSON
        with open(input_file, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for idx, entry in enumerate(data):
                        process_entry(entry, query_ids, gold_qrels, system_qrels, top_k,
                                    question_id_field, relevant_field, idx)
                else:
                    process_entry(data, query_ids, gold_qrels, system_qrels, top_k,
                                question_id_field, relevant_field)
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
                sys.exit(1)

    # Write output files
    write_trec_qrels(query_ids, gold_output, gold_qrels, iteration)
    write_trec_qrels(query_ids, system_output, system_qrels, iteration)

    print(f"Processed {len(gold_qrels)} queries")
    print(f"Gold qrels written to: {gold_output}")
    print(f"System qrels written to: {system_output}")


def process_entry(entry: dict, query_ids,
                  gold_qrels: Dict, system_qrels: Dict,
                  top_k: int, question_id_field: str, relevant_field: str,
                  entry_idx: int = None):
    """
    Process a single entry from the retrieval output.

    Args:
        entry: Dictionary containing question and retrieval results
        gold_qrels: Dictionary to store gold qrels
        system_qrels: Dictionary to store system qrels
        top_k: Maximum number of results to include
        question_id_field: Field name for question ID
        relevant_field: Field name for relevant docs
        entry_idx: Entry index for error messages
    """
    # Extract question information
    question = entry.get('question', {})
    if not question:
        question = entry  # If no 'question' field, assume entry is the question

    # Get query ID
    query_id = question.get(question_id_field)
    query_ids.append(query_id)
    if not query_id:
        print(f"Warning: Entry {entry_idx} missing '{question_id_field}' field, skipping",
              file=sys.stderr)
        return

    query_id = str(query_id)

    # Process gold standard (relevant documents)
    relevant_docs = question.get(relevant_field, [])
    if relevant_docs:
        if query_id not in gold_qrels:
            gold_qrels[query_id] = {}

        # Handle both single doc and list of docs
        if isinstance(relevant_docs, str):
            relevant_docs = [relevant_docs]

        for doc_id in relevant_docs:
            doc_id = str(doc_id)
            # Remove last two numbers from doc_id
            clean_doc_id = remove_last_two_numbers(doc_id)
            gold_qrels[query_id][clean_doc_id] = 1

    # Process system output (retrieved passages)
    retrieved_passages = entry.get('retrieved_passages', [])
    if retrieved_passages:
        if query_id not in system_qrels:
            system_qrels[query_id] = {}

        # Limit to top-k if specified
        passages_to_process = retrieved_passages[:top_k] if top_k else retrieved_passages

        for rank, passage in enumerate(passages_to_process, 1):
            doc_id = passage.get('id')
            if not doc_id:
                continue

            doc_id = str(doc_id)
            # Remove last two numbers from doc_id
            clean_doc_id = remove_last_two_numbers(doc_id)

            # Use rank as relevance score (higher rank = lower score)
            # Or use the passage score if available
            score = passage.get('score', 1.0 / rank)

            # For qrels, we typically use binary relevance (0 or 1)
            # But we can also use the rank or score. Keep the max doc entry.
            if clean_doc_id not in system_qrels[query_id] or score > system_qrels[query_id][clean_doc_id]:
                system_qrels[query_id][clean_doc_id] = score


def main():
    parser = argparse.ArgumentParser(
        description='Convert retrieval system output to TREC qrels format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage
  %(prog)s -i results.jsonl -g gold.qrels -s system.qrels

  # Only include top-100 results
  %(prog)s -i results.jsonl -g gold.qrels -s system.qrels -k 100

  # Custom field names
  %(prog)s -i results.json -g gold.qrels -s system.qrels \\
    --question-id-field id --relevant-field answers

TREC qrels format:
  query_id  iteration  doc_id  relevance

  Where:
    - query_id: Question/query identifier
    - iteration: Typically 0
    - doc_id: Document identifier (with last two numbers removed)
    - relevance: Binary (0/1) or graded relevance score
        '''
    )

    parser.add_argument('-i', '--input', type=Path, required=True,
                       help='Input JSON or JSONL file with retrieval results')
    parser.add_argument('-g', '--gold', type=Path, required=True,
                       help='Output file for gold standard qrels')
    parser.add_argument('-s', '--system', type=Path, required=True,
                       help='Output file for system output qrels')
    parser.add_argument('-k', '--top-k', type=int, default=None,
                       help='Only include top-k retrieved passages (default: all)')
    parser.add_argument('--iteration', type=int, default=0,
                       help='Iteration number for TREC format (default: 0)')
    parser.add_argument('--question-id-field', type=str, default='task_id',
                       help='Field name for question ID (default: task_id)')
    parser.add_argument('--relevant-field', type=str, default='relevant',
                       help='Field name for relevant documents (default: relevant)')

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Process the file
    process_retrieval_output(
        args.input,
        args.gold,
        args.system,
        args.top_k,
        args.iteration,
        args.question_id_field,
        args.relevant_field
    )


if __name__ == '__main__':
    main()
