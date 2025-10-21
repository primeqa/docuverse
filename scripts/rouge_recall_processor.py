import json
import argparse
import sys
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from tqdm.auto import tqdm


def compute_rouge_recall(reference: str, candidate: str) -> float:
    """
    Compute ROUGE-L recall score between reference and candidate texts.
    
    Args:
        reference: The reference text (query)
        candidate: The candidate text (retrieved passage)
        
    Returns:
        ROUGE-L recall score (0.0 to 1.0)
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Error: rouge_score package is required. Install with: pip install rouge-score", file=sys.stderr)
        sys.exit(1)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    
    # Return ROUGE-L recall score
    return scores['rougeL'].recall

def extract_text_from_path(data: Dict[str, Any], path: str) -> str:
    """
    Extract text from nested dictionary using dot-notation path.
    
    Args:
        data: Dictionary to search
        path: Dot-separated path (e.g., "question.text")
        
    Returns:
        Extracted text or empty string if not found
    """
    keys = path.split('.')
    current = data
    
    try:
        for key in keys:
            if isinstance(current, list) and key.isdigit():
                current = current[int(key)]
            elif isinstance(current, dict):
                current = current[key]
            else:
                return ""
        return str(current) if current is not None else ""
    except (KeyError, IndexError, TypeError):
        return ""

def process_single_query(args: Tuple[int, Dict[str, Any], float]) -> Dict[str, Any]:
    """
    Process a single query result and compute ROUGE recall.

    Args:
        args: Tuple containing (qid, query_result, rouge_threshold)

    Returns:
        Dictionary with processing results
    """
    qid, query_result, rouge_threshold = args

    result = {
        "qid": qid,
        "passage_id": None,
        "rouge_recall": None,
        "above_threshold": False,
        "error": None
    }

    # Extract query text
    query_text = extract_text_from_path(query_result, "question.text")
    if not query_text:
        # Try alternative path
        query_text = extract_text_from_path(query_result, "question/text")

    if not query_text:
        result["error"] = f"Could not find query text in result {qid}"
        return result

    # Extract first retrieved passage
    first_passage_text = extract_text_from_path(query_result, "retrieved_passages.0.text")
    first_passage_id = extract_text_from_path(query_result, "retrieved_passages.0.id")
    gold_id = extract_text_from_path(query_result, "question.relevant.0")

    if not first_passage_text:
        result["error"] = f"Could not find first retrieved passage in result {qid}"
        return result

    # Compute ROUGE recall
    rouge_recall = compute_rouge_recall(query_text, first_passage_text)

    result["passage_id"] = first_passage_id
    result["gold_id"] = gold_id
    result["rouge_recall"] = rouge_recall
    result["above_threshold"] = rouge_recall > rouge_threshold

    return result

def process_query_results(input_file: str, output_file: str = None, rouge_threshold: float = 0.9, num_workers: int = None) -> None:
    """
    Process JSON file with query results and output JSONL format with ROUGE recall scores.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSONL file (stdout if None)
        rouge_threshold: ROUGE recall threshold (default: 0.9)
        num_workers: Number of parallel workers (default: CPU count)
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Handle both single query result and list of query results
    if not isinstance(data, list):
        data = [data]

    if num_workers is None:
        num_workers = min(cpu_count(), len(data))

    # Prepare arguments for parallel processing
    worker_args = [(qid, query_result, rouge_threshold) for qid, query_result in enumerate(data)]

    high_rouge_count = 0
    total_queries = len(data)
    results = []

    # Open output file if specified
    output_handle = open(output_file, 'w', encoding='utf-8') if output_file else sys.stdout

    try:
        # Process in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_qid = {executor.submit(process_single_query, args): args[0] for args in worker_args}

            # Process completed futures with progress bar
            for future in tqdm(as_completed(future_to_qid), total=len(worker_args), desc="Processing queries"):
                try:
                    result = future.result()
                    results.append(result)

                    if result["error"]:
                        print(f"Warning: {result['error']}", file=sys.stderr)
                    elif result["above_threshold"]:
                        high_rouge_count += 1

                except Exception as e:
                    qid = future_to_qid[future]
                    print(f"Error processing query {qid}: {e}", file=sys.stderr)

        # Sort results by qid to maintain order
        results.sort(key=lambda x: x["qid"])

        # Write JSONL output
        for result in results:
            if result["rouge_recall"] is not None:  # Only output valid results
                output_line = {
                    "qid": result["qid"],
                    "passage_id": result["passage_id"],
                    "gold_id": result["gold_id"],
                    "rouge_recall": result["rouge_recall"],
                    "above_threshold": result["above_threshold"]
                }
                print(json.dumps(output_line), file=output_handle)

    finally:
        if output_file and output_handle != sys.stdout:
            output_handle.close()

    # Print summary to stderr
    valid_results = sum(1 for r in results if r["rouge_recall"] is not None)
    print(f"Processed {total_queries} queries", file=sys.stderr)
    print(f"Valid results: {valid_results}", file=sys.stderr)
    print(f"Found {high_rouge_count} results with ROUGE recall > {rouge_threshold:.1%}", file=sys.stderr)
    print(f"Success rate: {high_rouge_count/valid_results:.1%}" if valid_results > 0 else "No valid results", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description='Compute ROUGE scores between queries and first retrieved results, '
                   'outputting results in JSONL format'
    )
    parser.add_argument('input_file', help='Input JSON file with query results')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output JSONL file (default: stdout)')
    parser.add_argument('--threshold', '-t', type=float, default=0.9,
                       help='ROUGE recall threshold (default: 0.9)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')

    args = parser.parse_args()

    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: Threshold must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    # Validate workers
    if args.workers is not None and args.workers <= 0:
        print("Error: Number of workers must be positive", file=sys.stderr)
        sys.exit(1)

    process_query_results(args.input_file, args.output, args.threshold, args.workers)

if __name__ == "__main__":
    main()
