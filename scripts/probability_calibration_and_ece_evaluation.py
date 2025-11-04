#!/usr/bin/env python3
"""
Probability Calibration Script

This script takes two JSON/JSONL files (dev and test) containing SearchResult data,
extracts probabilities and ground truth labels using EvaluationEngine,
calibrates the probabilities using TemperatureScaling on dev data,
applies calibration to test data, and computes ECE scores before and after.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from tqdm.auto import tqdm

from docuverse.engines.data_template import DataTemplate, read_doc_query_format
from docuverse.utils.ece_brier.calibration_visualizations import plot_calibration_comparison, \
    create_full_calibration_report
from docuverse.utils.ece_brier.ece import calibration_curve
# Import necessary components
from docuverse.utils.evaluator import EvaluationEngine
from docuverse.engines.search_engine_config_params import EvaluationArguments, DocUVerseConfig
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils.ece_brier.probability_calibration import TemperatureScaling, ECECalculator, ProbabilityCalibrator, \
    IsotonicCalibration
import matplotlib.pyplot as plt

def read_jsonl_file(file_path: str) -> List[dict]:
    """Read a JSONL file and return list of dictionaries."""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def read_json_file(file_path: str) -> Union[List[dict], dict]:
    """Read a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_data_file(file_path: str) -> List[dict]:
    """Load data from JSON or JSONL file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix == '.jsonl':
        return read_jsonl_file(str(file_path))
    elif file_path.suffix == '.json':
        data = read_json_file(str(file_path))
        # If it's a single object, wrap it in a list
        if isinstance(data, dict):
            return [data]
        return data
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .json or .jsonl")


def convert_to_search_results(data: List[dict], num_samples:int=-1,
                              query_template: DataTemplate=None) -> List[SearchResult]:
    """Convert raw data to SearchResult objects."""
    search_results = []

    # Sample data if num_samples is positive
    if 0 < num_samples < len(data):
        data = np.random.choice(data, size=num_samples, replace=False)

    query_id_header = query_template.id_header
    query_text_header = query_template.text_header

    for item in tqdm(data, desc="Converting data to search results"):
        # Create a mock query object if needed
        if 'question' in item:
            question = SearchQueries.Query(template=query_template, **item['question'])
        else:
            raise RuntimeError(f"Unsupported response type: {item['question']}")

        # Create SearchResult
        search_result = SearchResult(question, item.get('retrieved_passages', []))
        search_results.append(search_result)

    return search_results


def extract_queries_from_search_results(search_results: List[SearchResult]) -> List[SearchQueries.Query]:
    """Extract queries from SearchResult objects."""
    return [sr.question for sr in search_results]


def extract_probabilities_and_labels(evaluation_engine: EvaluationEngine, 
                                    queries: List[SearchQueries.Query], 
                                    search_results: List[SearchResult]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract probabilities and ground truth labels using EvaluationEngine.
    
    Returns:
        Tuple of (probabilities, labels) as numpy arrays
    """
    # Compute evaluation to extract score pairs
    eval_output = evaluation_engine.compute_score(queries, search_results)
    
    if not eval_output.score_pairs:
        raise ValueError("No score pairs found. Make sure the data contains relevant information.")
    
    # Extract probabilities and labels from score_pairs
    # score_pairs is a list of lists of [probability, label] pairs
    all_probs = []
    all_labels = []
    
    for query_scores in eval_output.score_pairs:
        for prob, label in query_scores:
            all_probs.append(prob)
            all_labels.append(label)
    
    return np.array(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description='Calibrate probabilities and compute ECE scores')
    parser.add_argument('dev_file', help='Development file (JSON/JSONL)')
    parser.add_argument('test_file', help='Test file (JSON/JSONL)')
    parser.add_argument('--eval_measure', default='ece', 
                       help='Evaluation measure to use (default: ece)')
    parser.add_argument('--ranks', default='1,3,5', 
                       help='Ranks for evaluation (default: 1,3,5)')
    parser.add_argument('--n_bins', type=int, default=15, 
                       help='Number of bins for ECE calculation (default: 15)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print detailed output')
    parser.add_argument("-N", "--num_dev_samples", type=int, default=-1,
                        help='Number of samples to use for evaluation (default: 1000)')
    parser.add_argument("--data_template", type=str, default=None,
                        help='Data template (needs to be specified)')
    parser.add_argument("--forced_1s", type=int, default=0,
                        help="If defined, a number of (1,1) pairs will be added to the dev data, "
                             "to force alignment for idempotence.")
    parser.add_argument("--forced_test_1s", type=int, default=0,
                        help="If defined, a number of (1,1) pairs will be added to the test data, "
                             "to force alignment for idempotence.")
    parser.add_argument("-o", "--output_prefix", type=str, default="calibration",
                        help="The prefix for the output.")

    args = parser.parse_args()

    data_template = args.data_template
    if data_template is None:
        print("Data template not specified. Please specify a template with --data_template.", file=sys.stderr)
        exit(1)
    elif not Path(data_template).exists():
        print(f"Data template {data_template} not found.", file=sys.stderr)
        exit(2)
    else:
        data_template, query_template = read_doc_query_format(data_template)

    print("=" * 60)
    print("PROBABILITY CALIBRATION AND ECE EVALUATION")
    print("=" * 60)
    
    try:
        # Load data files
        print(f"\n1. Loading data files...")
        print(f"   Dev file: {args.dev_file}")
        print(f"   Test file: {args.test_file}")
        
        dev_data = load_data_file(args.dev_file)
        test_data = load_data_file(args.test_file)
        
        print(f"   Loaded {len(dev_data)} dev samples and {len(test_data)} test samples")
        
        # Convert to SearchResult objects
        print(f"\n2. Converting to SearchResult objects...")
        dev_search_results = convert_to_search_results(dev_data, args.num_dev_samples, query_template=query_template)
        test_search_results = convert_to_search_results(test_data, query_template=query_template)
        
        # Extract queries
        dev_queries = extract_queries_from_search_results(dev_search_results)
        test_queries = extract_queries_from_search_results(test_search_results)
        
        # Create evaluation engine
        print(f"\n3. Setting up evaluation engine...")
        # eval_config = EvaluationArguments(
        #     eval_measure=args.eval_measure,
        #     ranks=args.ranks
        # )
        eval_config = DocUVerseConfig(
            config={
                'actions': 'e',
                'query_template': query_template,
                'eval_measure': args.eval_measure,
                'ranks': args.ranks
            })
        evaluation_engine = EvaluationEngine(eval_config)
        
        # Extract probabilities and labels
        print(f"\n4. Extracting probabilities and labels...")
        dev_probs, dev_labels = extract_probabilities_and_labels(
            evaluation_engine, dev_queries, dev_search_results
        )
        test_probs, test_labels = extract_probabilities_and_labels(
            evaluation_engine, test_queries, test_search_results
        )

        dev_probs, dev_labels = add_1s(args.forced_1s, dev_labels, dev_probs)
        test_probs, test_labels = add_1s(args.forced_test_1s, test_labels, test_probs)

        print(f"   Dev set: {len(dev_probs)} probability-label pairs")
        print(f"   Test set: {len(test_probs)} probability-label pairs")
        
        if args.verbose:
            print(f"   Dev probabilities range: [{dev_probs.min():.4f}, {dev_probs.max():.4f}]")
            print(f"   Test probabilities range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
            print(f"   Dev positive rate: {dev_labels.mean():.4f}")
            print(f"   Test positive rate: {test_labels.mean():.4f}")
        
        # Initialize ECE calculator and Temperature Scaling
        print(f"\n5. Initializing calibration components...")
        ece_calculator = ECECalculator(n_bins=args.n_bins)
        # # temperature_scaling = TemperatureScaling()
        # temperature_scaling = IsotonicCalibration()
        calibrator = ProbabilityCalibrator(n_bins=args.n_bins)
        
        # Compute original ECE scores
        print(f"\n6. Computing original ECE scores...")
        original_dev_ece = ece_calculator.calculate(dev_probs, dev_labels)
        original_test_ece = ece_calculator.calculate(test_probs, test_labels)
        
        print(f"   Original Dev ECE:  {original_dev_ece:.6f}")
        print(f"   Original Test ECE: {original_test_ece:.6f}")
        
        # Calibrate on dev set
        print(f"\n7. Training temperature scaling on dev set...")
        # temperature_scaling.fit(dev_probs, dev_labels)
        # print(f"   Learned temperature: {temperature_scaling.temperature:.4f}")
        calibration_results  = calibrator.calibrate(test_probs, test_labels, dev_probs, dev_labels)

        
        # Apply calibration
        print(f"\n8. Applying calibration...")
        # calibrated_dev_probs = temperature_scaling.transform(dev_probs)
        # calibrated_test_probs = temperature_scaling.transform(test_probs)
        
        # Compute calibrated ECE scores
        print(f"\n9. Computing calibrated ECE scores...")
        calibrated_dev_ece = calibration_results['dev']['ece']
        calibrated_test_ece = calibration_results['test']['ece']
        cv_scores = calibration_results['cv_scores']

        methods = list(calibrated_dev_ece.keys())

        for method in methods:
            print(f"\n{method}:")
            print(f"      Calibrated Dev ECE:  {calibrated_dev_ece[method]:.6f}")
            print(f"      Calibrated Test ECE: {calibrated_test_ece[method]:.6f}")
        
        # Print calibration_results summary
        print(f"\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        dev_improvement = {key: (original_dev_ece - calibrated_dev_ece[key]) for key in methods}
        test_improvement = {key: (original_test_ece - calibrated_test_ece[key]) for key in methods}
        
        dev_improvement_pct = {method: ((dev_improvement[method] / original_dev_ece * 100) if original_dev_ece > 0 else 0)
                               for method in methods}
        test_improvement_pct = {method:((test_improvement[method] / original_test_ece * 100) if original_test_ece > 0 else 0)
                                for method in methods}

        print("\nCROSS-VALIDATION SCORES:")
        print("=" * 60)
        for method in methods:
            scores = cv_scores[method]
            mean_score = scores['mean']
            std_score = scores['std'] # np.std(scores)
            print(f"\n{method}:")
            print(f"  Mean ECE:  {mean_score:.6f} (±{std_score:.6f})")
            print(f"  Scores:    {', '.join([f'{score:.6f}' for score in scores['scores']])}")

        best_method = min(calibrated_test_ece.items(), key=lambda x: x[1])[0]
        print(f"\nBEST PERFORMING METHOD: {best_method}")
        print(f"  Test ECE: {calibrated_test_ece[best_method]:.6f}")
        print(f"\nDEV SET:")
        print(f"  Original ECE:    {original_dev_ece:.6f}")
        for method in methods:
            print(f"  {method} Calibrated ECE:  {calibrated_dev_ece[method]:.6f}")
            print(f"           Improvement:     {dev_improvement[method]:.6f} ({dev_improvement_pct[method]:+.2f}%)")
        
        print(f"\nTEST SET:")
        print(f"  Original ECE:    {original_test_ece:.6f}")
        for method in methods:
            print(f"  {method} Calibrated ECE:  {calibrated_test_ece[method]:.6f}")
            print(f"           Improvement:     {test_improvement[method]:.6f} ({test_improvement_pct[method]:+.2f}%)")
        
        # print(f"\nTEMPERATURE SCALING PARAMETERS:")
        # print(f"  Temperature:     {temperature_scaling.temperature:.4f}")

        calibrated_test_probs = calibration_results['test']['probs']
        fig2 = plot_calibration_comparison(test_probs, test_labels, cal_probs=calibrated_test_probs, n_bins=args.n_bins)
        plt.savefig(f'{args.output_prefix}_comparison.png', dpi=300, bbox_inches='tight')
        print(f"   Saved to: {args.output_prefix}_comparison.png")

        print("\n4. Creating comprehensive report...")
        fig4 = create_full_calibration_report(test_probs, test_labels,
                                              calibration_data=calibration_results['test'],
                                              n_bins=args.n_bins,
                                              calibrator=calibrator,
                                              save_path=f'{args.output_prefix}_report.png')
        print(f"   Saved to: {args.output_prefix}_report.png")
        
        if args.verbose:
            print(f"\nSAMPLE CALIBRATED PROBABILITIES (first 10):")
            print(f"{'Original':<12} {'Calibrated':<12} {'True Label':<12}")
            print("-" * 36)
            best_method = calibrator.best_method
            for i in range(min(50, len(test_probs))):
                print(f"{test_probs[i]:<12.4f} {calibrated_test_probs[best_method][i]:<12.4f} {test_labels[i]:<12}")
        
        print("\n" + "=" * 60)
        
        # Return success
        return 0
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def add_1s(num_1s: int, labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if num_1s > 0:
        probs = np.concatenate([probs, np.ones(num_1s)])
        labels = np.concatenate([labels, np.ones(num_1s)])
        print(f"    Added {num_1s} 1.0 samples to dev data")
    return probs, labels


if __name__ == "__main__":
    sys.exit(main())
