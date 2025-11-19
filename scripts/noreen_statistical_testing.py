import json
import random
from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm

try:
    from simple_colors1 import get_color_func
    green = get_color_func('green')
    red = get_color_func('red')
except ImportError:
    def red(val, style="bold"):
        """
        Print text in red color with optional bold formatting.

        Args:
            val (str): The text to print
            bold (bool): Whether to make the text bold (default: False)
        """
        # ANSI escape codes
        bold = style=="bold"
        RED = '\033[91m'
        BOLD = '\033[1m' if bold else ''
        RESET = '\033[0m'

        return f"{BOLD}{RED}{val}{RESET}"


    def green(val, style="bold"):
        """
        Print text in green color with optional bold formatting.

        Args:
            val (str): The text to print
            bold (bool): Whether to make the text bold (default: False)
        """
        # ANSI escape codes
        GREEN = '\033[92m'
        BOLD = '\033[1m' if style=="bold" else ''
        RESET = '\033[0m'

        return f"{BOLD}{GREEN}{val}{RESET}"

def get_args():
    """arguments to use
    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
    )
    parser.add_argument("--model", type=str)
    parser.add_argument("--baseline", type=str, default="qr",
                        help="baseline to test")
    parser.add_argument("--result", type=str, default="core",
                        help="The name of the system to compare to")
    parser.add_argument("-n", type=int, default=1000, help="number of runs")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    print(args)

    baseline = args.baseline
    result = args.result

    baseline_directory = Path(f'{args.data_dir}/{args.model}/{baseline}')
    result_directory = Path(f'{args.data_dir}/{args.model}/{result}')
    baseline_suffix = f"_{baseline}_docs_ndcg.jsonl"
    result_suffix = f"_{result}_docs_ndcg.jsonl"

    print(f"Baseline: {baseline_suffix}, result: {result_suffix}")
    tasks = [f.name[:-len(baseline_suffix)] for f in baseline_directory.glob("*ndcg.jsonl")]

    # Load all data first and compute observed statistics
    all_task_data = {}
    CQA_tasks = []

    for task in tasks:
        with open(f'{baseline_directory}/{task}{baseline_suffix}', 'r') as f:
            ndcg_baseline = [json.loads(line) for line in f][0]
        with open(f'{result_directory}/{task}{result_suffix}', 'r') as f:
            ndcg_result = [json.loads(line) for line in f][0]

        # Extract query-level scores as parallel lists
        baseline_scores = []
        result_scores = []
        for baseline_dict, result_dict in zip(ndcg_baseline, ndcg_result):
            for k in baseline_dict.keys():
                assert k in result_dict.keys()
                baseline_scores.append(baseline_dict[k])
                result_scores.append(result_dict[k])

        n_queries = len(baseline_scores)
        baseline_mean = sum(baseline_scores) / n_queries
        result_mean = sum(result_scores) / n_queries

        all_task_data[task] = {
            'baseline': baseline_scores,
            'result': result_scores,
            'n_queries': n_queries,
            'baseline_mean': baseline_mean,
            'result_mean': result_mean,
            'observed_diff': result_mean - baseline_mean
        }

        if task.startswith("cqadupstack"):
            CQA_tasks.append(task)

    # Compute observed statistics for CQA aggregate (macro-average across CQA tasks)
    cqa_task_baseline_means = [all_task_data[task]['baseline_mean'] for task in CQA_tasks]
    cqa_task_result_means = [all_task_data[task]['result_mean'] for task in CQA_tasks]
    cqa_observed_baseline = sum(cqa_task_baseline_means) / len(cqa_task_baseline_means)
    cqa_observed_result = sum(cqa_task_result_means) / len(cqa_task_result_means)
    cqa_observed_diff = cqa_observed_result - cqa_observed_baseline

    # Compute observed statistics for BEIR15 aggregate (macro-average: CQA as 1 task + non-CQA tasks)
    non_cqa_tasks = [task for task in tasks if not task.startswith("cqadupstack")]
    n_beir15_tasks = len(non_cqa_tasks) + 1  # non-CQA tasks + 1 for CQA aggregate

    beir15_observed_baseline = (cqa_observed_baseline + sum(all_task_data[task]['baseline_mean'] for task in non_cqa_tasks)) / n_beir15_tasks
    beir15_observed_result = (cqa_observed_result + sum(all_task_data[task]['result_mean'] for task in non_cqa_tasks)) / n_beir15_tasks
    beir15_observed_diff = beir15_observed_result - beir15_observed_baseline

    # Run permutation test
    n_experiment = args.n
    permuted_diffs = {task: [] for task in tasks}
    permuted_diffs['cqadupstack'] = []
    permuted_diffs['BEIR15'] = []

    for n in tqdm(range(n_experiment), desc="Running simulations:"):
        # Permute each task and collect task-level means
        task_permuted_means = {}

        for task in tasks:
            baseline_scores = all_task_data[task]['baseline']
            result_scores = all_task_data[task]['result']
            n_queries = all_task_data[task]['n_queries']

            # Stratified shuffling: for each query, randomly swap baseline and result with 50% probability
            permuted_baseline = []
            permuted_result = []
            for baseline_score, result_score in zip(baseline_scores, result_scores):
                if random.choice([True, False]):
                    # Swap
                    permuted_baseline.append(result_score)
                    permuted_result.append(baseline_score)
                else:
                    # Keep original
                    permuted_baseline.append(baseline_score)
                    permuted_result.append(result_score)

            # Compute permuted means for this task
            permuted_mean_baseline = sum(permuted_baseline) / n_queries
            permuted_mean_result = sum(permuted_result) / n_queries
            permuted_diff = permuted_mean_result - permuted_mean_baseline

            permuted_diffs[task].append(permuted_diff)
            task_permuted_means[task] = (permuted_mean_baseline, permuted_mean_result)

        # Compute CQA aggregate permuted difference (macro-average across CQA tasks)
        n_cqa_tasks = len(CQA_tasks)
        cqa_permuted_baseline = sum(task_permuted_means[task][0] for task in CQA_tasks) / n_cqa_tasks
        cqa_permuted_result = sum(task_permuted_means[task][1] for task in CQA_tasks) / n_cqa_tasks
        cqa_permuted_diff = cqa_permuted_result - cqa_permuted_baseline
        permuted_diffs['cqadupstack'].append(cqa_permuted_diff)

        # Compute BEIR15 aggregate: macro-average across all tasks (CQA as one task + non-CQA tasks)
        beir15_permuted_baseline = (cqa_permuted_baseline + sum(task_permuted_means[task][0] for task in non_cqa_tasks)) / n_beir15_tasks
        beir15_permuted_result = (cqa_permuted_result + sum(task_permuted_means[task][1] for task in non_cqa_tasks)) / n_beir15_tasks
        beir15_permuted_diff = beir15_permuted_result - beir15_permuted_baseline
        permuted_diffs['BEIR15'].append(beir15_permuted_diff)

    # Calculate p-values using correct formula: (count_extreme + 1) / (n_permutations + 1)
    print("\n" + "="*70)
    print("Per-task p-values:")
    print("="*70)
    def print_val(val, diff):
        if val < 0.05:
            if diff < 0:
                return red(f"{val:.4f}", "bold")
            else:
                return green(f"{val:.4f}", "bold")
        else:
            return f"{val:.4f}"

    for task in tasks:
        observed_diff = all_task_data[task]['observed_diff']
        count_extreme = sum(1 for d in permuted_diffs[task] if abs(d) >= abs(observed_diff))
        p_value = (count_extreme + 1) / (n_experiment + 1)
        print(f"{task:40s} | Observed diff: {observed_diff:+.6f} | p-value: {print_val(p_value, observed_diff)}")

    print("\n" + "="*70)
    print("Aggregate p-values:")
    print("="*70)
    count_extreme_cqa = sum(1 for d in permuted_diffs['cqadupstack'] if abs(d) >= abs(cqa_observed_diff))
    p_value_cqa = (count_extreme_cqa + 1) / (n_experiment + 1)
    print(f"{'cqadupstack (CQA macro-average)':40s} | Observed diff: {cqa_observed_diff:+.6f} | p-value: {print_val(p_value_cqa,cqa_observed_diff)}")

    count_extreme_beir = sum(1 for d in permuted_diffs['BEIR15'] if abs(d) >= abs(beir15_observed_diff))
    p_value_beir = (count_extreme_beir + 1) / (n_experiment + 1)
    print(f"{'BEIR15 (macro-average)':40s} | Observed diff: {beir15_observed_diff:+.6f} | p-value: {print_val(p_value_beir,beir15_observed_diff)}")
            
if __name__ == "__main__":
    main()