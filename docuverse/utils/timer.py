# *****************************************************************#
# (C) Copyright IBM Corporation 2019.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#
'''Provide timers for quick profiling.
Author: Radu Florian
raduf@us.ibm.com )'''
from datetime import datetime
import json
import os
import sys
import numpy as np
from typing import Dict, List, Optional

msec = 1
sec = 1000
minute = 60 * sec
hour = 60 * minute
day = 24 * hour
week = 7 * day
vals = [week, day, hour, minute, sec, msec]
names = ['week', 'day', 'hour', 'minute', 'second', 'millisecond']


class timer(object):
    '''Class to quickly compute timing, in a hierarchical fashion. To use, create an instance with
    a name, then use the tm.add_timing("name"). The variable "name" can contain namespaces
     (i.e. separate by using '::').
     The timer gather all the timers in a common space, and one can display the overall timing,
     by using timer.display_timing(), outputting something like this:
    Name                                      Time (s)      Rel %      Abs %  Spd(kw/s)  Spd(kc/s)
    MentionDetector                               27.9s      100.0%       99.3%       3.8      18.9
      decode                                      11.4s       40.9%       40.7%       9.2      46.1
        post_processing                            0.7s        5.9%        2.4%     157.9     787.2
          voting                                   0.1s        0.3%        0.4%    1054.9    5258.4
        read_examples                              2.3s       19.7%        8.0%      46.8     233.4
          Creating_pytorch_structures              0.1s        0.8%        0.4%    1054.9    5258.4
          convert_examples_to_features             2.2s       98.0%        7.9%      47.8     238.3
          create_examples_from_iob                 0.1s        0.8%        0.4%    1054.9    5258.4
        bert_run                                   8.5s       74.4%       30.3%      12.4      61.9
        data_prep                                  0.1s        0.0%        0.4%    1054.9    5258.4
      init                                        16.4s       59.0%       58.6%       6.4      32.0
        preamble                                   0.1s        0.0%        0.4%    1054.9    5258.4
        read_iob                                   0.2s        1.0%        0.6%     616.9    3075.1
      post_decoding                                0.1s        0.1%        0.4%    1054.9    5258.4
        write_output                               0.1s      100.0%        0.4%    1054.9    5258.4
    '''
    timing = {}
    timing_samples = {}
    timing_count_keys = {}  # Maps full timing key -> count key name (e.g., "queries", "docs")
    processed = 0

    # Available statistics for display_statistics()
    AVAILABLE_STATS = {
        "count": "Count",
        "sum": "Sum(ms)",
        "mean": "Mean(ms)",
        "median": "Med(ms)",
        "std": "Std(ms)",
        "min": "Min(ms)",
        "max": "Max(ms)",
        "p5": "P5(ms)",
        "p25": "P25(ms)",
        "p75": "P75(ms)",
        "p95": "P95(ms)",
        "p99": "P99(ms)",
    }
    DEFAULT_STATS = ["count", "sum", "mean", "median", "std", "p95", "p99", "max"]

    def __init__(self, name="timer", disable=False):
        self._orig = datetime.now()
        self._last = datetime.now()
        self._time = datetime.now()
        self.name = name
        self.disable = disable

    def mark(self):
        '''Starts a particular timing event.'''
        self._last = self._time
        self._time = datetime.now()

    def mark_and_return_time(self):
        '''Computes the humanly readable time from the previous mark() and returns it.'''
        self.mark()
        return self.time_since_last_mark()

    def mark_and_return_milliseconds(self):
        '''Marks and computes the milliseconds since the last mark()
        Returns:
            the milliseconds since the last mark()'''
        self.mark()
        return self.milliseconds_since_last_mark()

    def mark_and_return_microseconds(self):
        '''Marks and computes the microseconds since the last mark()
        Returns:
            The microseconds since the last mark()'''
        self.mark()
        return self.microseconds_since_last_mark()

    def time_since_last_mark(self):
        '''Computes the time since the last mark (but does not mark()).
        Returns:
            The humanly readable time since the last mark()
        '''
        # return self._time_between(self._time, self._last)
        return self.time_from_milliseconds(self.milliseconds_since_last_mark())

    def time_since_beginning(self):
        '''Returns the humanly interpretable time since the timer was created.
        Returns:
            The humanly readable time since the timer was created.
        '''
        self.mark()
        return self.time_from_milliseconds(self.milliseconds_since_beginning(), 0)

    def milliseconds_since_last_mark(self):
        '''Returns the milliseconds since the last mark().
        Returns:
            Milliseconds since the last mark.
        '''
        return self._milliseconds_between(self._time, self._last)

    def microseconds_since_last_mark(self):
        ''' Computes the microseconds since the last mark.
        Returns:
            microseconds since last mark.
        '''
        return timer._microseconds_between(self._time, self._last)

    def milliseconds_since_beginning(self):
        ''' Computes milliseconds since the beginning.
        Returns:
            milliseconds since the beginning.'''
        self.mark()
        return self._milliseconds_between(self._time, self._orig)

    @staticmethod
    def _microseconds_between(t1, t2):
        '''Internal function to compute microseconds between two times.
        Returns:
            the microseconds between two times.
        '''
        delta = t1 - t2
        return 1000000 * delta.seconds + delta.microseconds

    @staticmethod
    def _milliseconds_between(t1, t2):
        '''Computes and returns the number of milliseconds between two times.
        Returns:
            Returns the number of miliseconds between two times.'''
        delta = t1 - t2
        return int(1000.0 * delta.seconds + delta.microseconds / 1000.0)

    @staticmethod
    def _time_between(t1, t2):
        '''Compute the humnly interpretable time between two times.
        Returns:
            The humanly interpretable time between two times.'''
        # return str(t1 - t2)
        return timer.time_from_milliseconds(t1-t2)

    @staticmethod
    def time_from_milliseconds(millis, micros=0):
        '''Compute humanly readable time from milliseconds and microseconds.
        Arguments:
            millis - the time amount in milliseconds
            micros - the time amount in microseconds (if needed)
        Return:
            a string representing the humanly readable time represented by the given milliseconds
            and microseconds
        '''
        res = ""
        for n, v in zip(names, vals):
            val = int(millis / v)
            if val > 0:
                if res != "":
                    res += ", "
                res += "{} {}".format(val, n + "s" if val > 1 else n)
            millis -= v * val
        if res == "":
            res = "{} microsecond(s)".format(micros)
        return res

    def add_timing(self, key, ms=-1):
        '''Marks and adds the time to the specified label. The label can have "::" substring -
        allowing for hierarchical time reporting.
        Arguments:
            - key - the label to which to add timing
            - ms  - the millisecods to add. If not provided, the current timer is mark()ed, and
                    the timing from the last mark is used.
        '''
        if self.disable:
            return
        if ms == -1:
            ms = self.mark_and_return_microseconds()/1000.0
        timer.static_add_timing(self.name, key, ms)


    @staticmethod
    def static_add_timing(name, key, val):
        '''Adds the given number of milliseconds given in val to the given key.
        Also stores the individual sample for statistical analysis.
        Arguments:
            - key - the label to add the timing to
            - val - the number of milliseconds to add
        '''
        key = name + "::" + key
        if key not in timer.timing:
            timer.timing[key] = val
            timer.timing_samples[key] = [val]
        else:
            timer.timing[key] += val
            timer.timing_samples[key].append(val)

    @staticmethod
    def get_top_method(default="timer"):
        keys = list(timer.timing.keys())
        if len(keys)==0:
            return default
        else:
            return sorted(list(timer.timing.keys()))[0]

    @staticmethod
    def subtimer_from_top(name: str, default_parent:str="timer"):
        return timer(f"{timer.get_top_method(default_parent)}::{name}", disable=False)

    @staticmethod
    def _compute_timing_tree(namelist=None, level=0):
        '''Internal function to compute the namespace timing tree.
        Arguments:
            - namelist - the list of lists to display at this level of the tree (the list is
                         full from the root to leaf).
            - level    - the level of the current display_timing
        '''
        if namelist is None:
            namelist = [t.split("::") for t in timer.timing.keys()]

        if len(namelist) == 1:
            full_key = "::".join(namelist[0])
            return [{"key":"::".join(namelist[0][level:]), "children":[],
                     "time":timer.timing[full_key], "percent": 100,
                     "full_key": full_key}]
        else:
            _keys = sorted(namelist, key=lambda val: val[level])
            children = []
            dlist = {}
            leaves = {}
            prefixes = {}
            for key in _keys:
                K = key[level]
                if len(key) > level+1:
                    prefixes["::".join(key[:level+1])] = 1
                    if K not in dlist:
                        dlist[K] = [key]
                    else:
                        dlist[K] += [key]
                else:
                    leaves[K] = key


            res = []

            for K, L in dlist.items():
                children = timer._compute_timing_tree(L, level + 1)
                level_key = "::".join(L[0][0:level+1])
                time = sum([v['time'] for v in children]) \
                    if level_key not in timer.timing else timer.timing[level_key]
                time = max(time, 0.01)
                for child in children:
                    child['percent'] = child['time']/time*100
                res.append({"key": K, "children": children, "time": time, "percent": 100,
                            "full_key": level_key})

            for K, L in leaves.items():
                level_key = "::".join(L[0:level+1])
                if "::".join(L) not in prefixes:
                    children = timer._compute_timing_tree([L], level)
                    res.append(children[0])

        return res

    @staticmethod
    def display_timing(totalms, level=0, stat_list=None, keys: dict[str, int]|None=None,
                       key_associations: dict[str, str]|None=None,
                       sorted_by:str|None=None,
                       reverse=False, output_stream=None):
        '''Static method that will print the hierarchical times for the labeled times.
        The speeds computed are in kilo-words/sec and kilo-chars/sec.
        Arguments:
            - totalms - the total number of milliseconds - used to compute the absolute % times.
            - level   - the level of the tree to display (default 0)
            - stat_list - filter for the labels
            - keys: dict[str, int] - a list of keys to break performance by (e.g., num_words, num_chars, etc.)
            - key_associations: dict[str, str] - maps timing node names to count key names.
                    Looked up by full_key first, then by node display key.
                    Children inherit from their parent. Unassociated nodes show '-'.
                    Example: {"ingest": "docs", "search": "queries"}
                    If None, all nodes show all speed columns (backward-compatible).
            '''
        if keys is None:
            keys = {}
        if output_stream is None:
            output_stream = sys.stdout

        def _resolve_count_key(node, parent_count_key):
            '''Resolve the count key for a node: check key_associations by full_key,
            then by display key, then inherit from parent.'''
            if key_associations is None:
                return None  # No associations → show all columns
            full_key = node.get('full_key', '')
            # Check full_key, then each suffix (right-to-left) for flexible matching
            if full_key in key_associations:
                return key_associations[full_key]
            if node['key'] in key_associations:
                return key_associations[node['key']]
            return parent_count_key

        def _display_tree(node, level, keys: dict[str, int], parent_count_key=None):
            def process_val(v):
                vv = 1000*v*1.0/ms
                if vv < 10000:
                    return f"{vv:10.1f}"
                elif vv < 1000000:
                    return f"{vv/1000.0:10.1f}k"
                elif v < 10000000000:
                    return f"{vv/1000000.0:10.1f}M"
                else:
                    return f"{vv/100000000.0:10.1f}Bs"
            ms = max(100, node["time"])
            secs = max(0.1, ms/1000.0)
            perc = node["percent"]
            okey = (" " * 2 * level) + node['key']

            node_count_key = _resolve_count_key(node, parent_count_key)

            print("{:<40s}{:>10.1f}s {:>10.1f}% {:10.1f}%".
                  format(okey, secs, perc, ms*100 / totalms),
                  file=output_stream, end='')
            for k, v in keys.items():
                if key_associations is None or node_count_key == k:
                    print(process_val(v), end='', file=output_stream)
                else:
                    print(f"{'-':>10s}", end='', file=output_stream)
            print(file=output_stream)
            if sorted_by == "%":
                node["children"] = sorted(node["children"], key=lambda val: val["percent"], reverse=reverse)
            elif sorted_by == "name":
                node["children"] = sorted(node["children"], key=lambda val: val["key"], reverse=reverse)

            for child in node["children"]:
                _display_tree(child, level+1, keys, node_count_key)

        tree = timer._compute_timing_tree(None, 0)

        if len(tree) > 1:
            tree = {"key": "Root", 'children': tree, 'percent': 100, 'time': totalms,
                    'full_key': ''}
            for child in tree['children']:
                child['percent'] = child['time']*100/totalms
        else:
            tree = tree[0]

        print(f"{'Name':<40s} {'Time (s)':>10s}  {'Rel %':>10s}  {'Abs %':>10s}", file=output_stream, end='')
        for k, v in keys.items():
            print(f"{k+'/s':>10s}", file=output_stream, end='')
        print(file=output_stream)

        _display_tree(tree, level, keys)

    @staticmethod
    def _compute_stat(samples_array, stat_name):
        '''Compute a single statistic from a numpy array of samples.'''
        if stat_name == "count":
            return len(samples_array)
        elif stat_name == "sum":
            return float(np.sum(samples_array))
        elif stat_name == "mean":
            return float(np.mean(samples_array))
        elif stat_name == "median":
            return float(np.median(samples_array))
        elif stat_name == "std":
            return float(np.std(samples_array))
        elif stat_name == "min":
            return float(np.min(samples_array))
        elif stat_name == "max":
            return float(np.max(samples_array))
        elif stat_name.startswith("p"):
            percentile = int(stat_name[1:])
            return float(np.percentile(samples_array, percentile))
        else:
            raise ValueError(f"Unknown statistic: {stat_name}")

    @staticmethod
    def get_statistics(stats: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        '''Compute statistics for all timing keys from individual samples.

        Arguments:
            stats: list of statistic names to compute. If None, uses DEFAULT_STATS.
                   Available: "count", "sum", "mean", "median", "std", "min", "max",
                              "p5", "p25", "p75", "p95", "p99"

        Returns:
            Dict mapping timing key -> dict of {stat_name: value}
        '''
        if stats is None:
            stats = timer.DEFAULT_STATS

        result = {}
        for key, samples in timer.timing_samples.items():
            arr = np.array(samples)
            result[key] = {s: timer._compute_stat(arr, s) for s in stats}
        return result

    @staticmethod
    def display_statistics(stats: Optional[List[str]] = None,
                           output_stream=None,
                           name_width: int = 50,
                           col_width: int = 10,
                           strip_common_prefix: bool = True):
        '''Display a flat table of timing statistics computed from individual samples.

        Arguments:
            stats: list of statistic names to display as columns. If None, uses DEFAULT_STATS.
                   Available: "count", "sum", "mean", "median", "std", "min", "max",
                              "p5", "p25", "p75", "p95", "p99"
            output_stream: where to print (default sys.stdout)
            name_width: width of the key name column
            col_width: width of each statistic column
            strip_common_prefix: if True, strip the common prefix from all keys
        '''
        if output_stream is None:
            output_stream = sys.stdout
        if stats is None:
            stats = timer.DEFAULT_STATS

        all_stats = timer.get_statistics(stats)
        if not all_stats:
            print("No timing data collected.", file=output_stream)
            return

        keys = sorted(all_stats.keys())

        # Strip common prefix if requested
        display_keys = keys
        prefix = ""
        if strip_common_prefix and len(keys) > 1:
            parts = [k.split("::") for k in keys]
            common = []
            for level_parts in zip(*parts):
                if len(set(level_parts)) == 1:
                    common.append(level_parts[0])
                else:
                    break
            if common:
                prefix = "::".join(common) + "::"
                display_keys = [k[len(prefix):] if k.startswith(prefix) else k for k in keys]

        # Build header
        headers = [timer.AVAILABLE_STATS.get(s, s) for s in stats]
        header_line = f"{'Key':<{name_width}s}" + "".join(f"{h:>{col_width}s}" for h in headers)
        print(header_line, file=output_stream)
        print("-" * len(header_line), file=output_stream)

        # Build rows
        def _fmt(val, stat_name):
            if stat_name == "count":
                return f"{int(val):>{col_width}d}"
            elif abs(val) < 0.01:
                return f"{val:>{col_width}.4f}"
            elif abs(val) < 10:
                return f"{val:>{col_width}.3f}"
            elif abs(val) < 1000:
                return f"{val:>{col_width}.1f}"
            elif abs(val) < 1000000:
                return f"{val/1000:>{col_width-1}.1f}k"
            else:
                return f"{val/1000000:>{col_width-1}.1f}M"

        for key, display_key in zip(keys, display_keys):
            row_stats = all_stats[key]
            row = f"{display_key:<{name_width}s}"
            row += "".join(_fmt(row_stats[s], s) for s in stats)
            print(row, file=output_stream)

    @staticmethod
    def save_statistics(path: str,
                        stats: Optional[List[str]] = None,
                        save_samples: bool = False):
        '''Save timing statistics to disk.

        The format is determined by the file extension:
          - .json  — JSON with computed statistics (and optionally raw samples)
          - .csv   — CSV table (one row per timing key, one column per statistic)
          - .tsv   — TSV table (same layout as CSV, tab-separated)

        Arguments:
            path: output file path (.json, .csv, or .tsv)
            stats: which statistics to include. If None, uses DEFAULT_STATS.
            save_samples: if True and format is JSON, also store the raw sample
                          lists under a "samples" key.
        '''
        if stats is None:
            stats = timer.DEFAULT_STATS

        all_stats = timer.get_statistics(stats)
        ext = os.path.splitext(path)[1].lower()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if ext == ".json":
            out = {"statistics": {}}
            for key in sorted(all_stats):
                entry = all_stats[key]
                # Convert numpy/int types to plain Python for JSON serialisation
                entry = {s: (int(v) if s == "count" else round(v, 6))
                         for s, v in entry.items()}
                out["statistics"][key] = entry
            if save_samples:
                out["samples"] = {k: [round(v, 6) for v in vs]
                                  for k, vs in sorted(timer.timing_samples.items())}
            with open(path, "w") as f:
                json.dump(out, f, indent=2)

        elif ext in (".csv", ".tsv"):
            sep = "," if ext == ".csv" else "\t"
            lines = [sep.join(["key"] + stats)]
            for key in sorted(all_stats):
                vals = all_stats[key]
                row = [key] + [
                    str(int(vals[s])) if s == "count" else f"{vals[s]:.6f}"
                    for s in stats
                ]
                lines.append(sep.join(row))
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")

        else:
            raise ValueError(
                f"Unsupported extension '{ext}'. Use .json, .csv, or .tsv"
            )

    @staticmethod
    def reset():
        '''Reset all timing data.'''
        timer.timing = {}
        timer.timing_samples = {}
        timer.processed = 0

    @staticmethod
    def test():
        timer.reset()
        tm = timer("Something")
        tm.add_timing("method1", 1000)
        tm.add_timing("method1::submethod1", 500)
        tm.add_timing("method1::submethod2", 300)
        tm.add_timing("method2::submethod1", 600)
        tm.add_timing("method2::submethod2", 600)
        tm.add_timing("method3::submethod1::subsubmethod1", 100)
        tm.add_timing("method3::submethod1::subsubmethod2", 100)
        tm.add_timing("method4", 2000)

        tm = timer("Something_2")
        tm.add_timing("method3::sumethod1::subsubmethod1", 100)
        tm.add_timing("method4::submethod1::subsubmethod1", 100)
        tm.add_timing("method4::submethod1::subsubmethod2", 100)
        tm.add_timing("method4::submethod1", 250)

        print("=== Without key_associations (backward-compatible): ===")
        tm.display_timing(9000, keys={'pigs':100, 'cats':10000})

        # Test with key_associations
        print("\n=== With key_associations: ===")
        timer.reset()
        tm = timer("ingest_and_test")
        tm.add_timing("ingest", 5000)
        tm.add_timing("ingest::encode", 4000)
        tm.add_timing("ingest::encode::tokenize", 1000)
        tm.add_timing("ingest::encode::model_forward", 3000)
        tm.add_timing("ingest::data_insertion", 1000)
        tm.add_timing("search", 3000)
        tm.add_timing("search::encode", 500)
        tm.add_timing("search::encode::tokenize", 100)
        tm.add_timing("search::encode::model_forward", 400)
        tm.add_timing("search::encode::tokenize", 120)
        tm.add_timing("search::encode::model_forward", 450)
        tm.add_timing("search::milvus_search", 2500)

        timer.display_timing(8000,
                             keys={'docs': 5000, 'queries': 200},
                             key_associations={'ingest': 'docs', 'search': 'queries'},
                             stat_list="sum,mean")

        # Test display_statistics with multiple samples
        print("\n--- Statistics Demo ---")
        timer.reset()
        import random
        random.seed(42)
        tm = timer("ingest_and_test::search::retrieve")
        for _ in range(100):
            tm.add_timing("encode::tokenize", random.gauss(5, 1))
            tm.add_timing("encode::model_forward", random.gauss(20, 5))
            tm.add_timing("milvus_search", random.gauss(3, 0.5))

        # Default statistics
        print("\nDefault stats:")
        timer.display_statistics()

        # Custom statistics selection
        print("\nCustom stats (just count, mean, p95):")
        timer.display_statistics(stats=["count", "mean", "p95"])

        # Save to disk
        import tempfile
        tmpdir = tempfile.mkdtemp()
        for ext in (".json", ".csv", ".tsv"):
            p = os.path.join(tmpdir, f"timing{ext}")
            timer.save_statistics(p)
            print(f"\nSaved {ext}: {p}")
            with open(p) as f:
                print(f.read()[:300])
        # JSON with raw samples
        p = os.path.join(tmpdir, "timing_samples.json")
        timer.save_statistics(p, stats=["count", "mean", "p95"], save_samples=True)
        print(f"\nSaved JSON with samples: {p}")
        with open(p) as f:
            content = json.load(f)
            print(f"  keys: {list(content.keys())}")
            first = list(content['samples'].keys())[0]
            print(f"  samples['{first}']: {len(content['samples'][first])} values")


if __name__ == "__main__":
    timer.test()
