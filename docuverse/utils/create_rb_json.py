import json
import argparse
from docuverse.utils.ticker import Ticker
from docuverse.utils.fof import FoF
import os

def makedir_for_file(file):
    _dir = os.path.dirname(file)
    if not os.path.exists(_dir):
        os.makedirs(_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse input and optional output file")
    parser.add_argument("file", type=str, help="Path to the input file")
    parser.add_argument("-o", "--output_dir", type=str, default=".", required=False, help="Path to the output file")
    args = parser.parse_args()
    tk = Ticker(message="Processing files: ")
    fof = FoF(args.file)
    for i, fl in enumerate(fof):
        fl = fl.strip()
        tk.tick(force=True, new_value=f"{i}/{len(fof)}: {fl}")
        toc_data = json.load(open(fl))
        data = json.load(open(fl.replace("_toc.json", ".json")))
        for toc, entry in zip(toc_data, data):
            for key in ['title', 'level', 'page']:
                entry[f"section_{key}"] = toc[key]
        output_file = os.path.join(args.output_dir, fof.basefile(fl).replace("_toc.json", ".jsonl"))
        makedir_for_file(output_file)
        # json.dump(data, open(output_file, "w"))
        with open(output_file, "w") as out:
            for d in data:
                out.write(json.dumps(d) + "\n")

    tk.clear()