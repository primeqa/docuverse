import argparse

import numpy as np
import pandas as pd
import json
from tqdm import tqdm

field_map = {
    "is_essuperuser_ibmentitlement": "",
    'scopes': 'metadata.scopes',
    'doc_vector': '',
    'field_keyword_01': 'metadata.field_keyword_01',
    'ibmdocstype': 'metadata.ibmdocstype',
    'url': 'metadata.url',
    'doc_id': '_id',
    'is_public_ibmentitlement': 'metadata.is_public_ibmentitlement',
    'sub_scopes': 'metadata.sub_scopes',
    'is_entitled': '',
    'language': '',
    'last_updated': 'metadata.last_updated',
    'content': 'text',
    'chunk_num': '',
    'publish_date': '',
    'digital_content_codes': 'metadata.dcc',
    'ibmdocskey': 'metadata.ibmdocskey',
    'description': 'metadata.description',
    'ibmdocsproduct': 'metadata.ibmdocsproduct',
    'title': 'title'
}


def add_to_doc(doc, new_key, value):
    if new_key.find('.') != -1:
        vals = new_key.split('.')
    else:
        vals = [new_key]
    r = doc
    for v in vals[:-1]:
        if v not in doc:
            doc[v] = {}
        r = doc[v]
    r[vals[-1]] = list(value) if type(value)==np.ndarray else value

def append(res, datum):
    for _index, _row in tqdm(datum.iterrows()):
        # Convert the row to a JSON string and append a newline character
        # res.append(row.to_dict())
        dd = _row.to_dict()
        doc = {}
        for k, v in dd.items():
            if k in field_map and field_map[k] != "":
                add_to_doc(doc, field_map[k], v)
        res.append(doc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='+', help="The input files")
    parser.add_argument('-l', '--list-keys', action="store_true",
                        help="If provided, it will list the keys in the first file and exit.")
    parser.add_argument('-e', '--exclude_keys',
                        help="If provided, it will ignore the given keys")
    parser.add_argument('-o', "--output", type=str, help="The output file")
    args = parser.parse_args()
    input_files = args.input_files
    result = list()

    if args.list_keys:
        df = pd.read_parquet(input_files[0])
        for index, row in df.iterrows():
            # print("Keys: ", "\n * ".join(row.keys()))
            for k, v in row.to_dict().items():
                print(f"{k}\t", end='')
                if type(v) is np.ndarray:
                    print(f"{v[:4]}")
                else:
                    print(f"{v}")
            exit(0)
    for file in input_files:
        df = pd.read_parquet(file)
        # add_to_doc(result, df)
        append(result, df)

    with open(args.output, 'w') as f:
        for doc in result:
            f.write(json.dumps(doc) + '\n')
