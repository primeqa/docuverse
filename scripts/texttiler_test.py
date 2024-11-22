from docuverse.utils.text_tiler import TextTiler
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer

parser = ArgumentParser(description="Test for TextTiler")
parser.add_argument("file", type=str, help="The input file with the lines to split, one per line.")
if __name__ == '__main__':
    args = parser.parse_args()
    # model_name = "/home/raduf/mnlp-models-rag/06302024/slate.30m.english.rtrvr"
    model_name = "thenlper/gte-small"
    model = SentenceTransformer(model_name)
    max_size = 512
    stride = 100
    tiler = TextTiler(max_doc_size=max_size, stride=stride, tokenizer=model.tokenizer, aligned_on_sentences=True)

    with open(args.file) as inp:
        for line in inp:
            line = line.strip()
            res = tiler.split_text()
