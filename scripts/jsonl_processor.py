import json
import argparse
import sys

from tqdm.auto import tqdm

from docuverse.utils import get_param

def process_jsonl(input_file, output_file):
    """
    Process JSONL input and create output with 'text', 'id', and 'document_id' fields.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in tqdm(enumerate(infile, 1)):
            try:
                # Parse the JSON line
                data = json.loads(line.strip())
                context = get_param(data, 'contexts')[0]
                text = get_param(context, "text")
                id = get_param(data, "task_id")
                answers = [get_param(context, 'document_id')]
                metadata = {
                    'title': get_param(context, 'title'),
                }
                outfile.write(json.dumps({'text': text, 'id': id, 'answers': answers,
                                          "metadata": metadata}) + '\n')

                # Extract contexts (documents)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON on line {line_num}: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}", file=sys.stderr)
                continue

def main():
    parser = argparse.ArgumentParser(description='Convert JSONL format to extract text, id, and document_id fields')
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('output_file', help='Output JSONL file path')
    
    args = parser.parse_args()
    
    try:
        process_jsonl(args.input_file, args.output_file)
        print(f"Successfully processed {args.input_file} -> {args.output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
