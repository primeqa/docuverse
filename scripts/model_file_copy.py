import yaml
import os
import copy
import argparse

# Define template file name
template_yaml_file = "ibmsw_milvus_dense.granite-125m.test.short.yaml"

def create_model_files(input_file_name):
    """
    Reads an input file containing (model_name, short_name) pairs,
    updates a YAML template with these values, and saves new YAML files.
    """
    try:
        # Read the template YAML file
        with open(template_yaml_file, 'r') as f:
            template_data = yaml.safe_load(f)

        # Read the input file
        with open(input_file_name, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove leading/trailing parentheses and split by comma
            line = line.strip('()')
            parts = line.split(' ')
            if len(parts) != 2:
                print(f"Skipping malformed line: '{line}'. Expected format: 'model_name,short_name'.")
                continue

            new_model_name = parts[0].strip()
            new_short_name = parts[1].strip()

            # Create a deep copy of the template data to modify for each pair
            modified_data = copy.deepcopy(template_data)

            change_param(modified_data, "model_name", new_model_name)
            change_param(modified_data, "short_model_name", new_short_name.replace("-", "_"))

            # Update the 'short_model_name' parameter, assuming it's a top-level key
            # Define output file name
            output_yaml_file = template_yaml_file.replace("granite-125m", new_short_name)

            # Write the modified YAML to a new file
            with open(output_yaml_file, 'w') as f:
                yaml.safe_dump(modified_data, f, sort_keys=False, indent=4) # sort_keys=False to preserve order

            print(f"Generated '{output_yaml_file}' for model '{new_model_name}' and short name '{new_short_name}'.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure '{input_file_name}' and '{template_yaml_file}' exist in the same directory as this script.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{template_yaml_file}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def change_param(data, key, model_name_from_input):
    if isinstance(data['retriever'], dict)  and 'retriever' in data and key in data['retriever']:
        data['retriever'][key] = model_name_from_input
    else:
        print(
            f"Warning: 'config.model_name' key not found in '{template_yaml_file}'. "
            f"Skipping replacement for model name '{model_name_from_input}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate model files from template and input file.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input file containing model name and short name pairs')
    args = parser.parse_args()

    # Set global input_file from command line argument
    input_file = args.input_file

    create_model_files(input_file)
