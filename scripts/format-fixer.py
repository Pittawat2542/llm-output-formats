import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from src.parsers import extract_content, parse_json, parse_xml, parse_yaml
from openai import OpenAI
import json
from tqdm import tqdm
import argparse

from src.prompts import get_task_template
from src.utils import json_to_xml, json_to_yaml

fixing_prompt = """Fix the following incorrect {format} data. Correct the syntax and provide new values if needed. If there is nothing wrong, only return the {format} in a code block (between ```{format_lower} and ```). The original message is provided between ```{format_lower} and ```. Return output in {format} format and only the {format} in the Markdown code block. {format}.

Error: {error}

Template:
{template}

Original:
```{format_lower}
{original}
```

Fixed:"""


def fix_format(root_path: Path, error_categories: pd.DataFrame, fixer_model: str):
    output_path = root_path / "fixed_outputs" / f"{fixer_model}"
    parsed_output_path = root_path / "parsed_fixed_outputs" / f"{fixer_model}"
    output_path.mkdir(exist_ok=True, parents=True)
    parsed_output_path.mkdir(exist_ok=True, parents=True)

    openai = OpenAI(
        base_url=os.getenv("LOCAL_BASE_URL"),
        api_key="lmstudio"
    )

    for i, row in tqdm(error_categories.iterrows(), total=error_categories.shape[0]):
        error_category = row["error"]
        task = row["task"]
        model = row["model"]
        output_format = row["format"]
        file = row["file"]

        file_path = llms_path_outputs / task / model / output_format / file
        fixed_file_path = output_path / task / model / output_format / file
        parsed_fixed_file_path = parsed_output_path / task / model / output_format / file.replace(".txt", ".json")

        if fixed_file_path.exists():
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="ISO-8859-1") as f:
                data = f.read()

        template = get_task_template(task)
        if output_format == "xml":
            template = json_to_xml(template)
        elif output_format == "yaml":
            template = json_to_yaml(template)

        original_data = extract_content(data, output_format)
        error_msg = f"{error_category}: {row['error_message']}"
        prompt = fixing_prompt.format(
            format=output_format.upper(),
            format_lower=output_format,
            error=error_msg,
            original=original_data,
            template=template
        )

        response = openai.chat.completions.create(model=fixer_model, messages=[{"role": "user", "content": prompt}])
        fixed = response.choices[0].message.content

        fixed_data = None
        if output_format == "json":
            try:
                fixed_data = parse_json(fixed)

            except Exception as e:
                print(f"Error parsing fixed data: {e}")
        elif output_format == "xml":
            try:
                fixed_data = parse_xml(fixed)
            except Exception as e:
                print(f"Error parsing fixed data: {e}")
        elif output_format == "yaml":
            try:
                fixed_data = parse_yaml(fixed)
            except Exception as e:
                print(f"Error parsing fixed data: {e}")

        fixed_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fixed_file_path, "w", encoding="utf-8") as f:
            f.write(fixed)

        if fixed_data is not None:
            parsed_fixed_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(parsed_fixed_file_path, "w", encoding="utf-8") as f:
                json.dump(fixed_data, f, indent=2, default=str)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../experiment_results")
    parser.add_argument("--fixer_model", type=str)
    args = parser.parse_args()
    root_path = Path(args.root_path)
    fixer_model = args.fixer_model

    llms_path_outputs = root_path / "outputs"
    summarized_error_category_file = llms_path_outputs / "filtered_error_categories.csv"

    error_categories = pd.read_csv(summarized_error_category_file)

    fix_format(root_path, error_categories, fixer_model)
