import os
from pathlib import Path

from src.prompts import generation_prompt, possible_tasks, possible_output_formats

OUTPUT_PATH = Path("../prompts")

if __name__ == '__main__':
    for task in possible_tasks:
        for output_format in possible_output_formats:
            if not os.path.exists(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)
            with open(f"{OUTPUT_PATH}/{task}_{output_format}.md", "w") as f:
                f.write(generation_prompt(task, output_format))
