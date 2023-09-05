import os
from pathlib import Path

from src.constants import TASKS, FORMATS
from src.prompts import generation_prompt

OUTPUT_PATH = Path("../prompts")

if __name__ == '__main__':
    for task in TASKS:
        for output_format in FORMATS:
            if not os.path.exists(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)
            with open(f"{OUTPUT_PATH}/{task}_{output_format}.md", "w") as f:
                f.write(generation_prompt(task, output_format))
