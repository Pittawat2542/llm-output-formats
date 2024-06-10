from pathlib import Path
import pandas as pd

from src.parsers import parse_json, parse_xml, parse_yaml

templates = {
    "character": {
        "game": {
            "character": {
                "id": "id",
                "first_name": "first name",
                "last_name": "last name",
                "species": "species",
                "age": "exact age or description",
                "role": "role of the character",
                "background": "background story",
                "place_of_birth": "location",
                "physical_appearance": [{
                    "eye_color": "eye color",
                    "hair_color": "hair color",
                    "height": "height in float value",
                    "weight": "weight in float value"
                }]
            }
        }
    },
    "dialogue": {
        "game": {
            "dialogues": [{
                "order": "dialogue order",
                "speaker": "speaker name",
                "text": "dialogue or narration"
            }]
        }
    },
    "enemy": {
        "game": {
            "enemy": {
                "id": "id",
                "name": "name",
                "description": "description",
                "stats": {
                    "hp": "hp int value",
                    "mp": "mp int value",
                    "atk": "atk int value",
                    "def": "def int value",
                    "spd": "spd int value"
                }
            }
        }
    },
    "quest": {
        "game": {
            "id": "id",
            "title": "quest title",
            "objective": "quest objective",
            "description": "quest description",
            "reward": "quest reward",
            "quest_giver": "quest giver",
            "tasks": [{
                "order": "task order",
                "objective": "task objective",
                "description": "task description",
                "location": "task location"
            }]
        }
    },
    "story": {
        "game": {
            "title": "game title",
            "synopsis": "game story synopsis",
            "beginning": "the beginning of the game",
            "ending": "the ending of the game"
        }
    }
}

parsers = {
    "json": parse_json,
    "xml": parse_xml,
    "yaml": parse_yaml
}


def error_analysis(output_folder_path: Path):
    tasks = [task for task in output_folder_path.iterdir() if task.is_dir()]
    error_categories = {
        "empty_response": [],
        "incorrect_syntax": [],
        "key_completeness": [],
    }
    for task in tasks:
        models = [model for model in task.iterdir() if model.is_dir()]
        for model in models:
            formats = [format for format in model.iterdir() if format.is_dir()]
            for format in formats:
                files = [file for file in format.iterdir() if file.is_file()]
                for file in files:
                    try:
                        with open(file, "r", encoding="utf-8") as f:
                            text = f.read()
                    except UnicodeDecodeError:
                        with open(file, "r", encoding="ISO-8859-1") as f:
                            text = f.read()

                    result = None
                    parser = parsers[format.parts[-1]]
                    try:
                        result = parser(text)
                    except ValueError as e:
                        error_categories["empty_response"].append({
                            "task": task.name,
                            "model": model.name,
                            "format": format.name,
                            "file": file.name,
                            "error": "empty_response",
                            "error_message": str(e)
                        })
                    except Exception as e:
                        error_categories["incorrect_syntax"].append({
                            "task": task.name,
                            "model": model.name,
                            "format": format.name,
                            "file": file.name,
                            "error": "incorrect_syntax",
                            "error_message": str(e)
                        })

                    if result is not None:
                        if type(result) != dict:
                            error_categories["key_completeness"].append({
                                "task": task.name,
                                "model": model.name,
                                "format": format.name,
                                "file": file.name,
                                "error": "key_completeness",
                                "error_message": "The root of the result is not a dictionary."
                            })
                            continue
                        for key in templates[task.name]:
                            if key not in result and key.replace("_", "-") not in result:
                                error_categories["key_completeness"].append({
                                    "task": task.name,
                                    "model": model.name,
                                    "format": format.name,
                                    "file": file.name,
                                    "error": "key_completeness",
                                    "error_message": f"The key '{key}' is missing."
                                })
                            else:
                                if type(templates[task.name][key]) != dict:
                                    continue
                                for sub_key in templates[task.name][key]:
                                    if type(result[key]) != dict:
                                        error_categories["key_completeness"].append({
                                            "task": task.name,
                                            "model": model.name,
                                            "format": format.name,
                                            "file": file.name,
                                            "error": "key_completeness",
                                            "error_message": f"The key '{key}' is not a dictionary."
                                        })
                                        break
                                    if sub_key not in result[key] and sub_key.replace("_", "-") not in result[key]:
                                        error_categories["key_completeness"].append({
                                            "task": task.name,
                                            "model": model.name,
                                            "format": format.name,
                                            "file": file.name,
                                            "error": "key_completeness",
                                            "error_message": f"The key '{sub_key}' is missing."
                                        })

    df = pd.DataFrame(error_categories["empty_response"] + error_categories["incorrect_syntax"] + error_categories[
        "key_completeness"])
    df = df.groupby(["task", "model", "format", "file", "error"]).agg(
        {"error_message": lambda x: ", ".join(x)}).reset_index()
    df.to_csv(output_folder_path / "error_categories.csv", index=False)
