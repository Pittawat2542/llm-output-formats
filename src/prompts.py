from typing import Literal

from src.utils import json_to_xml, json_to_yaml

possible_tasks = ["story", "quest", "character", "dialogue", "enemy"]
possible_output_formats = ["json", "xml", "yaml"]
Task = Literal["story", "quest", "character", "dialogue", "enemy"]
OutputFormat = Literal["json", "xml", "yaml"]


def generation_prompt(task: Task = "story",
                      output_format: OutputFormat = "json"):
    guard_task(task)
    guard_output_format(output_format)

    task_name = get_task_name(task)
    output_template_json = get_task_template(task)

    output_template = None
    if output_format == "json":
        output_template = output_template_json
    elif output_format == "xml":
        output_template = json_to_xml(output_template_json)
    elif output_format == "yaml":
        output_template = json_to_yaml(output_template_json)

    return f"""Generate a {task_name}. {get_magic_phrase(output_format)}

# Output format
```{output_format}
{output_template}
```"""


def guard_task(task: Task = "story"):
    if task not in possible_tasks:
        raise ValueError('`task` must be either "story", "quest", "character", "dialogue", or "enemy"')


def guard_output_format(output_format: OutputFormat = "json"):
    if output_format not in possible_output_formats:
        raise ValueError('`format` must be either "json", "xml", or "yaml"')


def get_magic_phrase(output_format: OutputFormat = "json"):
    guard_output_format(output_format)

    return f"Return output in {output_format.upper()} format and only the {output_format.upper()} in the Markdown code block. {output_format.upper()}."


def get_task_name(task: Task = "story"):
    if task not in possible_tasks:
        raise ValueError('`task` must be either "story" or "quest"')

    if task == "story":
        return "game story synopsis"
    elif task == "quest":
        return "quest information"
    elif task == "character":
        return "character profile"
    elif task == "dialogue":
        return "dialogue"
    elif task == "enemy":
        return "enemy battle status"


def get_task_template(task: Task = "story"):
    if task not in possible_tasks:
        raise ValueError('`task` must be either "story" or "quest"')

    if task == "story":
        return '''{
    "game": {
        "title": "game title",
        "synopsis": "game story synopsis",
        "beginning": "the beginning of the game",
        "ending": "the ending of the game"
    }
}'''
    elif task == "quest":
        return '''{
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
}'''
    elif task == "character":
        return '''{
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
}'''
    elif task == "dialogue":
        return '''{
    "game": {
        "dialogues": [{
            "order": "dialogue order",
            "speaker": "speaker name",
            "text": "dialogue or narration"
        }]
    }
}'''
    elif task == "enemy":
        return '''{
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
}'''
