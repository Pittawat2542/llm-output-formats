from src.constants import TASKS, FORMATS
from src.types import Task, OutputFormat
from src.utils import json_to_xml, json_to_yaml


def generation_prompt(task: Task = "story",
                      output_format: OutputFormat = "json"):
    guard_task(task)
    guard_output_format(output_format)

    task_name = get_task_name(task)
    output_template_json = get_task_template(task)

    output_template = None
    match output_format:
        case "json":
            output_template = output_template_json
        case "xml":
            output_template = json_to_xml(output_template_json)
        case "yaml":
            output_template = json_to_yaml(output_template_json)

    return f"""Generate a {task_name}. {get_magic_phrase(output_format)}

# Output format
```{output_format}
{output_template}
```"""


def guard_task(task: Task = "story"):
    if task not in TASKS:
        raise ValueError('`task` must be either "story", "quest", "character", "dialogue", or "enemy"')


def guard_output_format(output_format: OutputFormat = "json"):
    if output_format not in FORMATS:
        raise ValueError('`format` must be either "json", "xml", or "yaml"')


def get_magic_phrase(output_format: OutputFormat = "json"):
    guard_output_format(output_format)

    return f"Return output in {output_format.upper()} format and only the {output_format.upper()} in the Markdown code block. {output_format.upper()}."


def get_task_name(task: Task = "story"):
    if task not in TASKS:
        raise ValueError('`task` must be either "story" or "quest"')

    match task:
        case "story":
            return "game story synopsis"
        case "quest":
            return "quest information"
        case "character":
            return "character profile"
        case "dialogue":
            return "dialogue"
        case "enemy":
            return "enemy battle status"


def get_task_template(task: Task = "story"):
    if task not in TASKS:
        raise ValueError('`task` must be either "story" or "quest"')

    match task:
        case "story":
            return '''{
    "game": {
        "title": "game title",
        "synopsis": "game story synopsis",
        "beginning": "the beginning of the game",
        "ending": "the ending of the game"
    }
}'''
        case "quest":
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
        case "character":
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
        case "dialogue":
            return '''{
    "game": {
        "dialogues": [{
            "order": "dialogue order",
            "speaker": "speaker name",
            "text": "dialogue or narration"
        }]
    }
}'''
        case "enemy":
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
