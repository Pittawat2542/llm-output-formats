Generate a quest information. Return output in JSON format and only the JSON in the Markdown code block. JSON.

# Output format
```json
{
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
}
```