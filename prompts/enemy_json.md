Generate a enemy battle status. Return output in JSON format and only the JSON in the Markdown code block. JSON.

# Output format
```json
{
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
}
```