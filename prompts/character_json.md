Generate a character profile. Return output in JSON format and only the JSON in the Markdown code block. JSON.

# Output format
```json
{
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
}
```