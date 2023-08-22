Generate a quest information. Return output in YAML format and only the YAML in the Markdown code block. YAML.

# Output format
```yaml
game:
  description: quest description
  id: id
  objective: quest objective
  quest_giver: quest giver
  reward: quest reward
  tasks:
  - description: task description
    location: task location
    objective: task objective
    order: task order
  title: quest title
```