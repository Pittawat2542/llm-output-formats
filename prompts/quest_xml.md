Generate a quest information. Return output in XML format and only the XML in the Markdown code block. XML.

# Output format
```xml
<?xml version="1.0" ?>
<game>
	<id>id</id>
	<title>quest title</title>
	<objective>quest objective</objective>
	<description>quest description</description>
	<reward>quest reward</reward>
	<quest-giver>quest giver</quest-giver>
	<tasks>
		<task>
			<order>task order</order>
			<objective>task objective</objective>
			<description>task description</description>
			<location>task location</location>
		</task>
	</tasks>
</game>
```