Generate a character profile. Return output in XML format and only the XML in the Markdown code block. XML.

# Output format
```xml
<?xml version="1.0" ?>
<game>
	<character>
		<id>id</id>
		<first-name>first name</first-name>
		<last-name>last name</last-name>
		<species>species</species>
		<age>exact age or description</age>
		<role>role of the character</role>
		<background>background story</background>
		<place-of-birth>location</place-of-birth>
		<physical-appearance>
			<physical-appearance>
				<eye-color>eye color</eye-color>
				<hair-color>hair color</hair-color>
				<height>height in float value</height>
				<weight>weight in float value</weight>
			</physical-appearance>
		</physical-appearance>
	</character>
</game>
```