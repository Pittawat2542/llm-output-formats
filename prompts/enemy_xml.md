Generate a enemy battle status. Return output in XML format and only the XML in the Markdown code block. XML.

# Output format
```xml
<?xml version="1.0" ?>
<game>
	<enemy>
		<id>id</id>
		<name>name</name>
		<description>description</description>
		<stats>
			<hp>hp int value</hp>
			<mp>mp int value</mp>
			<atk>atk int value</atk>
			<def>def int value</def>
			<spd>spd int value</spd>
		</stats>
	</enemy>
</game>
```