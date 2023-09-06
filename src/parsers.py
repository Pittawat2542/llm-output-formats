import json
import xmltodict
import yaml


def parse_json(json_str: str) -> dict:
    if "```json" in json_str:
        json_str = json_str.replace("```json", "").replace("```", "")
    if "```" in json_str:
        json_str = json_str.replace("```", "")

    return json.loads(json_str, strict=False)


def parse_xml(xml_str: str) -> dict:
    if '<?xml version="1.0" ?>' in xml_str:
        xml_str = xml_str.replace('<?xml version="1.0" ?>', "")
    if "```xml" in xml_str:
        xml_str = xml_str.replace("```xml", "").replace("```", "")
    if "```" in xml_str:
        xml_str = xml_str.replace("```", "")

    return xmltodict.parse(xml_str)


def parse_yaml(yaml_str: str) -> dict:
    if "```yaml" in yaml_str:
        yaml_str = yaml_str.replace("```yaml", "").replace("```", "")
    if "```" in yaml_str:
        yaml_str = yaml_str.replace("```", "")

    return yaml.safe_load(yaml_str)
