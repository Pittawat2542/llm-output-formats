import json
import re

import xmltodict
import yaml


def extract_content(content: str, format: str) -> str:
    quote_pattern = {
        "json": r'```(json)?\n([\s\S]*)```',
        "xml": r'```(xml)?\n(<\?xml version=("|\')1.0("|\').*\?>)?([\s\S]*)```',
        "yaml": r'```(yaml)?\n([\s\S]*)```'
    }

    match = re.findall(quote_pattern[format], content, re.DOTALL)

    if len(match) == 0:
        return ""

    return match[-1][-1]


def parse_json(json_str: str) -> dict:
    content = extract_content(json_str, "json")

    if content == "":
        raise ValueError("No JSON content found")

    return json.loads(content, strict=False)


def parse_xml(xml_str: str) -> dict:
    content = extract_content(xml_str, "xml")

    if content == "":
        raise ValueError("No XML content found")

    return xmltodict.parse(content)


def parse_yaml(yaml_str: str) -> dict:
    content = extract_content(yaml_str, "yaml")

    if content == "":
        raise ValueError("No YAML content found")

    return yaml.safe_load(content)
