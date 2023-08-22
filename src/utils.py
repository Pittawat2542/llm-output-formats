import json
import xml.dom.minidom
import inflect
import yaml
from collections.abc import Iterable


def json_to_xml(json_string):
    def dict_to_xml(dictionary, xml_string):
        for key, value in dictionary.items():
            key = str(key).replace('_', '-')
            if isinstance(value, dict):
                xml_string += f'<{key}>{dict_to_xml(value, "")}</{key}>'
            elif isinstance(value, Iterable) and not isinstance(value, str):
                xml_string += f'<{key}>'
                wrapper = inf.singular_noun(key) if inf.singular_noun(key) is not False else key
                for item in value:
                    xml_string += f'<{wrapper}>{dict_to_xml(item, "")}</{wrapper}>'
                xml_string += f'</{key}>'
            else:
                xml_string += f'<{key}>{value}</{key}>'
        return xml_string.strip()

    inf = inflect.engine()
    xml_str = f'{dict_to_xml(json.loads(json_string), "")}'
    xml_str = f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_str}'
    xml_obj = xml.dom.minidom.parseString(xml_str)
    return xml_obj.toprettyxml().strip()


def json_to_yaml(json_string):
    yaml_string = yaml.dump(json.loads(json_string))
    return yaml_string.strip()
