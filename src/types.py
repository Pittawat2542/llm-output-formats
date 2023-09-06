import typing

from src.constants import TASKS, FORMATS

Task: typing.List[TASKS] = list(typing.get_args(TASKS))
OutputFormat: typing.List[FORMATS] = list(typing.get_args(FORMATS))
