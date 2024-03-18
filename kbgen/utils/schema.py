from collections import defaultdict
from typing import Iterable


def get_ids(schema: dict, start_id: int = 0) -> dict:
    """This function flattens the schema into a dictionary of keys and ids.
    The keys are the paths from the root to the node.
    The ids are consecutive integers starting from start_id.
    The goal is to assign a unique id to each node in the tree.
    Args:
        schema (dict): The schema of the dataset.
        prefix (str, optional): The prefix of the key. Defaults to "".
        start_id (int, optional): The starting id. Defaults to 0.

    Returns:
        dict: Dictionary of keys and ids.
    """
    key_ids = {}

    def _get_ids(schema: dict, prefix: str = ""):
        for key, value in schema.items():
            if isinstance(key, str):
                key = prefix + key
                key_ids[key] = len(key_ids) + start_id
                if isinstance(value, dict):
                    _get_ids(value, key + ".")
            else:
                _get_ids(value, prefix)

    _get_ids(schema)
    return key_ids


def by_types(schema) -> "dict[int, list[str]]":
    """This function returns the types of each field in the schema. Essentially,
    it reverses the keys and values in the schema.
    Args:
        schema (dict): The schema of the dataset.
    Returns:
        dict: Dictionary of types and their respective fields in a list.
    """
    types = defaultdict(list)

    def _types(schema: dict, prefix: str = ""):
        for key, value in schema.items():
            if isinstance(value, int):
                types[value].append(prefix + key)
            elif isinstance(value, dict):
                _types(value, prefix + key + ".")
            else:
                raise ValueError(
                    f"Unexpected value type: {type(value)} for key: {key}, value: {value}"
                )

    _types(schema)
    return types


def match_fields_to_nodes(fields: Iterable[str], nodes: Iterable[str]):
    """This function matches the fields to the schema.
    Args:
        fields (list[str]): The fields to match.
        schema (list[str]): The node names (including path) to match to.
    Returns:
        dict: Dictionary of field name and the corresponding path name.
    """
    translations = {}
    for field in fields:
        for node in nodes:
            if node.endswith(field):
                translations[field] = node
    return translations
