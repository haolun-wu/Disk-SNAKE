import argparse
from typing import Union


def parse_args(config):
    """
    Parse command line arguments based on a configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration options and their default values.

    Returns:
        NamespaceDict: A dictionary-like object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        elif isinstance(value, list):
            parser.add_argument(f"--{key}", type=type(value[0]), default=value, nargs="+")
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = NamespaceDict(parser.parse_args())
    return args


class NamespaceDict(dict, argparse.Namespace):
    def __init__(self, obj: Union[dict, argparse.Namespace]):
        """
        A dictionary-like object that allows attribute-style access to its keys.

        Args:
            obj (Union[dict, argparse.Namespace]): A dictionary or namespace object to
            initialize the NamespaceDict with.
        """
        if isinstance(obj, argparse.Namespace):
            obj = vars(obj)
        super().__init__(obj)

    def __getattr__(self, name):
        """
        Get the value of a key in the NamespaceDict.

        Args:
            name (str): The name of the key to get the value of.

        Returns:
            The value of the key.
        """
        return self[name]

    def __setattr__(self, name, value):
        """
        Set the value of a key in the NamespaceDict.

        Args:
            name (str): The name of the key to set the value of.
            value: The value to set the key to.
        """
        self[name] = value
