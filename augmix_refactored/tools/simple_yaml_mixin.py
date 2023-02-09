from typing import Any
import yaml
import os
from yaml import Loader, Dumper

class SimpleYamlMixin:
    """Adds support for reading / writing the object, which should be a dataclass to yaml."""

    def save_as_yaml(self, file: str) -> str:
        """Saves the current object in yaml format.

        Parameters
        ----------
        file : str
            The file path to save. Should have yaml or yml extension.

        Returns
        -------
        str
            The file path.
        """
        # Assuming a dataclass with underlying dict
        as_dict = vars(self)
        # Wrapping object
        parent_dict = dict()
        parent_dict[type(self).__name__] = as_dict
        
        # Dumping to yaml
        yaml_str = yaml.dump(parent_dict)
        
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "w") as f:
            f.write(yaml_str)
        return file

    @classmethod
    def load_from_yaml(cls, file: str) -> Any:
        """Loads the current instance from a file in yaml format.

        Parameters
        ----------
        file : str
            The file to load.

        Returns
        -------
        Any
            An instance of current cls.

        Raises
        ------
        ValueError
            If file not exists, or is wrongly formatted.
        """
        if not os.path.exists(file):
            raise ValueError(f"The file: {file} dont exists. Can not load {type(cls).__name__}")
        as_dict =  None
        with open(file, "r") as f:
            as_dict = yaml.safe_load(f) 
        key = cls.__name__
        if key not in as_dict:
            raise ValueError(f"Could not find {key} within document root. Is it correctly formatted?")
        content = as_dict[key]
        # Create Object
        return cls(**content)
        