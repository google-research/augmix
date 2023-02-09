from argparse import ArgumentParser
from typing import Any, Dict, Literal, Optional, List, Type, get_args
import inspect
import argparse
from dataclasses import Field, MISSING
import logging
from typing_inspect import is_literal_type, is_optional_type
from simple_parsing.docstring import get_attribute_docstring

class UnsupportedTypeError(ValueError):
    """Error when a field has an unsupported type."""
    pass

class ArgparserMixin:
    """Mixin wich provides functionality to construct a argparser for an object and applies its data."""
    
    @classmethod
    def _map_type_to_parser_arg(cls, field: Field, _type: Optional[Type] = None) -> Dict[str, Any]:
        """Mapping field types to argparse arguments.

        Parameters
        ----------
        field : Field
            The field which should be mapped.
        _type : Optional[Type]
            Alterating field type on recursive calls, default None.
            
        Returns
        -------
        Dict[str, Any]
            kwargs for the argparse add argument call.

        Raises
        ------
        UnsupportedTypeError
            If the type is not supported for comparison.
        """
        if not _type:
            _type = field.type
        if isinstance(_type, Type) and issubclass(_type,(str, int, float)):
            return dict(type=_type)
        elif isinstance(_type, Type) and issubclass(_type, bool):
            # Check default and switch accordingly
            if not field.default:
                return dict(action="store_true")
            else:
                return dict(action="store_false")
        elif is_literal_type(_type):
            arg = get_args(_type)
            ret = dict()
            if len(arg) > 0:
                ret["choices"] = list(arg)
                ret["type"] = type(arg[0])
            return ret
        elif is_optional_type(_type):
            # Unpack optional type.
            _new_type = get_args(_type)[0]
            return cls._map_type_to_parser_arg(field, _new_type)
        else:
            raise UnsupportedTypeError(f"Dont know how to handle type: {_type} of field: {field.name}.")

    @classmethod
    def _get_parser_members(cls) -> List[Field]:
        """Returning the parser members which are in the dataclass.
        """
        # Get all dataclass properties
        members = inspect.getmembers(cls)
        all_fields:List[Field] = list(next((x[1] for x in members if x[0] == '__dataclass_fields__'), dict()).values())
        
        # Non private fields
        fields = [x for x in all_fields if not x.name.startswith('_')]
        return fields

    @classmethod
    def get_parser(cls, parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        """Creates / fills an Argumentparser with the fields of the current class.
        Inheriting class must be a dataclass to get annotations and fields.
        By default only puplic field are used (=field with a leading underscore "_" are ignored.)

        Parameters
        ----------
        parser : Optional[ArgumentParser]
            An existing argument parser. If not specified a new one will be created. Defaults to None.

        Returns
        -------
        ArgumentParser
            The filled argument parser.
        """
        # Create parser if None
        if not parser:
            parser = argparse.ArgumentParser(
                description=f'Default argument parser for {cls.__name__}',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        fields = cls._get_parser_members()

        for field in fields:
            name = field.name.replace("_", "-")
            try:
                args = cls._map_type_to_parser_arg(field)
            except UnsupportedTypeError as err:
                logging.warning(f"Could not create parser arg for field: {field.name} due to a {type(err).__name__} \n {str(err)}")
                continue
            # default value
            if field.default != MISSING:
                args["default"] = field.default
            # docstring 
            args["help"] = str(get_attribute_docstring(cls, field_name=field.name).docstring_below)
            parser.add_argument("--" + name, **args)

        return parser
    
    @classmethod
    def from_parsed_args(cls, parsed_args: Any) -> 'ArgparserMixin':
        """Creates an ArgparserMixin object from parsed_args which is the result
        of the argparser.parse_args() method.

        Parameters
        ----------
        parsed_args : Any
            The parsed arguments.

        Returns
        -------
        ArgparserMixin
            The instance of the object filled with cli data.
        """
        fields = cls._get_parser_members()
        # Look for matching fieldnames
        ret = dict()
        for field in fields:
            if hasattr(parsed_args, field.name):
                ret[field.name] = getattr(parsed_args, field.name)
        return cls(**ret)

    def apply_parsed_args(self, parsed_args: Any) -> None:
        """Applies parsed_args, which is the result
        of the argparser.parse_args() method, to an existing object.
        But only if its different to the objects default.

        Parameters
        ----------
        parsed_args : Any
            The parsed arguments.
        """
        fields = type(self)._get_parser_members()
        # Look for matching fieldnames
        ret = dict()
        for field in fields:
            if hasattr(parsed_args, field.name):
                value = getattr(parsed_args, field.name)
                if value != field.default:
                    setattr(self, field.name, value)
        