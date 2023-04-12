#!/usr/bin/env python
# coding=utf-8
"""This Python code defines a class Dataset with methods for initializing, loading,
and manipulating datasets from different backends such as Hugging Face and JSON.
 
The `Dataset` class includes methods for loading datasets from a dictionary and a Hugging
Face dataset, mapping datasets, and retrieving the backend dataset and arguments.
"""



# Importing necessary libraries and modules
import json
import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from datasets import Dataset as HFDataset

from lmflow.args import DatasetArguments

DATASET_TYPES = [
    "text_only",
    "text2text",
    "chat_list"
]

KEY_TYPE = "type"
KEY_INSTANCES = "instances"

class Dataset:
    r"""
    Initializes the Dataset object with the given parameters.

    Parameters
    ------------
    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    backend : str,  default="huggingface"
        A string representing the dataset backend. Defaults to "huggingface".
    
    args : Optional.
        Positional arguments.
    
    kwargs : Optional.
        Keyword arguments.
    """
    def __init__(self, data_args=None, backend: str="huggingface", *args, **kwargs):
        self.data_args = data_args
        self.backend = backend
        self.backend_dataset = None
        self.type = None        # Original type of the dataset
        self.dataset_path = data_args.dataset_path

        if data_args.dataset_path is None:
            return

        if backend == "huggingface":
            data_files = [
                x.absolute().as_posix()
                 for x in Path(self.dataset_path).glob("*.json")
            ]

            # Iterate through all the files and ensure they have the same data type
            for single_file in data_files:
                with open(single_file) as fin:
                    json_data = json.load(fin)
                    if KEY_TYPE not in json_data.keys():
                        raise ValueError(
                            f'"{KEY_TYPE}" field must be specified for data, e.g.'
                            '{\n'
                            f'   "{KEY_TYPE}: "text_only",\n'
                            f'   "{KEY_INSTANCES}": [\n'
                            '       { "text": "Sentence 1: This is a sentence." }\n'
                            '       { "text": "Sentence 2: This is another sentence." }\n'
                            f'   ]\n'
                            '}'
                        )

                    if self.type is None:
                        self.type = json_data[KEY_TYPE]
                    elif self.type != json_data[KEY_TYPE]:
                        raise ValueError(
                            'All task files must have same data types. Previous'
                            f' files have type "{self.type}", but in file'
                            f' {single_file}, it has type "{self.type}".'
                        )

            # Load the dataset using the HuggingFace dataset library
            extensions = "json"
            raw_dataset = load_dataset(
                extensions,
                data_files=data_files,
                field=KEY_INSTANCES,
                split="train",
                use_auth_token=None,
            )
            self.backend_dataset = raw_dataset
        elif backend == "json":
            # TODO (@Jiachun)
            pass
        else:
            raise NotImplementedError(f'Unsupported dataset backend "{backend}"')

    def log_examples(self, num_examples: int=3):
        r"""
        Log the first few examples from the dataset.

        Parameters
        ------------

        num_examples : int, default=5
            The number of examples to log.
        """
        if self.backend == "huggingface":
            if int(os.getenv("LOCAL_RANK", "-1")) <= 0:
                for i in range(num_examples):
                    print(f"Dataset example {i}: {self.backend_dataset[i]}")
        else:
            raise NotImplementedError(
                f'Currently .log_examples is not supported for backend "{backend}"'
            )

    def clean_empty_example(self, examples):
        return [x for x in examples if x['fields']!=""]

    def _check_data_type(self):
        # TODO: check if data type and data structure matches, raise messages
        # with hints
        pass


    def from_dict(self, dict_obj: dict, *args, **kwargs):
        r"""
        Create a Dataset object from a dictionary.

        Return a Dataset given a dict with format:
            {
                "type": TYPE,
                "instances": [
                    {
                        "key_1": VALUE_1.1,
                        "key_2": VALUE_1.2,
                        ...
                    },
                    {
                        "key_1": VALUE_2.1,
                        "key_2": VALUE_2.2,
                        ...
                    },
                    ...
                ]
            }

        Parameters
        -----------

        dict_obj : dict.
            A dictionary containing the dataset information.
        
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.

        Returns
        ---------

        self : Dataset object.
        """
        if self.backend == "huggingface":
            if KEY_TYPE not in dict_obj:
                raise ValueError(
                    f'"{KEY_TYPE}" must be provided to initialize a dataset'
                )
            if KEY_INSTANCES not in dict_obj:
                raise ValueError(
                    f'"{KEY_INSTANCES}" must be provided to initialize a dataset'
                )

            self.type = dict_obj[KEY_TYPE]

            hf_dict = {}
            if len(dict_obj[KEY_INSTANCES]) > 0:
                for key in dict_obj[KEY_INSTANCES][0].keys():
                    hf_dict[key] = [ instance[key] for instance in dict_obj[KEY_INSTANCES] ]

            self.backend_dataset = HFDataset.from_dict(hf_dict, *args, **kwargs)
            return self
        else:
            raise NotImplementedError(
                f'Currently .from_dict is not supported for backend "{backend}"'
            )


    @classmethod
    def create_from_dict(cls, dict_obj, *args, **kwargs):
        r"""
        Returns
        --------

        Returns a Dataset object given a dict.
        """
        empty_data_args = DatasetArguments(dataset_path=None)
        dataset = Dataset(empty_data_args)
        return dataset.from_dict(dict_obj)


    def to_dict(self):
        r"""
        Returns
        ---------

        Return a dict represents the dataset:
            {
                "type": TYPE,
                "instances": [
                    {
                        "key_1": VALUE_1.1,
                        "key_2": VALUE_1.2,
                        ...
                    },
                    {
                        "key_1": VALUE_2.1,
                        "key_2": VALUE_2.2,
                        ...
                    },
                    ...
                ]
            }

        A python dict object represents the content of this dataset.
        """
        if self.backend == "huggingface":
            dict_obj = {}
            dict_obj[KEY_TYPE] = self.get_type()

            hf_dict = self.backend_dataset.to_dict()
            dict_obj[KEY_INSTANCES] = []

            first_key = None
            for key in hf_dict.keys():
                first_key = key
                break

            if first_key is not None:
                num_instances = len(hf_dict[first_key])
                dict_obj[KEY_INSTANCES] = [
                    {
                        key: hf_dict[key][i] for key in hf_dict.keys()
                    }
                    for i in range(num_instances)
                ]

            return dict_obj
        else:
            raise NotImplementedError(
                f'Current .to_dict is not supported for backend "{backend}"'
            )


    def map(self, *args, **kwargs):
        r"""
        Parameters
        ------------
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.

        Returns
        ---------

        self : Dataset object.
        """
        # If the dataset uses Hugging Face as the backend, 
        # call the `map()` function of the Hugging Face backend dataset
        if self.backend == "huggingface":
            # Set the mapped dataset as the backend dataset of the current dataset
            mapped_backend_dataset = self.backend_dataset.map(*args, **kwargs)
            self.backend_dataset = mapped_backend_dataset
            return self
        else:
            # If the backend is not Hugging Face, raise a NotImplementedError
            raise NotImplementedError(
                f'Currently .map is not supported for backend "{backend}"'
            )


    def get_backend(self) -> Optional[str]:
        r"""
        Returns
        ---------

        self.backend
        """
        return self.backend


    def get_backend_dataset(self):
        r"""
        Returns
        ---------

        self.backend_dataset
        """
        return self.backend_dataset

    
    def get_data_args(self):
        r"""
        Returns
        ---------

        self.data_args
        """
        return self.data_args


    def get_type(self):
        r"""
        Returns
        ---------

        self.type
        """
        return self.type
