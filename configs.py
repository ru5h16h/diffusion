"""Module for handling configurations."""

import datetime
import os
import string
import sys
from typing import Any

import yaml

cfg = None


def get_current_ts():
  """Gets current timestamp."""
  return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


class Configs(dict):
  """Custom Dictionary to use it for accessing configurations.

  Attributes:
    path: Path to the configuration file.
  """

  def __init__(self, path: str):
    self.path = path
    self.load_yaml()

  def __getitem__(self, __key: Any) -> Any:
    if isinstance(__key, str):
      return super().__getitem__(__key)
    elif isinstance(__key, tuple):
      sub_dict = None
      for key in __key:
        if sub_dict is None:
          sub_dict = super().__getitem__(key)
        else:
          sub_dict = sub_dict[key]
      return sub_dict

  def replace_placeholders(self, data, sub):
    if isinstance(data, dict):
      for key, value in data.items():
        if isinstance(value, str):
          data[key] = string.Template(value).safe_substitute(sub)
        elif isinstance(value, (dict, list)):
          self.replace_placeholders(value, sub)
    elif isinstance(data, list):
      for i in range(len(data)):
        self.replace_placeholders(data[i], sub)
    return data

  def load_yaml(self):
    with open(self.path, 'r') as file:
      content = file.read()
    data = yaml.safe_load(content)
    ts = get_current_ts()
    data = self.replace_placeholders(
        data=data,
        sub={"timestamp": ts},
    )
    self.update(data)

  def dump_config(self):
    config_dir = self["dump_cfg_path"]
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "configs.yaml")
    with open(config_path, 'w') as yaml_file:
      yaml.dump(self, yaml_file)


if __name__ == "__main__":
  sys.exit("Intended for import.")