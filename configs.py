import ast
import collections
import logging
import sys
from typing import Any

import utils


def load_dict_from_file(file_path):
  local_vars = {}
  with open(file_path, 'r') as file:
    exec(file.read(), {}, local_vars)
  return local_vars.get('cfg', {})


class Configs(dict):

  def __init__(
      self,
      default_configs,
      configs_path=None,
      args=None,
      save_configs=True,
      config_args=None,
      type_hints=None,
  ):
    self.to_save_configs = save_configs

    if args is not None:
      default_configs["args"] = vars(args)
    if type_hints is not None:
      default_configs["type_hints"] = type_hints

    self.update(default_configs)

    if configs_path:
      cfg_args = load_dict_from_file(configs_path)
      self.update_rec(cfg_args)

    if config_args:
      self.update_config_args(config_args)

    if self.to_save_configs:
      self.save_configs()

  def update_config_args(self, config_args):
    for idx in range(0, len(config_args), 2):
      this_config = config_args[idx].strip()
      if not this_config.startswith("--configs"):
        raise ValueError(f"Invalid config args: {this_config}.")
      key = tuple(this_config.replace("--configs.", "").split("."))
      if key not in self:
        raise ValueError(f"Invalid key: {key}")
      value_type = self[tuple(["type_hints"] + list(key))]
      if value_type != str:
        value_type = ast.literal_eval
      new_val = value_type(config_args[idx + 1])
      logging.info(f"{key}: {self[key]} -> {new_val}")
      self[key] = new_val

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

  def save_configs(self):
    cfg_path = utils.get_path(self, "configs")
    utils.write_txt(cfg_path, self)

  def __setitem__(self, keys, value):
    if isinstance(keys, str):
      super().__setitem__(keys, value)
    elif isinstance(keys, collections.abc.Iterable):
      d = self
      *init_keys, last_key = keys
      for key in init_keys:
        if key not in d:
          d[key] = {}
        if not isinstance(d[key], dict):
          raise ValueError(
              f"Key {key} is already associated with a non-dictionary value.")
        d = d[key]
      d[last_key] = value
    if self.to_save_configs:
      self.save_configs()

  def update_rec(self, cfg_args, keys=[]):
    for key, val in cfg_args.items():
      keys.append(key)
      if isinstance(val, collections.abc.Mapping):
        self.update_rec(val, keys[:])
      else:
        logging.info(f"{tuple(keys)}: {self[tuple(keys)]} -> {cfg_args[key]}")
        self[tuple(keys)] = cfg_args[key]
      keys.pop()

  def __contains__(self, key: object) -> bool:
    if isinstance(key, str):
      return super().__contains__(key)
    elif isinstance(key, collections.abc.Iterable):
      current_dict = self
      for subkey in key:
        if isinstance(current_dict, dict) and subkey in current_dict:
          current_dict = current_dict[subkey]
        else:
          return False
      return True
    else:
      raise ValueError(f"Invalid key type: {type(key)}")


if __name__ == "__main__":
  sys.exit("Intended for import.")
