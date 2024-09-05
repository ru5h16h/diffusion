import collections
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
  ):
    if args is not None:
      default_configs["args"] = vars(args)
    self.to_save_configs = save_configs
    self.update(default_configs)
    if configs_path:
      cfg_args = load_dict_from_file(configs_path)
      self.update_rec(cfg_args)
    if self.to_save_configs:
      self.save_configs()

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
    utils.write_json(cfg_path, self)

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
        self[keys] = cfg_args[key]
      keys.pop()
