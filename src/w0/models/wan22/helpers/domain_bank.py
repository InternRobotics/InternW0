"""Dense domain banks keep a fixed parameter graph across mixed-domain ranks."""

import math
import copy

import torch
from torch import nn


def validate_domain_configs(configs, action_dim):
    configs = copy.deepcopy(list(configs or []))
    names = []
    for config in configs:
        for key in ("name", "dataset", "embodiment"):
            if not isinstance(config.get(key), str) or not config[key]:
                raise ValueError(f"domain_configs requires a nonempty {key}.")
        if config["name"] != config["dataset"] + "/" + config["embodiment"]:
            raise ValueError("Domain bank names must be dataset/embodiment.")
        for key in ("action_dim", "proprio_dim"):
            value = config.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Domain {key} must be a positive integer.")
        if config["action_dim"] > action_dim:
            raise ValueError("action_dim must cover every domain's action dimensions.")
        for kind in ("action", "proprio"):
            indices = config.get(kind + "_valid_indices")
            if indices is not None:
                if (not indices or len(indices) != len(set(indices)) or
                        any(type(i) is not int or i < 0 or i >= config[kind + "_dim"] for i in indices)):
                    raise ValueError(f"Invalid {kind}_valid_indices for {config['name']}")
        names.append(config["name"])
    if len(names) != len(set(names)):
        raise ValueError("Duplicate dataset/embodiment domain bank entry.")
    return configs


def domain_dimension_mask(configs, ids, width, kind):
    """Support explicit sparse slots while preserving legacy prefix domains."""
    masks = torch.zeros(len(configs), width, dtype=torch.bool, device=ids.device)
    for index, config in enumerate(configs):
        slots = config.get(kind + "_valid_indices", list(range(config[kind + "_dim"])))
        masks[index, slots] = True
    return masks[ids]


class DomainLinear(nn.Module):
    """Per-sample linear maps stored as one dense weight/bias parameter bank."""

    def __init__(self, in_features, out_features, num_domains):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(num_domains, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(num_domains, out_features))
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1 / math.sqrt(in_features), 1 / math.sqrt(in_features))

    def forward(self, x, domain_id):
        b = x.shape[0]
        weights, bias = self.weight[domain_id], self.bias[domain_id]
        out = torch.bmm(x.reshape(b, -1, self.in_features), weights.transpose(1, 2)) + bias[:, None]
        return out.reshape(*x.shape[:-1], self.out_features)
