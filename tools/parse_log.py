#!/usr/bin/env python

from typing import Mapping, Sequence
import re
import matplotlib.pyplot as plt


def parse_log(log_file: str, fields: Sequence[str]):
    """parse fields from the log file

    Args:
        log_file (str): the log file path.
        fields (Sequence[str]): The wanted fields.

    Returns: dict. key is the field name and value is the parsed values.

    """
    with open(log_file, "r") as f:
        lines = f.readlines()

    res = {}
    for field in fields:
        res[field] = []

    for line in lines:
        matches = re.findall(r"(\w+): ([0-9.]*[0-9])", line)
        for (k, v) in matches:
            if k in fields:
                res[k].append(float(v))

    return res


def plot_losses(losses: Mapping[str, Sequence], save_path: str):
    """TODO: Docstring for plot_losses.

    Args:
        losses (Mapping[str, Sequence]): Each (k,v) pair is the legend and the corresponding loss values.
        save_path (str): The figure save path.

    Returns: TODO

    """
    for (name, loss) in losses.items():
        x = list(range(1, len(loss) + 1))
        plt.plot(x, loss, label=name)
    plt.legend()
    plt.ylim([0, 0.6])
    plt.savefig(save_path)


def plot_exps(exps: Mapping[str, str], save_path: str):
    fields = ["loss"]
    losses = {}
    for (name, log_path) in exps.items():
        log = parse_log(log_path, fields)
        losses[name] = log["loss"]
    plot_losses(losses, save_path)


if __name__ == "__main__":
    exps = {
        "PF": "workdir/2.b.ii.3/20211201_003813.log",
        "Mem Bank": "workdir/3.a.i/20211208_162636.log",
        "Mem+LS-Attn": "workdir/3.c.i/20211213_160056.log",
        "Moment Mem": "workdir/3.d.i/20211214_044253.log",
    }
    save_path = "exp_loss_curves.png"
    plot_exps(exps, save_path)
