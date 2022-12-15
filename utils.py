import csv
from collections import OrderedDict

import wandb

try:
    import wandb
except ImportError:
    pass


def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False, log_wandb=False, resume=""):
    summary = OrderedDict(epoch=epoch)
    summary.update([("train_" + k, v) for k, v in train_metrics.items()])
    summary.update([("eval_" + k, v) for k, v in eval_metrics.items()])

    if log_wandb and resume != "":
        with open(filename, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    row[key] = float(value)
                wandb.log(row)

    if log_wandb:
        wandb.log(summary)

    with open(filename, mode="a") as cf:
        writer = csv.DictWriter(cf, fieldnames=summary.keys())
        if write_header and (resume == ""):  # first iteration (epoch == 1 can't be used)
            writer.writeheader()
        writer.writerow(summary)

