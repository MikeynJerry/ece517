import argparse
import json
import matplotlib.pyplot as plt


def load_stats(path):
    with open(path, "r") as f:
        return json.load(f)


def plot(experiments, datakeys, as_subplots):
    paths = [f"experiments/{experiment}/stats.json" for experiment in experiments]
    stats = [load_stats(path) for path in paths]

    if as_subplots:
        fig, axes = plt.subplots(ncols=len(datakeys), figsize=(4 * len(datakeys), 5))
        if len(datakeys) == 1:
            axes = [axes]
        for i, datakey in enumerate(datakeys):
            for experiment, name in zip(stats, experiments):
                axes[i].plot(
                    range(len(experiment[datakey])), experiment[datakey], label=name
                )
            axes[i].legend()
            axes[i].set_title(datakey)
        plt.show()
    else:
        for datakey in datakeys:
            for experiment, name in zip(stats, experiments):
                plt.plot(
                    range(len(experiment[datakey])), experiment[datakey], label=name
                )
            plt.legend()
            plt.title(datakey)
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--experiments", nargs="+")
    parser.add_argument(
        "--datakeys", nargs="+", default=["avg_loss", "loss_vs_iteration", "eps"]
    )
    parser.add_argument("--as-subplots", action="store_true")

    args = parser.parse_args()

    plot(**args.__dict__)
