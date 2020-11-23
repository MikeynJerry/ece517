import argparse
import json
import matplotlib.pyplot as plt


def load_stats(path):
    with open(path, "r") as f:
        return json.load(f)


def plot(experiments, data):
    paths = [f"experiments/{experiment}/stats.json" for experiment in experiments]
    stats = [load_stats(path) for path in paths]

    for datakey in data:
        for experiment, name in zip(stats, experiments):
            plt.plot(range(len(experiment[datakey])), experiment[datakey], label=name)
        plt.legend()
        plt.title(datakey)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--experiments", nargs="+")
    parser.add_argument(
        "--data", nargs="+", default=["avg_loss", "loss_vs_iteration", "eps"]
    )

    args = parser.parse_args()

    plot(**args.__dict__)
