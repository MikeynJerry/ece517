import argparse
import json
import matplotlib.pyplot as plt


def load_stats(path):
    with open(path, "r") as f:
        return json.load(f)


def plot(experiments, datakeys, as_subplots):
    paths = [
        f"experiments/{experiment}/testing_stats.json" for experiment in experiments
    ]
    stats = [load_stats(path) for path in paths]

    keys = [data.keys() for data in stats]

    if as_subplots:
        fig, axes = plt.subplots(ncols=len(datakeys), figsize=(4 * len(datakeys), 5))
        if len(datakeys) == 1:
            axes = [axes]
        for i, datakey in enumerate(datakeys):
            for j in range(len(experiments)):
                x = [int(key) for key in keys[j]]
                y = [
                    sum(stats[j][key][datakey]) / len(stats[j][key][datakey])
                    for key in keys[j]
                ]
                print(x, y)
                axes[i].plot(x, y, label=experiments[j])

            axes[i].legend()
            axes[i].set_title(datakey)
        plt.show()
    else:
        for datakey in datakeys:
            for i in range(len(experiments)):
                x = [int(key) for key in keys[i]]
                y = [
                    sum(stats[i][key][datakey]) / len(stats[i][key][datakey])
                    for key in keys[i]
                ]
                plt.plot(x, y, label=experiments[i])

            plt.legend()
            plt.title(datakey)
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--experiments", nargs="+")
    parser.add_argument("--datakeys", nargs="+", default=["wins", "scores"])
    parser.add_argument("--as-subplots", action="store_true")

    args = parser.parse_args()

    plot(**args.__dict__)
