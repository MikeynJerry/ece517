import matplotlib.pyplot as plt
from itertools import accumulate

sequences = [
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 4, 4, 4, 4, 4],
    [2, 4, 2, 4, 2, 4, 2, 4, 2, 4],
    [4, 4, 4, 4, 4, 2, 2, 2, 2, 2]
]
rewards = [0] * 3
for series, sequence in enumerate(sequences):
    for alpha in [0.5, 0.25, 1]:
        q = list(
            accumulate(sequence, lambda q, reward: q + alpha * (reward - q), initial=0)
        )
#        q = [0] * 11
#        for n, reward in enumerate(sequence):
#            q[n + 1] = q[n] + alpha * (reward - q[n])

        print(f"Alpha: {alpha}, Series {series+1}, Last Update: {q[-1]}")
        plt.plot(q, label=f"alpha = {alpha}")
    mean = [sum(sequence) / len(sequence)] * 11
    plt.plot(mean, label="Average Reward", linestyle='--')
    plt.title(f"Series {series+1}")

    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig(f'series_{series+1}.png', dpi=300)
    plt.show()