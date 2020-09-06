'''
Code for Problem 6
'''

from itertools import accumulate

sequences = [
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], # Series 1
    [2, 2, 2, 2, 2, 4, 4, 4, 4, 4], # Series 2
    [2, 4, 2, 4, 2, 4, 2, 4, 2, 4], # Series 3
    [4, 4, 4, 4, 4, 2, 2, 2, 2, 2]  # A down-step series
]

for series, sequence in enumerate(sequences):
    for alpha in [0.5, 0.25, 1]:
        q = list(
            accumulate(sequence, lambda q, reward: q + alpha * (reward - q), initial=0)
        )

        print(f"Alpha: {alpha}, Series {series+1}, Last Update: {q[-1]}")