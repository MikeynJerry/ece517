import json
import matplotlib.pyplot as plt

with open("stats.json", "r") as f:
    stats = json.load(f)

losses = stats["losses"]
plt.plot(range(len(losses)), losses)
plt.title("Loss")
plt.show()

eps = stats["eps"]
plt.plot(range(len(eps)), eps)
plt.title("Eps")
plt.show()
