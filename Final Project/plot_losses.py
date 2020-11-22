import json
import matplotlib.pyplot as plt


def create_plots(datasets):
    paths = ["experiments/"+d+"/stats.json" for d in datasets]
    
    for i, path in enumerate(paths):
        with open(path, "r") as f:
            stats = json.load(f)
        
        losses = stats["avg_loss"]
        plt.plot(range(len(losses)), losses, label=datasets[i])
    plt.legend()
    plt.title("Loss")
    plt.show()
    
    '''    
    eps = stats["eps"]
    plt.plot(range(len(eps)), eps)
    plt.title("Eps")
    plt.show()'''


datasets = ["priority_5000_mse", "basicreplay_5000_mse"]
create_plots(datasets)