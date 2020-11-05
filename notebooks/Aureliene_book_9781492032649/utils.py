# let's add some code to have each job emit logs in a particular location
import os
import pandas as pd
import matplotlib.pyplot as plt

root_logdir = os.path.join(os.curdir, "ml_logs")

def get_run_logidr():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def draw_history(history):
    pd.DataFrame(history.history).plot(figsize=(10, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

   
def print_history(history):
    draw_history(history)
 
def draw_histories(histories, names):
    fig = plt.figure(figsize=(20, 10))
    legend = []
    for history, name in zip(histories, names):
        df = pd.DataFrame(history.history)
        plt.plot(df)
        legend.extend(list(f"{name}-" + df.columns))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
    plt.legend(legend)
    plt.show()
