import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_curves(history):
    pd.DataFrame(history.history).plot()
    plt.title("Model training curves")
    plt.show()
