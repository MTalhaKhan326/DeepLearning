import torch
import matplotlib.pyplot as plt

def plot_data(X, y):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    feature_names = ['age', 'size', 'distance', 'floor']
    correlations = [torch.corrcoef(torch.stack([X[:, i], y[:, 0]]))[0, 1].abs().item() for i in range(4)]
    top2 = sorted(range(4), key=lambda i: correlations[i], reverse=True)[:2]
    for ax, i in zip(axes, top2):
        ax.scatter(X[:, i].numpy(), y.numpy(), alpha=0.4, s=10)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('price')
        ax.set_title(f'{feature_names[i]} vs price')
    plt.tight_layout()
    plt.show()

def plot_fit(X, y, model):
    feature_names = ['age', 'size', 'distance', 'floor']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, ax in enumerate(axes):
        x_range = torch.linspace(X[:, i].min(), X[:, i].max(), 100)
        X_sweep = torch.zeros(100, 4)
        X_sweep[:, i] = x_range
        with torch.no_grad():
            y_sweep = model.forward(X_sweep).squeeze().numpy()
        ax.scatter(X[:, i].numpy(), y.numpy(), alpha=0.3, s=10, label='data')
        ax.plot(x_range.numpy(), y_sweep, color='red', linewidth=2, label='model')
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('price')
        ax.set_title(feature_names[i])
    plt.tight_layout()
    plt.show()