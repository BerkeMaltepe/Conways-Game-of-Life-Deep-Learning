import matplotlib.pyplot as plt

def plot_losses(losses, model_name):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title(f'Training Loss per Epoch for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'results/{model_name}_loss.png')
    plt.close()

