import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

names = ['sgd', 'adam', 'rmsprop']

titles = {
    'sgd': 'SGD Optimizer',
    'adam': 'Adam Optimizer',
    'rmsprop': 'RMSprop Optimizer'
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, name in zip(axes, names):
    with open(f'results/{name}.json') as f:
        history = json.load(f)['history']

    ax.plot(history['accuracy'], marker='o', label='Training Accuracy')
    ax.plot(history['val_accuracy'], marker='o', label='Validation Accuracy')

    ax.set_title(titles[name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_xticks([0, 1, 2], ['1', '2', '3'])
    ax.legend()
    ax.grid(True)

fig.suptitle('Comparison of Different Optimizers')
fig.tight_layout()

fig.savefig('optimizers.png', dpi=150)

print('Optimizer comparison graph saved as optimizers.png')
