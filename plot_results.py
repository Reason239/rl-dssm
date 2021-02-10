import matplotlib.pyplot as plt
import pickle

experiment_path = 'experiments/'
experiment_name = 'test_32_8_l'

with open(experiment_path + experiment_name + '/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

train_losses = metrics['train_losses']
test_losses = metrics['test_losses']
train_accs = metrics['train_accs']
test_accs = metrics['test_accs']

plt.subplot(211)
plt.plot(train_losses, label='Train loss')
plt.plot(test_losses, label='Test loss')
plt.title('FPS Loss')
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.legend()

plt.subplot(212)
plt.plot(train_accs, label='Train accuracy')
plt.plot(test_accs, label='Test accuracy')
plt.title(f'Accuracy (max_test={max(test_accs):.2f})')
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend()

plt.subplots_adjust(hspace=0.5)
fig = plt.gcf()
fig.savefig(experiment_path + experiment_name + '/plot.png')
plt.show()
