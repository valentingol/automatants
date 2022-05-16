import matplotlib.pyplot as plt

def plt_curves(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    x = range(len(history['train_loss']))
    plt.title('Loss')
    plt.plot(x, history['train_loss'], 'r--', label='train')
    plt.plot(x, history['val_loss'], 'r', label='val')
    plt.ylim(0, 1.0)
    plt.legend(loc='best')
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(x, history['train_metric'], 'b--', label='train')
    plt.plot(x, history['val_metric'], 'b', label='val')
    plt.ylim(0, 1.0)
    plt.legend(loc='best')


def plt_curves_test(test_loss, test_metric):
    # Plot the test loss and test accuracy
    print(f'Test loss: {test_loss: .4f}\n'
          f'Test metric: {test_metric: .4f}')
    plt.subplot(1, 2, 1)
    plt.axhline(y=test_loss, color='#00a80e', linestyle=':', label='test')
    plt.subplot(1, 2, 2)
    plt.axhline(y=test_metric, color='#00a80e', linestyle=':', label='test')
    plt.legend(loc='best')
