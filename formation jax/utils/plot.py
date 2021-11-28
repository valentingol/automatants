import matplotlib.pyplot as plt

def plt_curves(train_losses, train_metrics, val_losses, val_metrics):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    x = range(len(train_losses))
    plt.title('Loss')
    plt.plot(x, train_losses, 'r--', label='train')
    plt.plot(x, val_losses, 'r', label='val')
    plt.ylim(0, 2.4)
    plt.legend(loc='best')
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(x, train_metrics, 'b--', label='train')
    plt.plot(x, val_metrics, 'b', label='val')
    plt.ylim(0, 1)
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
