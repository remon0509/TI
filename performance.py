import numpy as np
import matplotlib.pyplot as plt

def multiclass_accuracy(pred, true, n_classes):
  pred = np.argmax(pred, axis=1)
  true = np.argmax(true, axis=1)
  acc = []

  for label in range(n_classes):
    result = [i for i, j in zip(pred, true) if j == label]
    acc.append(result.count(label)/len(result))
  return acc


def plot_confusion_matrix(cm, target_names, title='Confusion Matrix'):
    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=10)
        plt.yticks(tick_marks, target_names, fontsize=10)
    
    thresh = cm.max()/2
    idx = [[i, j] for i in range(cm.shape[0]) for j in range(cm.shape[1])]
    
    for i in idx:
        plt.text(i[1], i[0], cm[i[0], i[1]],
                 horizontalalignment = 'center',
                 color='white' if cm[i[0], i[1]]>thresh else 'black',
                 fontsize=15)
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
