import numpy as np

def multiclass_accuracy(pred, true, n_classes):
  pred = np.argmax(pred, axis=1)
  true = np.argmax(true, axis=1)
  acc = []

  for label in range(n_classes):
    result = [i for i, j in zip(pred, true) if j == label]
    acc.append(result.count(label)/len(result))
  return acc
