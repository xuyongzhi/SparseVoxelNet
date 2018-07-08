import numpy as np
import matplotlib.pyplot as plt

def label_weight1():
  label_1normed = np.arange(0,1,0.01)
  weights = {}
  for k in [1.0, 1.01, 1.02, 1.5]:
    weights[k] = 1/np.log(k + label_1normed)

  for k in weights:
    plt.plot(label_1normed, weights[k], label=str(k))

  plt.legend()
  plt.show()


if __name__ == '__main__':
  label_weight1()

