import numpy as np

def get_loss(out, target):
    # shape: 512*7*7
    loss = np.sum(np.sum(np.abs(out - target), axis = 1), axis = 1)
    return loss

a = np.array([[[1,1,1],[2,2,2],[3,3,3]], [[1,1,1],[2,2,2],[3,3,3]]])
b = np.zeros((2,3,3))
print(get_loss(a, b))