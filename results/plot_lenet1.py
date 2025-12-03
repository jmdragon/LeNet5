import numpy as np
import matplotlib.pyplot as plt

data = np.load("lenet1_errors.npz")
train_err = data["train_err"]
test_err = data["test_err"]

epochs = np.arange(1, len(train_err) + 1)

plt.figure()
plt.plot(epochs, train_err, label="Train error")
plt.plot(epochs, test_err, label="Test error")
plt.xlabel("Epoch")
plt.ylabel("Error rate")
plt.title("LeNet-5: training and test error vs epoch")
plt.legend()
plt.grid(True)
plt.show()
