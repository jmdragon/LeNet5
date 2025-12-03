# train_lenet1.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

import mnist
from lenet1 import LeNet5RBF
from prototypes_from_digit import build_digit_prototypes_from_digit


def map_loss(outputs, labels, j=0.1):
    """
    MAP-style loss from Eq. (9) in the LeNet-5 paper.

    outputs: (B,10) penalties y_i(Z,W)
    labels:  (B,) correct class indices
    """
    B, num_classes = outputs.shape

    # y_D: penalty of the correct class
    y_correct = outputs[torch.arange(B), labels]  # (B,)

    # penalties of incorrect classes
    mask = torch.ones_like(outputs, dtype=torch.bool)
    mask[torch.arange(B), labels] = False
    y_incorrect = outputs[mask].view(B, num_classes - 1)  # (B, 9)

    # log( e^{-j} + sum_{i != D} e^{-y_i} )
    j_tensor = torch.tensor(j, device=outputs.device)
    log_term = torch.log(torch.exp(-j_tensor) + torch.sum(torch.exp(-y_incorrect), dim=1))

    loss = (y_correct + log_term).mean()
    return loss


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 28x28 -> 32x32 padding (right + bottom, 2 pixels each)
    pad = torchvision.transforms.Pad(2, fill=0, padding_mode="constant")

    mnist_train = mnist.MNIST(split="train", transform=pad)
    mnist_test = mnist.MNIST(split="test", transform=pad)

    train_loader = DataLoader(mnist_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    # --- Build RBF prototypes from DIGIT ---
    print("Building RBF prototypes from DIGIT data...")
    prototypes = build_digit_prototypes_from_digit()  # (10,84)
    print("Prototypes shape:", prototypes.shape)

    model = LeNet5RBF(prototypes).to(device)

    lr = 0.001  # steepest gradient method with c ≈ 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    num_epochs = 20
    train_err_history = []
    test_err_history = []

    for epoch in range(1, num_epochs + 1):
        # ---------- Train ----------
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)               # (1,10)
            loss = map_loss(outputs, labels, j=0.1)
            loss.backward()
            optimizer.step()

            # prediction = argmin penalty
            preds = torch.argmin(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_err = 1.0 - train_correct / train_total

        # ---------- Evaluate on test set ----------
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = torch.argmin(outputs, dim=1)

                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_err = 1.0 - test_correct / test_total

        train_err_history.append(train_err)
        test_err_history.append(test_err)

        print(
            f"Epoch {epoch:02d}: "
            f"train_err={train_err:.4f}, test_err={test_err:.4f}"
        )

    # Save for plotting later
    np.savez(
        "results/lenet1_errors.npz",
        train_err=np.array(train_err_history),
        test_err=np.array(test_err_history),
    )
    print("Saved error curves to lenet1_errors.npz")

    # Save model for test1.py
    torch.save(model, "models/LeNet1.pth")
    print("Saved trained model to LeNet1.pth")


if __name__ == "__main__":
    main()
