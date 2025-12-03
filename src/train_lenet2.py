# train_lenet2.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import mnist
from lenet2 import LeNet2


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Transforms 
    # Always pad MNIST 28x28 -> 32x32 to match LeNet
    pad = torchvision.transforms.Pad(2, fill=0, padding_mode="constant")

    # Training: pad + random affine for robustness
    train_transform = transforms.Compose([
        pad,
        transforms.RandomAffine(
            degrees=30,              # random rotation in [-30, 30]
            translate=(0.1, 0.1),    # up to 10% width/height shift
            scale=(0.8, 1.2)         # random scaling
        )
    ])

    # Test: only pad (no augmentation)
    test_transform = pad

    mnist_train = mnist.MNIST(split="train", transform=train_transform)
    mnist_test = mnist.MNIST(split="test", transform=test_transform)

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)

    model = LeNet2().to(device)

    # Loss + optimizer (standard cross-entropy classification)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 20
    train_err_history = []
    test_err_history = []

    for epoch in range(1, num_epochs + 1):
        # --------- Train ---------
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)           # (B,10)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_err = 1.0 - train_correct / train_total

        # --------- Evaluate ---------
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                preds = torch.argmax(logits, dim=1)

                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_err = 1.0 - test_correct / test_total

        train_err_history.append(train_err)
        test_err_history.append(test_err)

        print(
            f"Epoch {epoch:02d}: "
            f"train_err={train_err:.4f}, test_err={test_err:.4f}"
        )

    # Save curves for your report if you want to compare with LeNet1
    np.savez(
        "results/lenet2_errors.npz",
        train_err=np.array(train_err_history),
        test_err=np.array(test_err_history),
    )
    print("Saved error curves to lenet2_errors.npz")

    # Save model for grading
    torch.save(model, "models/LeNet2.pth")
    print("Saved trained model to LeNet2.pth")


if __name__ == "__main__":
    main()
