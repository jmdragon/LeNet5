# eval_lenet1.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

import mnist
from lenet1 import LeNet5RBF  # needed so torch.load can unpickle
from prototypes_from_digit import build_digit_prototypes_from_digit


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad = torchvision.transforms.Pad(2, fill=0, padding_mode="constant")

    mnist_test = mnist.MNIST(split="test", transform=pad)
    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    # Load model (PyTorch 2.6+)
    model = torch.load("LeNet1.pth", weights_only=False)
    model.to(device)
    model.eval()

    num_classes = 10
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)  # [true, pred]

    # For "most confusing example" per true digit
    # We'll store: (max_confidence, image_tensor, true_label, pred_label)
    # Confidence defined as: penalty(second_best) - penalty(best_wrong)
    # Larger difference = more confident (wrong) prediction.
    most_confusing = {
        d: {
            "confidence": -np.inf,
            "image": None,
            "true": None,
            "pred": None,
        }
        for d in range(num_classes)
    }

    os.makedirs("most_confusing_lenet1", exist_ok=True)

    with torch.no_grad():
        for idx, (image, label) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device)

            outputs = model(image)  # (1,10), penalties
            penalties = outputs[0]  # (10,)

            pred = torch.argmin(penalties).item()
            true = label.item()

            conf_mat[true, pred] += 1

            # Track confusing examples: only if misclassified
            if pred != true:
                # Sort penalties ascending (best first)
                sorted_vals, sorted_idx = torch.sort(penalties)
                best_idx = sorted_idx[0].item()         # predicted class
                second_idx = sorted_idx[1].item()       # runner-up

                # Confidence: how much better best is vs second best
                confidence = (sorted_vals[1] - sorted_vals[0]).item()

                rec = most_confusing[true]
                if confidence > rec["confidence"]:
                    rec["confidence"] = confidence
                    rec["image"] = image.cpu().clone()
                    rec["true"] = true
                    rec["pred"] = pred

    # Print confusion matrix
    print("Confusion matrix (rows=true, cols=pred):")
    print(conf_mat)

    # Save confusion matrix
    np.savetxt("lenet1_confusion_matrix.txt", conf_mat, fmt="%d")
    print("Saved confusion matrix to lenet1_confusion_matrix.txt")

    # Save most confusing examples as PNGs
    from torchvision.utils import save_image

    for d in range(num_classes):
        rec = most_confusing[d]
        if rec["image"] is not None:
            img = rec["image"]  # (1,1,32,32)
            fname = f"most_confusing_lenet1/true_{d}_pred_{rec['pred']}.png"
            save_image(img, fname)
            print(
                f"Digit {d}: most confusing example misclassified as {rec['pred']} "
                f"(confidence={rec['confidence']:.4f}) saved to {fname}"
            )
        else:
            print(f"Digit {d}: no misclassifications found.")


if __name__ == "__main__":
    main()
