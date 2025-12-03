from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mnist
import torch
import numpy as np
import torchvision

 
def test(dataloader, model):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sum_correct = 0   # <-- IMPORTANT: initialize before the loop
    total = 0

    with torch.no_grad():
        for batch, (image, label) in enumerate(dataloader):
            image = image.to(device)
            label = label.to(device)

            # model outputs RBF penalties: smaller = better
            outputs = model(image)                # shape (1, 10)
            preds = torch.argmin(outputs, dim=1)  # shape (1,)

            sum_correct += (preds == label).sum().item()
            total += label.size(0)

    test_accuracy = sum_correct / total
    print("test accuracy:", test_accuracy)

 

def main():

    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')

    mnist_test=mnist.MNIST(split="test",transform=pad)

    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)

    model = torch.load("models/LeNet1.pth", weights_only=False)

    test(test_dataloader,model)

 

if __name__=="__main__":

    main()
