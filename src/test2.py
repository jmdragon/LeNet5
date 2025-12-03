from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mnist
import torch
import numpy as np
import torchvision

 
def test(dataloader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sum_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)                
            preds = torch.argmax(logits, dim=1)    

            sum_correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = sum_correct / total
    print("test accuracy:", accuracy)

 

def main():

    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')

    mnist_test=mnist.MNIST(split="test",transform=pad)

    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)

    model = torch.load("models/LeNet2.pth", weights_only=False)

    test(test_dataloader,model)

 

if __name__=="__main__":

    main()
