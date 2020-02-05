# c'est le travail dont je t'ai parlé, c'est juste pour comprendre un peu mieux le fonctionnement des réseaux de neurones ;)

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
batch_size = 32

train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True, transform=transform), batch_size=batch_size, shuffle=True)


weights = torch.randn(784, 10, requires_grad=True)


# fonction pour voir le pourcentage de fiabilité de notre réseau de neurones
def test(weights, test_loader):
    test_size = len(test_loader.dataset)
    correct = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.view((-1, 28*28))
        outputs = torch.matmul(data, weights)
        softmax = F.softmax(outputs, dim=1)
        pred = softmax.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    acc = correct / test_size
    print(" fiabilité : {}%".format(acc*100))
    return

#test(weights, test_loader)

# cette commande permet de lancer l'apprentissage de notre réseau de neurones
# >>> apprentissage(weights)

def apprentissage(weights):
    it = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        if weights.grad is not None:
            weights.grad.zero_()
            
        data = data.view((-1, 28*28))

        outputs = torch.matmul(data, weights)
        
        log_softmax = F.log_softmax(outputs, dim=1)
        
        loss = F.nll_loss(log_softmax, targets)


        print("Loss shape: {}\n".format(loss), end="")
        
        loss.backward()

        with torch.no_grad():
            weights -= 0.1*weights.grad

        it += 1
        if it % 100 == 0:
            test(weights, test_loader)

        if it > 10000:
            break

# permet de tester notre réseau de neurones sur des images contenant des chiffres
def prediction():
    batch_idx, (data, target) = next(enumerate(test_loader))
    data = data.view((-1,  28*28))

    outputs = torch.matmul(data, weights)
    softmax = F.softmax(outputs, dim=1)
    pred = softmax.argmax(dim=1, keepdim=True)

    plt.imshow(data[0].view(28,28), cmap="gray")
    plt.title("predicted class {}".format(pred[0]))
    plt.show()

test(weights, test_loader)
