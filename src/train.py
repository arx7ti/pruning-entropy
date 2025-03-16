import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
import torchvision

from .utils import boltzmann_entropy, gaussian_entropy


def train_on_MNIST(
    model,
    trainset,
    testset,
    batch_size=512,
    n_epochs=10,
    lr=0.001,
    root='./data',
    entropy=boltzmann_entropy,
    mask=False,
    pruning=False,
    thresh=0.1,
    prune_every=200,
    reg_lambda=0.001,
    noise_std=1e-5,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    trainset = torchvision.datasets.MNIST(root=root,
                                          train=True,
                                          download=True,
                                          transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True)

    testset = torchvision.datasets.MNIST(root=root,
                                         train=False,
                                         download=True,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, val_accuracies, entropies = [], [], [], []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if pruning:
                # Lasso regularization
                reg_loss = 0.0

                for layer in get_children(model):
                    if isinstance(layer, nn.Linear):
                        reg_loss += torch.norm(layer.weight, p=1)

                loss += reg_lambda * reg_loss

            loss.backward()

            if pruning:
                # Noise injection
                for param in model.parameters():
                    if param.requires_grad and isinstance(param, nn.Linear):
                        param.grad += noise_std * torch.randn_like(param.grad)

            optimizer.step()

            running_loss += loss.item() * batch_size

        H = entropy(model, mask=mask, thresh=thresh)
        entropies.append(H)

        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * batch_size

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(trainset))
        val_losses.append(val_loss / len(testset))
        val_accuracies.append(correct / total)

        print(
            f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}, H: {entropies[-1]:.4f}"
        )

    return train_losses, val_losses, val_accuracies, entropies
