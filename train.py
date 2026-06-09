import torch
import torch.nn as nn
from torchvision import datasets, transforms, models


def resnet18():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)
    )

    return model


if __name__ == '__main__':
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2022, 0.1994, 0.2010])
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2022, 0.1994, 0.2010])
    ])
    trainset = datasets.CIFAR10(
        root='.data', train=True, download=True, transform=train_transforms
    )
    testset = datasets.CIFAR10(
        root='.data', train=False, download=True, transform=test_transforms
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=4
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    log_file = open('train-log.txt','w',encoding='utf-8')
    best_acc = 0
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(100 * correct / total)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        test_losses.append(test_loss / len(test_loader))
        test_accs.append(100 * correct / total)

        print(f'Epoch {epoch+1:02d}/{epochs}  '
              f'Train Loss {train_losses[-1]:.4f}, Acc {train_accs[-1]:.2f}% '
              f'Test Loss {test_losses[-1]:.4f}, Acc {test_accs[-1]:.2f}%')

        log_file.write(f'Epoch {epoch+1:02d}/{epochs}  '
              f'Train Loss {train_losses[-1]:.4f}, Acc {train_accs[-1]:.2f}% '
              f'Test Loss {test_losses[-1]:.4f}, Acc {test_accs[-1]:.2f}%\n')
        scheduler.step()

        if test_accs[-1] > best_acc:
            best_acc = test_accs[-1]
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    torch.save(model.state_dict(), 'resnet18_cifar10.pth')
    print('训练完成！')
    