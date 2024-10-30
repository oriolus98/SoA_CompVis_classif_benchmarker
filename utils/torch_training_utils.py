import torch
import torch.nn as nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class TopLayers(nn.Module):
    def __init__(self, input_size = 576, num_classes = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.swish = nn.Hardswish()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return(x)
    

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    total_loss, correct = 0, 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item() * X.size(0) 
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 30 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    avg_train_loss = total_loss / size
    train_accuracy = correct / size
    return avg_train_loss, train_accuracy


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct