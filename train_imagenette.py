import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from outdated_van import VANv2
import torch.optim.lr_scheduler as lr_scheduler
import random
from torch.nn import DataParallel
import argparse

torch.manual_seed(1)
random.seed(1)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

batch_size = 64

parser = argparse.ArgumentParser(
    prog="train yay",
    description="VAN tester among us",
)

parser.add_argument('--token_improved', type=int, metavar='N', default=0)
parser.add_argument('--nopw', type=int, metavar='N', default=1)
parser.add_argument('--scale', type=int, metavar='N', default=0)
parser.add_argument('--mlp', type=int, metavar='N', default=1)
parser.add_argument('--name', type=str, default="VANv2")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--size', type=str, default='t')

args = parser.parse_args()
name = args.name+"_"+args.size

all_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

train_set = datasets.ImageFolder("data/imagewoof2/train", transform=train_transform)
testval_set = datasets.ImageFolder("data/imagewoof2/val", transform=all_transform)

val_set, test_set = random_split(testval_set, [0.67, 0.33])


train_loader = DataLoader(train_set, batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

model = VANv2(nopw=args.nopw, mlp=args.mlp, token_improved=args.token_improved, scale=args.scale, act=0, size=args.size, img_size=256, num_classes=10)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')

model_optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999))
model_scheduler = lr_scheduler.ReduceLROnPlateau(model_optimizer, patience=5, threshold=0.01)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_epoch(model, optimizer, criterion, name, epoch, scheduler=None):
    model.train()
    # total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        model.zero_grad()
        loss.backward()
        optimizer.step()
    validate_score = validate(model, criterion)
    if scheduler is not None:
        scheduler.step(validate_score)
    if epoch == args.epochs or epoch == args.epochs-1:
        torch.save(model.state_dict(), f"{name}.pt")
    return validate_score

def validate(model, criterion):
    model.eval()
    with torch.no_grad():
        loss = 0
        num = 0
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss += criterion(output, labels).item()
            num+=1
            
    return loss / num

def train(n_epochs):
    model_losses = {}
    early_stopper = EarlyStopper(patience=10, min_delta=0.02)

    for epoch in range(n_epochs):
        print(f"Epoch: {epoch+1}/{n_epochs}, ", end="")
        if model_scheduler:
            model_loss = train_epoch(model, model_optimizer, criterion, f"{name}_epoch{epoch+1}", epoch+1, model_scheduler)
        else:
            model_loss = train_epoch(model, model_optimizer, criterion, f"{name}_epoch{epoch+1}", epoch+1)

        model_losses[epoch+1] = model_loss
        if early_stopper.early_stop(model_loss):          
            torch.save(model.state_dict(), f"{name}_epoch{epoch+1}.pt")   
            break

        print(f"{name} Average Loss: {model_loss}")

train(args.epochs)


def test(model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, output = torch.max(output, dim=1)
            total += labels.size(0)
            correct += (output == labels).sum().item()

    return correct / float(total)

model_accuracy = test(model)
print(f"{name} Test: {model_accuracy}")
