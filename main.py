from data import augumentation, get_data_loaders
from training import train, test
from model import get_model, ResNet18
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

if __name__ == "__main__":
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    
    train_transform, val_transform = augumentation()
    train_loader, val_loader = get_data_loaders(train_transform, val_transform, batch_size=128)
    
    # models and hyper params
    torch.manual_seed(96)
    lr = 0.1
    model = ResNet18()
    model = model.to(device)
    if device == 'cuda:4':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train(train_loader, val_loader, 
        model, criterion, optimizer, 
        scheduler, device)