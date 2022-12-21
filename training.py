import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
from datetime import timedelta


def epoch_train(loader, model, criterion, opt, device):
    """
    training per epoch
    """
    model.train(True)
    model.eval()

    total_loss = 0.0
    correct = 0

    for data in loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        total_loss += loss.item() * loader.batch_size

        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_accuracy = correct / len(loader.dataset)

    return avg_loss, avg_accuracy


def epoch_val(loader, model, criterion, device):
    """
    validating per epoch
    """
    model.train(True)
    model.eval()
    total_loss = 0.0
    correct = 0

    for data in loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * loader.batch_size

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_accuracy = correct / len(loader.dataset)

    return avg_loss, avg_accuracy


def log_experiment(experiment, epoch, train_loss, train_acc, test_loss, test_acc):
    experiment.log_metric("Train Loss", train_loss, step=epoch)
    experiment.log_metric("Train Accuracy", train_acc, step=epoch)
    experiment.log_metric("Val Loss", test_loss, step=epoch)
    experiment.log_metric("Test Accuracy", test_acc, step=epoch)


def train(
    experiment,
    checkpoint_name,
    train_loader,
    test_loader,
    model,
    criterion,
    opt,
    scheduler,
    device,
    n_epochs=50,
    checkpoint_path="./checkpoints/",
):
    """
    training loop
    """
    for epoch in tqdm(range(n_epochs)):
        train_loss, train_acc = epoch_train(train_loader, model, criterion, opt, device)
        test_loss, test_acc = epoch_val(test_loader, model, criterion, device)

        print(
            f"[Epoch {epoch + 1}] train loss: {train_loss:.3f}; train acc: {train_acc:.2f}; "
            + f"test loss: {test_loss:.3f}; test acc: {test_acc:.2f}"
        )
        log_experiment(experiment, epoch, train_loss, train_acc, test_loss, test_acc)
        scheduler.step()

        # saving every 5 epoches
        if epoch % 5 == 0:
            PATH = checkpoint_path
            PATH += checkpoint_name
            torch.save(model.state_dict(), PATH)


def evaluate(testloader, model, device):
    """
    evaluate loop
    """
    correct = 0
    total = 0
    start = time.time()
    total_time = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            t0 = time.time()
            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = model(images)
            t1 = time.time()
            total_time = total_time + (t1 - t0)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    time_elapse = time.time() - start
    print("CPU prediction time", float(total_time) / (i + 1), i + 1)
    print("inference time:", str(timedelta(seconds=time_elapse)))
    print(
        f"Accuracy of the network on the test images: {100 * correct / len(testloader.dataset)} %"
    )
