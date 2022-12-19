from data import augumentation, get_data_loaders
from training import train, test
from model import ResNet18, DenseNet_
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from comet_ml import Experiment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--num_epoches', default=50, type=float, help='number of epoches')
parser.add_argument(
    "--decompose_mode",
    default="None",
    choices=["None", "CP", "Tucker"],
    help="choose the type of decomposition for CNN",
)


def set_up_exp():
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="Dcj59uwP496tLKvrjrWEU79R0",
        project_name="cp-tucker-decomposition",
        workspace="highly0",
    )
    return experiment


if __name__ == "__main__":
    args = parser.parse_args()

    # experiment for plotting
    experiment = set_up_exp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, val_transform = augumentation()
    train_loader, val_loader = get_data_loaders(
        train_transform, val_transform, batch_size=128
    )

    # models and hyper params
    torch.manual_seed(96)
    model = DenseNet_()
    model_name = str(model.__class__.__name__)
    model = model.to(device)
    if device == "cuda:4":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    exp_name = f"{model_name}_{args.decompose_mode}_{args.num_epoches}"
    experiment.set_name(exp_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train(
        experiment,
        exp_name,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        n_epochs=args.num_epoches,
    )
