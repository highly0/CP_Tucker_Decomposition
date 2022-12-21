from data import augumentation, get_data_loaders
from training import train, evaluate
import models.densenet as densenet_class
from models.densenet import DenseNet_
from models.resnet import ResNet18
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from decompose import decompose_layer
from comet_ml import Experiment
import argparse
from torchstat import stat

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--num_epoches", default=50, type=float, help="number of epoches")
parser.add_argument(
    "--mode",
    default="train",
    choices=["decompose", "train", "evaluate_none", "evaluate_decomposed"],
    help="choose to decompose, train, or evaluate (normal evaluate without decomposition or with decomposition)",
)
parser.add_argument(
    "--decompose_mode",
    default="Tucker",
    choices=["CP", "Tucker"],
    help="choose the type of decomposition for CNN",
)
parser.add_argument(
    "--cnn_type",
    default="Densenet",
    choices=["Resnet18", "Densenet"],
    help="choose the type of CNN",
)


def set_up_exp():
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="Dcj59uwP496tLKvrjrWEU79R0",
        project_name="cp-tucker-decomposition",
        workspace="highly0",
    )
    return experiment


def custom_print(s):
    # for writing model stat to text file
    with open(f"./results/{exp_type}", "w+") as f:
        print(s, file=f)


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, val_transform = augumentation()
    train_loader, val_loader = get_data_loaders(
        train_transform, val_transform, batch_size=128
    )
    # models and hyper params
    torch.manual_seed(96)
    if args.cnn_type == "Resnet18":
        model = ResNet18()
    else:
        model = DenseNet_()
    model_name = str(model.__class__.__name__)
    model = model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # decompose mode
    if args.mode == "decompose":
        if args.cnn_type == "Resnet18":
            PATH = f"./checkpoints/ResNet_None_{args.num_epoches}"
            model.load_state_dict(torch.load(PATH))
            model = model.to(device)
            model.eval()
            model.cpu()

            for n, m in model.named_children():
                num_children = sum(1 for i in m.children())
                if num_children != 0:
                    # in a layer of resnet
                    layer = getattr(model, n)
                    # decomp every bottleneck
                    for i in range(num_children):
                        bottleneck = layer[i]
                        conv2 = getattr(bottleneck, "conv2")

                        # decompose current conv2d layer with CP/Tucker
                        new_layer = decompose_layer(args.decompose_mode, conv2)
                        # set old layer to new and delete
                        setattr(bottleneck, "conv2", nn.Sequential(*new_layer))
                        del conv2
                        del bottleneck
                    del layer
                torch.save(
                    model,
                    f"./checkpoints/ResNet_{args.decompose_mode}_{args.num_epoches}",
                )
            print(f"finished {args.decompose_mode} decomposition")
            stat(model, (3, 32, 32))
        else:
            PATH = f"./checkpoints/DenseNet_None_{args.num_epoches}"
            model.load_state_dict(torch.load(PATH))
            model = model.to(device)
            model.eval()
            model.cpu()

            for n, m in model.named_children():
                num_children = sum(1 for i in m.children())
                if num_children != 0:
                    # in a layer of resnet
                    layer = getattr(model, n)

                    # decomp every transition
                    if isinstance(layer, densenet_class.Transition):
                        conv2 = getattr(layer, "conv")
                        # decompose current conv2d layer with CP/Tucker
                        new_layer = decompose_layer(args.decompose_mode, conv2)
                        # set old layer to new and delete
                        setattr(layer, "conv2", nn.Sequential(*new_layer))
                        del conv2
                        del layer
                    else:
                        # decomp every bottleneck
                        for i in range(num_children):
                            bottleneck = layer[i]
                            conv2 = getattr(bottleneck, "conv2")
                            # decompose current conv2d layer with CP/Tucker
                            new_layer = decompose_layer(args.decompose_mode, conv2)
                            # set old layer to new and delete
                            setattr(bottleneck, "conv2", nn.Sequential(*new_layer))
                            del conv2
                            del bottleneck
                        del layer
                torch.save(
                    model,
                    f"./checkpoints/DenseNet_{args.decompose_mode}_{args.num_epoches}",
                )
            print(f"finished {args.decompose_mode} decomposition")
            stat(model, (3, 32, 32))
    elif args.mode == "train":  # training
        # experiment for plotting
        experiment = set_up_exp()
        exp_name = f"{model_name}_None_{args.num_epoches}"
        experiment.set_name(exp_name)

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
    else:  # evaluate
        if args.mode == "evaluate_none":
            exp_type = f"{model_name}_None_{args.num_epoches}"
            PATH = f"./checkpoints/{exp_type}"
            model.load_state_dict(torch.load(PATH))
            model = model.to("cpu")
        else:
            exp_type = f"{model_name}_{args.decompose_mode}_{args.num_epoches}"
            PATH = f"./checkpoints/{exp_type}"
            model = torch.load(PATH)

        # calculate flops, mAdds, memory
        stat(model, (3, 32, 32))
        model = model.to(device)
        if device == "cuda":
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        evaluate(val_loader, model, device)
