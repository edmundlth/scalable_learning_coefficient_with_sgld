import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
import os
import concurrent

from EntropySGD import EntropySGD
from experiment import MNISTExperiment
from model import MNISTNet, CustomRegressionDataset, create_gaussian_noise_regression_dataset

MNIST_CLASS_LABELS = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

def parse_commandline():
    parser = argparse.ArgumentParser(description="PyTorch Entropy-SGD")
    parser.add_argument("--hidden_layer_sizes", help="size of feedforward layers other than the input and output layer", nargs="+", type=int, default=[1024, 1024])
    parser.add_argument("--batch_size", help="Batch size", type=int, default=512)
    parser.add_argument("--num_training_data", help="Training data set size. If none, full data is used.", type=int, default=None)
    parser.add_argument("--L", help="Langevin iterations", type=int, default=5)
    parser.add_argument(
        "--seeds",
        help="Experiments are repeated for each RNG seed given",
        nargs="+",
        type=int,
        default=[0],
    )
    parser.add_argument("--epochs", help="epochs", type=int, default=10)
    parser.add_argument("--num_gradient_step", help="Total number of gradient steps taken. If specified, ignore --epoch and calculate epoch number from dataset size and batch size.", type=int, default=None)
    parser.add_argument(
        "--optimizer", help="sgd | entropy-sgd | adam", type=str, default="sgd"
    )
    parser.add_argument(
        "--dataset_name", help="mnist | 2dtanh", type=str, default="mnist"
    )
    parser.add_argument(
        "--outputdir",
        help="Path to output directory. Create if not exist.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_result",
        help="save experiment result if specified.",
        action="store_true",
    )
    parser.add_argument(
        "--save_model",
        help="save trained pytorch model if specified.",
        action="store_true",
    )
    parser.add_argument(
        "--save_plot", help="save plots to file if specified", action="store_true"
    )
    parser.add_argument("--show", help="plt.show() if specified.", action="store_true")
    parser.add_argument("--max_workers", help="Maximum number of parallel process running the experiments independently for each given rngseed", type=int, default=None)

    # parser.add_argument('-m', help='mnistfc | mnistconv | allcnn', type=str, default='mnistconv')
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--sgld_gamma', type=float, default=None, help="gamma parameter in SGLD. If not specified, it is chosen automatically by inspecting the norm of loss gradient.")
    parser.add_argument('--sgld_num_chains', type=int, default=4, help="number of independent SGLD chains for estimating local free energy.")
    parser.add_argument('--sgld_num_iter', type=int, default=100, help="number of SGLD steps.")
    parser.add_argument('--sgld_noise_std', type=float, default=1e-5, help="standard deviation of gaussian noise in SGLD.")

    parser.add_argument('--data_rootdir', type=str, default="./data", help="Directory where MNIST data is stored or download to.")
    return parser


def main(args, rngseed):
    print(f"Starting experiment for rngseed: {rngseed}")
    def _get_save_filepath(filename):
        filepath = os.path.join(args.outputdir, f"{rngseed}_{filename}")
        print(f"Filepath constructed: {filepath}")
        return filepath
    
    torch.manual_seed(rngseed)
    np.random.seed(rngseed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ########################################################################
    # Define dataset, network and loss function
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if args.dataset_name.lower() == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        trainset = torchvision.datasets.MNIST(
            root=args.data_rootdir, train=True, download=True, transform=transform
        )
        if args.num_training_data is not None and args.num_training_data < len(trainset):
            random_indices = torch.randperm(len(trainset))[:args.num_training_data]
            trainset = torch.utils.data.Subset(trainset, random_indices)
        
        testset = torchvision.datasets.MNIST(
            root=args.data_rootdir, train=False, download=True, transform=transform
        )
        net = MNISTNet(
            args.hidden_layer_sizes,
            input_dim=28 * 28, output_dim=10, 
            activation=F.relu, with_bias=True
        )
        criterion = nn.CrossEntropyLoss()

    elif args.dataset_name.lower() == "2dtanh":
        XMIN, XMAX = -2.0, 2.0
        X = torch.rand((args.num_training_data, 1)).float() * (XMAX - XMIN) + XMIN
        X_test = torch.rand((10000, 1)).float() * (XMAX - XMIN) + XMIN

        a = 2.0
        b = 2.0
        noise_std = 1 / np.sqrt(2)
        f = lambda x: a * torch.tanh(b * x).float()
        trainset = create_gaussian_noise_regression_dataset(f, X, noise_std)
        testset = create_gaussian_noise_regression_dataset(f, X_test, noise_std)
        net = MNISTNet(
            args.hidden_layer_sizes,
            input_dim=1, 
            output_dim=1, 
            activation=F.tanh, 
            with_bias=False
        )
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    n_train = len(trainloader.dataset)
    print(f"Number of training data: {n_train}")

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    n_test = len(testloader.dataset)
    print(f"Number of testing data: {n_test}")

    net.to(device)
    print(net)
    network_param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {network_param_count}")

    ########################################################################
    # Train the network
    # ^^^^^^^^^^^^^^^^^^^^

    if args.num_gradient_step is not None:
        num_step_per_epoch = n_train / trainloader.batch_size
        num_epoch = int(np.round(args.num_gradient_step / num_step_per_epoch, decimals=0))
    else:
        num_epoch = args.epochs
    print(f"Num epoch: {num_epoch}")

    if args.optimizer.lower() in ["entropy-sgd", "esgd"]:
        optimizer = EntropySGD(net.parameters(), eta=args.lr, momentum=0.9, nesterov=False, L=args.L)
        experiment = MNISTExperiment(
            net, trainloader, testloader, criterion, optimizer, device, 
            sgld_num_chains=args.sgld_num_chains, sgld_num_iter=args.sgld_num_iter, sgld_gamma=args.sgld_gamma, sgld_noise_std=args.sgld_noise_std
        )
        experiment.run_entropy_sgd(args.L, num_epoch)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args.lr, momentum=0.9, nesterov=True
        )
        experiment = MNISTExperiment(
            net, trainloader, testloader, criterion, optimizer, device, sgld_num_chains=args.sgld_num_chains, sgld_num_iter=args.sgld_num_iter, sgld_gamma=args.sgld_gamma, sgld_noise_std=args.sgld_noise_std
        )
        experiment.run_sgd(num_epoch)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            net.parameters(), lr=args.lr
        )
        experiment = MNISTExperiment(
            net, trainloader, testloader, criterion, optimizer, device, sgld_num_chains=args.sgld_num_chains, sgld_num_iter=args.sgld_num_iter, sgld_gamma=args.sgld_gamma, sgld_noise_std=args.sgld_noise_std
        )
        experiment.run_sgd(num_epoch)
        
    print("Finished Training")
    # _map_float = lambda x: list(map(float, x))
    result = {
        "rngseed": rngseed,
        "network_param_count": network_param_count,
        "layers": args.hidden_layer_sizes,
        "optimizer_type": args.optimizer,
        "sgld_gamma": args.sgld_gamma, 
        "sgld_num_iter": args.sgld_num_iter, 
        "sgld_num_chains": args.sgld_num_chains, 
        "sgld_noise_std": args.sgld_noise_std,
        "num_epoch": num_epoch,
        "batch_size": args.batch_size,
        "n_train": n_train, 
        "n_test": n_test,
        "lr": args.lr, 
    }
    result.update(experiment.records)
    print(json.dumps(result, indent=2))
    if args.save_result:
        outfilepath = _get_save_filepath("result.json")
        print(f"Saving result at: {outfilepath}")
        with open(outfilepath, "w") as outfile:
            json.dump(result, outfile, indent=2)

    ########################################################################
    # Plotting
    # ^^^^^^^^^^^^^^^^^^^^
    l_epochs = list(range(num_epoch))
    lfes = experiment.records["lfe"]
    energies = experiment.records["energy"]
    hatlambdas = experiment.records["hatlambda"]
    test_errors = experiment.records["test_error"]
    train_errors = experiment.records["train_error"]

    
    print("Generating plots...")
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 12))
    ax = axes[0]
    ax.plot(l_epochs, lfes, "x--", label="local free energy")
    ax.plot(l_epochs, energies, "x--", label="$nL_n(w_t)$")
    ax.set_xlabel("epoch")
    ax.legend()

    ax = axes[1]
    ax.plot(
        l_epochs,
        np.array(hatlambdas) * np.log(n_train),
        "x--",
        color="green",
        label="$\lambda(w_t) \log n$",
    )
    ax.set_xlabel("epoch")
    ax.legend()
    plt.suptitle(f"{args.optimizer} final hatlambda {hatlambdas[-1]}")

    ax = axes[2]
    ax.plot(l_epochs, test_errors, "kx--", label="test")
    ax.plot(l_epochs, train_errors, "kx--", label="train")
    ax.set_xlabel("epoch")
    ax.set_ylabel("percent error")
    ax.legend()

    print("Saving plots....")
    if args.save_plot:
        fig.savefig(_get_save_filepath("plots.png"))
    if args.show:
        plt.show()

    ########################################################################
    # Let's quickly save our trained model:
    if args.save_model:
        torch.save(net.state_dict(), _get_save_filepath("model.pth"))
        for model_i, model_copy in enumerate(experiment.snapshot_models):
            torch.save(model_copy.state_dict(), _get_save_filepath(f"model_{model_i}.pth"))

    return result


if __name__ == "__main__":
    args = parse_commandline().parse_args()
    print(f"Commandline arguments:\n{vars(args)}")
    if args.outputdir:
        os.makedirs(args.outputdir, exist_ok=True)
        filepath = os.path.join(args.outputdir, "commandline_args.json")
        with open(filepath, 'w') as cmdfile:
            json.dump(vars(args), cmdfile, indent=2)
    
    # use parallel workers on given list of rngseeds
    if args.max_workers is not None and args.max_workers > 1 and len(args.seeds) > 1: 
        input1 = [args for _ in range(len(args.seeds))]
        input2 = args.seeds
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            results = list(executor.map(main, input1, input2))
    else:
        for i, rngseed in enumerate(args.seeds):
            print(f"Running seed {i + 1} / {len(args.seeds)} with value: {rngseed}")
            main(args, rngseed)
            print(f"Finished")
