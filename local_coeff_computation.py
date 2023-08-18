import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import json

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from mnist_expt import MNISTNet
from experiment import MNISTExperiment


# Define the experiment
EXPT_NAME = "mnist_local_coeff_expt"
ex = Experiment(EXPT_NAME)
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Add a MongoDB observer
ex.observers.append(
    MongoObserver.create(
        url="mongodb://localhost:27017/", db_name=EXPT_NAME
    )
)

@ex.config
def load_config():
    data_rootdir = "./data"
    num_training_data = 60000
    hidden_layer_sizes = (1024, 1024)
    batch_size = 512
    lr = 0.01
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_filepath = "./outputs/expt20230809_60000_512_t200_lr0.01_SGLDITER50_GAMMA100_sgd/701_model.pth"  # Default path to model
    config_filepath = "./outputs/expt20230809_60000_512_t200_lr0.01_SGLDITER50_GAMMA100_sgd/commandline_args.json"  # Default path to config
    sgld_num_chains = 4
    sgld_gamma = 100.0
    sgld_num_iter = 50
    sgld_noise_std = 1e-5
    


# Main function for training continuation
@ex.automain
def compute_local_learning_coefficient(
    _run,
    model_filepath,
    config_filepath, 
    data_rootdir,
    num_training_data,
    hidden_layer_sizes,
    batch_size,
    lr,
    sgld_num_chains,
    sgld_gamma,
    sgld_num_iter,
    sgld_noise_std,
    device,
    seed,
):
    device = torch.device(device)
    # Load the saved model
    net = MNISTNet(hidden_layer_sizes=hidden_layer_sizes)
    net.load_state_dict(torch.load(model_filepath, map_location=device))
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = torchvision.datasets.MNIST(
        root=data_rootdir, train=True, download=True, transform=transform
    )
    if num_training_data is not None and num_training_data < len(trainset):
        random_indices = torch.randperm(len(trainset))[:num_training_data]
        trainset = torch.utils.data.Subset(trainset, random_indices)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    n_train = len(trainloader.dataset)
    print(f"Number of training data: {n_train}")

    testset = torchvision.datasets.MNIST(
        root=data_rootdir, train=False, download=True, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    n_test = len(testloader.dataset)
    print(f"Number of testing data: {n_test}")

    net.to(device)
    print(net)
    network_param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {network_param_count}")

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    experiment = MNISTExperiment(
        net, 
        trainloader=trainloader, 
        testloader=testloader, 
        criterion=criterion,
        optimizer=None, 
        device=device, 
        sgld_num_chains=sgld_num_chains, 
        sgld_gamma=sgld_gamma, 
        sgld_num_iter=sgld_num_iter, 
        sgld_noise_std=sgld_noise_std, 
    )
    local_free_energy, energy, lambdahat, lfe_chain_std, func_var  = experiment.compute_fenergy_energy_rlct()
    train_accuracy = experiment.eval(trainloader)
    test_accuracy = experiment.eval(testloader)

    train_loss = energy / n_train
    with torch.no_grad():
        losses = []
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss * inputs.shape[0])
        test_loss = np.mean(losses)

    nuhat = func_var / 2 * np.log(num_training_data)
    print(lambdahat, train_loss, test_loss)
    _run.info["local_free_energy"] = float(local_free_energy)
    _run.info["lfe_chain_std"] = float(lfe_chain_std)
    _run.info["lambdahat"] = float(lambdahat)
    _run.info["func_var_beta"] = float(func_var)
    _run.info["nuhat"] = float(nuhat)
    _run.info["gengaphat"] = float(nuhat / num_training_data)
    _run.info["train_error"] = float(1 - train_accuracy)
    _run.info["test_error"] = float(1 - test_accuracy)
    _run.info["train_loss"] = float(train_loss)
    _run.info["test_loss"] = float(test_loss)

    with open(config_filepath) as config_file:
        training_config = json.load(config_file)
    _run.info["training_config"] = training_config
