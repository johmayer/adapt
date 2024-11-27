import os
import zipfile
import torch
import timeit
import requests
import multiprocessing
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn as nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib

import adapt
from models.resnet import resnet34
from adapt.references.classification.train import evaluate, train_one_epoch, load_data

def collect_stats(model, data_loader, num_batches):
     """Feed data to the network and collect statistic"""

     # Enable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.disable_quant()
                 module.enable_calib()
             else:
                 module.disable()

     for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
         model(image.cpu())
         if i >= num_batches:
             break

     # Disable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.enable_quant()
                 module.disable_calib()
             else:
                 module.enable()

def compute_amax(model, **kwargs):
 # Load calib result
 for name, module in model.named_modules():
     if isinstance(module, quant_nn.TensorQuantizer):
         if module._calibrator is not None:
             if isinstance(module._calibrator, calib.MaxCalibrator):
                 module.load_calib_amax()
             else:
                 module.load_calib_amax(**kwargs)
         print(F"{name:40}: {module}")
 model.cpu()

# function to execute calibration of the model
def calibrate_model(model, data_t):
    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        stats = collect_stats(model, data_t, num_batches=2)
        amax = compute_amax(model, method="percentile", percentile=99.99)

# function to load the data
def val_dataloader(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)):

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    dataset = CIFAR10(root="datasets/cifar10_data", train=False, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )
    return dataloader

# function to evaluate the model performance
def eval(model, data):
    correct = 0
    total = 0

    model.eval()
    start_time = timeit.default_timer()
    with torch.no_grad():
        for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
            images, labels = images.to("cpu"), labels.to("cpu")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(timeit.default_timer() - start_time)
    print('Accuracy of the network on the 10000 test images: %.4f %%' % (
        100 * correct / total))
    return correct / total

def main():

    # Determine the number of CPU cores
    num_cores = multiprocessing.cpu_count()

    threads = num_cores * 2 # two threads are available on github codespaces per core
    torch.set_num_threads(threads)

    # maybe better performance
    # Set environment variables for performance tuning
    os.environ["OMP_PLACES"] = "cores"
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_WAIT_POLICY"] = "active"

    # set approximate multiplier
    axx_mult = 'SPR12_44'

    # laod model for evaluation
    model = resnet34(pretrained=True, axx_mult = axx_mult)
    model.eval() # for evaluation

    transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)),
            ]
        )
    dataset = CIFAR10(root="datasets/cifar10_data", train=True, download=True, transform=transform)

    evens = list(range(0, len(dataset), 10))
    trainset_1 = torch.utils.data.Subset(dataset, evens)

    data = val_dataloader()

    # data_t is used for calibration purposes and is a subset of train-set
    data_t = DataLoader(trainset_1, batch_size=128,
                                                shuffle=False, num_workers=0)

    # calibrate model
    calibrate_model(model, data_t)

    # evaluate the model
    # initial_accuracy = eval(model, data)

    # finetune the model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    EPOCHS = 15

    # load model if necessary
    load_epoch = 6
    if load_epoch > 0:
        model.load_state_dict(torch.load(f"./saved/retrained_model_epoch_{load_epoch}.pth"))
        calibrate_model(model, data_t)

    # finetune the model for one epoch based on data_t subset
    for epoch in range(EPOCHS):
        EPOCH_NUM = epoch + 1
        if EPOCH_NUM > load_epoch:
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            train_one_epoch(model, criterion, optimizer, data_t, "cpu", EPOCH_NUM, 1)
            calibrate_model(model, data_t)

            # Save the model after each epoch
            model_path = f"./saved/retrained_model_epoch_{EPOCH_NUM}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path} after epoch {EPOCH_NUM}")
        else:
            print(f"Skipping epoch {EPOCH_NUM} as pretrained model exists")

        
    # evaluate model after each epoch
    final_accuracy = eval(model, data)

    # print accuracy
    # print('\n\nAccuracy before retraining: %.4f %%' % (100 * initial_accuracy))
    print('\n\nAccuracy after retraining: %.4f %%' % (100 * final_accuracy))


if __name__ == "__main__":
    main()