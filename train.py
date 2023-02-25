import argparse
import numpy as np
from comet_ml import Experiment

import torch
import torchmetrics
from torch.utils.data import DataLoader

from utils.utils import set_seed, split_data
from utils.dataloader import HeroNameDataset
from model.model import HeroModel


device = "cuda" if torch.cuda.is_available() else "cpu"

# Define a new experiment 
experiment = Experiment(project_name="Hero_Name_Recognition", api_key='tFwkmJwXOnq1iKgGI7i2eMKlr')

def train(model,
          criterion,
          optimizer,
          train_dataloader,
          valid_dataloader,
          save_path,
          train_epoch=20):
    
    valid_loss_min = np.Inf
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=64).to(device)

    for epoch in range(0, train_epoch):
        train_loss = []
        train_acc = []
        valid_loss = []
        valid_acc = []

        # Training
        model.train()
        for data in train_dataloader:
            img, label = data
            img, label = img.to(device), label.to(device)
            
            output = model(img)

            # Loss computation
            loss = criterion(output, label)

            # Backpropagation of gradients
            optimizer.zero_grad()                                                                                                         
            loss.backward()
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in a batch
            train_loss.append(loss.item())

            # Calculate accuracy by torchmetrics
            train_acc.append(accuracy(output, label.to(torch.int8)))

        # Evaluate
        with torch.no_grad():
            model.eval()
            for data in valid_dataloader:
                img, label = data
                img, label = img.to(device), label.to(device)

                output = model(img)

                # Validation loss
                val_loss = criterion(output, label)
                valid_loss.append(val_loss.item())

                # Calculate validation accuracy
                valid_acc.append(accuracy(output, label.to(torch.int8)))

            # Calculate average losses
            train_loss_avg = torch.mean(torch.FloatTensor(train_loss))
            valid_loss_avg = torch.mean(torch.FloatTensor(valid_loss))

            # Calculate average accuracy
            train_acc_avg = torch.mean(torch.FloatTensor(train_acc))
            valid_acc_avg = torch.mean(torch.FloatTensor(valid_acc))

            print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss_avg:.4f} \t\tValidation Loss: {valid_loss_avg:.4f}')
            print(f'\t\tTraining Accuracy: {100 * train_acc_avg:.2f}%\t Validation Accuracy: {100 * valid_acc_avg:.2f}%')

            # Save the model if validation loss decreases
            if valid_loss_avg < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_path + f'weight_{epoch}_{valid_loss_avg}.pt')
                print(f'Save weight with the loss {valid_loss_avg}, model improved the loss from {valid_loss_min} to {valid_loss_avg}')
                valid_loss_min = valid_loss_avg
            
            # Log metrics
            experiment.log_metric("train loss", train_loss_avg, step = epoch)
            experiment.log_metric("valid loss", valid_loss_avg, step = epoch)
            experiment.log_metric("train accuracy", train_acc_avg, step = epoch)
            experiment.log_metric("valid accuracy", valid_acc_avg, step = epoch)

    return model

def get_args():
    parser = argparse.ArgumentParser(description='Training arguments for Hero Name Recognition')
    parser.add_argument('--data', '--d', type=str, default='data/train', help='Path to folder containing training data')
    parser.add_argument('--label', '--lb', type=str, default='data/hero_names.txt', help='Path to label file')
    parser.add_argument('--validation-split', '--v', dest='val', type=float, default=0.2, help='Fraction of data used as validation')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Model name from timm to be used as backbone')
    parser.add_argument('--epochs', '--e', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4, help='Learning rate', dest='lr')
    parser.add_argument('--batch-size', '--b', type=int, default=32, help='Batch size')
    parser.add_argument('--save-path', '--s', type=str, default='checkpoint/', help='Path to save checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Set seed for reproducibility')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    # Load data
    train_data, val_data = split_data(data_path = args.data, label_path = args.label, validation_split = args.val)

    train_dataset = HeroNameDataset(data_path = args.data, split_data = train_data, label_path = args.label)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)

    valid_dataset = HeroNameDataset(data_path = args.data, split_data = val_data, label_path = args.label)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=True)

    # Load model
    model = HeroModel(args.backbone)
    # model.load_state_dict(torch.load(args.save_path + 'weight_297_0.06355572491884232.pt'))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    model = model.to(device)
    model = train(model,
                criterion,
                optimizer,
                train_dataloader,
                valid_dataloader,
                save_path = args.save_path,
                train_epoch = args.epochs)