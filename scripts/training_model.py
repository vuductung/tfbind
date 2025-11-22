import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from tfbind.loader import DnaDataset, collate_fn, load_dataset
from tfbind.model import DNAConvNet
from tfbind.training import train_val
from tfbind.utils.helper import get_device, get_paths, load_config, set_seed


def main(config_name="base.yml", pretrained_model_name="CTCF_model"):
    path = get_paths("local")

    # Load configuration
    config_path = os.path.join(path["config_dir"], config_name)
    config = load_config(config_path)

    # Set seed
    set_seed(config["training"]["seed"])

    # Paths
    filepath_train = os.path.join(path["data_dir"], config["data"]["train_file"])
    filepath_test = os.path.join(path["data_dir"], config["data"]["test_file"])

    # Load datasets
    loaded_train_dataset = load_dataset(filepath_train)
    loaded_test_dataset = load_dataset(filepath_test)

    dna_dataset_train = DnaDataset(labels=loaded_train_dataset["labels"], sequences=loaded_train_dataset["sequences"])
    dna_dataset_test = DnaDataset(labels=loaded_test_dataset["labels"], sequences=loaded_test_dataset["sequences"])

    # Create dataloaders
    train_dataloader = DataLoader(
        dna_dataset_train,
        batch_size=config["training"]["batch_size_train"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["training"]["num_workers"],
    )
    test_dataloader = DataLoader(
        dna_dataset_test,
        batch_size=config["training"]["batch_size_test"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["training"]["num_workers"],
    )

    # Device and model
    device = get_device(config["device"])
    print(f"Using device: {device}")

    model = DNAConvNet().to(device)

    ### CHANGE THIS CODE!!!!!!!!!!!!!!!!!!!!
    if pretrained_model_name:
        pretrained_param_dir = os.path.join(path["model_dir"], pretrained_model_name + ".pth")
        with torch.no_grad():
            dummy_input = torch.randn(1, 4, 200).to(device)
            _ = model(dummy_input)
        model.load_state_dict(torch.load(pretrained_param_dir, map_location=device))
        print(f"Loaded pretrained model from {pretrained_model_name}.pth")
    ### CHANGE THIS CODE!!!!!!!!!!!!!!!!

    # Loss, optimizer, and scheduler
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["lr"], weight_decay=config["optimizer"]["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["scheduler"]["T_max"])

    # Train
    train_val(
        epochs=config["training"]["epochs"],
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        show_plot=config["show_plot"],
        save_model_dir=path["model_dir"],
        model_name=config["model"]["name"],
        print_interval=config["training"]["print_interval"],
        scheduler=scheduler,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DNA binding model")
    parser.add_argument("--config", type=str, default="base.yml", help="Path to config file")
    args = parser.parse_args()

    main(args.config)
