import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from trashDataset import TrashDataset

# Function to calculate mean and standard deviation
def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=0)

    mean = 0.0
    std = 0.0
    num_samples = 0

    for data in loader:
        images = data['image']
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples

    return mean, std

if __name__ == "__main__":
    # Set your CSV file and root directory
    csv_file = "dataset/labels.txt"
    rootdir = "dataset"

    # Set up the dataset and calculate mean and std
    trainset = TrashDataset(csvFile=csv_file, root_dir=rootdir, transform=transforms.ToTensor())
    mean, std = calculate_mean_std(trainset)

    print("Calculated mean:", mean)
    print("Calculated std:", std)
