from trashDataset import TrashDataset
from skimage import io, transform

test = TrashDataset(csvFile='dataset/labels.txt', root_dir='dataset', transform=None)
print(test.__getitem__(0))