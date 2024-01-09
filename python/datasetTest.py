from trashDataset import TrashDataset
import cv2
import torch
from torchvision import transforms, datasets

test = TrashDataset(csvFile='dataset/labels.txt', root_dir='dataset', transform=None)
#sample = test.__getitem__(167)
#print(sample)

#print(sample['weight'])
#while True:
#    cv2.imshow('image', sample['image'])
#    key = cv2.waitKey(1)
#    if key == ord('q'):
#        break

#cv2.destroyAllWindows()

dataset_loader = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)
for i_batch, sample_batched in enumerate(dataset_loader):
    print(i_batch, sample_batched['image'].size(),sample_batched['weight'], sample_batched['class'])
    #cv2.imshow('image', sample_batched['image'].numpy()[0])
    #key = cv2.waitKey(1)
    #if key == ord('q'):
    #    break
cv2.destroyAllWindows()