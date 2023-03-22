import os
import json
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


label_dict = {"REAL": 0, "FAKE": 1}

class DeepFakeData(Dataset):
    def __init__(self, data_path, label_file, partition='train', random_state=42):
        self.data_path = data_path
        self.partition = partition

        with open(os.path.join(data_path, label_file)) as f:
            self.labels = json.load(f)
        self.data = sorted(list(self.labels.keys()))

        self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=random_state)

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        img_name = self.train[item] if self.partition == 'train' else self.test[item]

        img = Image.open(os.path.join(self.data_path, f"{img_name}.jpg"))
        img = self.preprocess(img)

        label = self.labels[img_name]
        return img, label_dict[label]

    def __len__(self):
        return len(self.train) if self.partition == 'train' else len(self.test)
    

def get_dataloaders():
    train_dataset = DeepFakeData(data_path="./data/processed_images", label_file="labels.json", partition='train')
    test_dataset = DeepFakeData(data_path="./data/processed_images", label_file="labels.json", partition='test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle = False)

    return train_loader, test_loader