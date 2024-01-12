from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, ):


# Train / Validation Split (70% : 30%)
train_len = int(len(train_df) * 0.7)
val_df = train_df.iloc[train_len:]
train_df = train_df.iloc[:train_len]
train_labels = train_df.iloc[:, 2:].values.reshape(-1, 4, 4)
val_labels = val_df.iloc[:, 2:].values.reshape(-1, 4, 4)
(train_labels.shape, val_labels.shape)


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transform=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        # PIL 이미지로 불러오기
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.label_list is not None:
            label = torch.tensor(self.label_list[index], dtype=torch.long) - 1
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = CustomDataset(train_df['img_path'].values, train_labels, train_transform)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val_df['img_path'].values, val_labels, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

test_dataset = CustomDataset(test_df['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
