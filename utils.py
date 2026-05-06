import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import matplotlib.pyplot as plt 
from torchvision.transforms import v2
import timm
import random
import math
import os

register_heif_opener()


def make_train_transform(input_size:int):
    train_transform = v2.Compose([
    v2.RandomResizedCrop(int(input_size*1.1), scale=(0.5, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(10, fill=(124, 116, 104)),
    v2.CenterCrop(input_size),
    v2.RandomApply([v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3)], p=0.6),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.1),
    v2.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
    v2.RandomGrayscale(p=0.05),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.2, scale=(0.02, 0.04)), 
    ])
    return train_transform

def make_inference_transform(input_size:int):
    val_transform = v2.Compose([
    v2.Resize(int(input_size*1.1)),
    v2.CenterCrop(input_size),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return val_transform

def make_query_index_split(test_dataset_path:str, num_of_queries_per_class:int, seed= None):
    class_names = sorted(os.listdir(test_dataset_path))
    itoclassname = dict(enumerate(class_names))
    classnametoi = {name:i for i, name in itoclassname.items()}
    data = []
    for name in class_names:
        for image in os.listdir(f"{test_dataset_path}/{name}"):
            data.append((f"{test_dataset_path}/{name}/{image}", classnametoi[name]))

    query_data = []
    index_data = []
    for label in itoclassname:
        class_data = [x for x in data if x[1] == label]
        if len(class_data) <= num_of_queries_per_class:
            index_data.extend(class_data)
            continue
        if seed:
            random.Random(seed).shuffle(class_data)
        else:
            random.shuffle(class_data)
        query_data.extend(class_data[:num_of_queries_per_class])
        index_data.extend(class_data[num_of_queries_per_class:])
    return (query_data, index_data), itoclassname

def build_emb_matrix(model:nn.Module, loader:DataLoader, device):
    model.eval()
    embs = []
    labels = []
    with torch.no_grad():
        for img_batch, label_batch in loader:
            emb_batch = model(img_batch.to(device))
            emb_batch = F.normalize(emb_batch, dim=1)
            embs.append(emb_batch.cpu())
            labels.append(label_batch)
    return torch.cat(embs), torch.cat(labels)

def mean_average_precision_at_k(query_embs, query_labels, index_embs, index_labels, k=20):
    sims = query_embs @ index_embs.T
    aps = []
    for i in range(len(query_embs)):
        topk_idx = sims[i].topk(k).indices
        topk_labels = index_labels[topk_idx]
        relevant = (topk_labels == query_labels[i])
        if relevant.sum() == 0:
            continue
        precisions = []
        hits = 0
        for j, rel in enumerate(relevant):
            if rel:
                hits += 1
                precisions.append(hits / (j + 1))
        aps.append(sum(precisions) / relevant.sum().item())
    return sum(aps) / len(aps)

def recall_at_k(query_embs, query_labels, index_embs, index_labels, k=5):
    sims = query_embs @ index_embs.T
    topk = sims.topk(k).indices
    correct = (index_labels[topk] == query_labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()

def validate_test_set(model, query_dataset, index_dataset, batch_size, device):
    model.eval()
    query_loader = DataLoader(
        query_dataset,
        batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    index_loader = DataLoader(
        index_dataset,
        batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    query_embs, query_labels = build_emb_matrix(model, query_loader, device)
    index_embs, index_labels = build_emb_matrix(model, index_loader, device)

    map_score = mean_average_precision_at_k(query_embs, query_labels, index_embs, index_labels, k=20)
    r1 = recall_at_k(query_embs, query_labels, index_embs, index_labels, k=1)
    r3 = recall_at_k(query_embs, query_labels, index_embs, index_labels, k=3)
    return map_score, r1, r3

class TestDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.data)
    

class TrainDataset(Dataset):
    def __init__(self, train_dataset_path:str, transform=None):
        self.train_dataset_path = train_dataset_path
        self.df = pd.read_csv(f"{train_dataset_path}/image_to_label.csv")
        self.image_names = self.df["image_name"].values
        self.labels = self.df["label"].values
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.image_names[index]
        label = int(self.labels[index])
        img_path = f"{self.train_dataset_path}/gldv2/{img_name[0]}/{img_name[1]}/{img_name[2]}/{img_name}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.image_names)
    

class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.3, scale=64):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta_cap = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1).to(embeddings.dtype)
        cos = embeddings @ weights.T
        sin = torch.sqrt(1.0 - torch.pow(cos, 2).clamp(1e-7, 1)).clamp(0, 1)
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.theta_cap, phi, cos - self.mm)
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cos)
        
        return output * self.scale


class GemPool(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        if x.dim() == 2:
            return x 
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 
            (x.size(-2), x.size(-1))
        ).pow(1. / self.p).squeeze(-1).squeeze(-1)


class LandmarkModel(nn.Module):
    def __init__(self, model_name:str, num_classes, emb_dim=512):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        self.pooling = GemPool()
        self.neck = nn.Sequential(
            nn.Linear(self.backbone.num_features, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )
        self.head = ArcFaceHead(emb_dim, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        pooled = self.pooling(features)
        embeddings = self.neck(pooled)
        embeddings = F.normalize(embeddings, dim=1)

        if labels is not None:
            return self.head(embeddings, labels)
        
        return embeddings

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

if __name__ == "__main__":
    def untransform(tensor):
        tensor = tensor.cpu().detach()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(1, 2, 0).numpy()
    
    def plot_data(data):
        plt.figure(figsize=(18, 18))
        for i in range(6):
            for j in range(6):
                plt.subplot(6,6,6*i+j+1)
                img, label =data[random.randint(0, len(data)-1)]
                plt.imshow(untransform(img))
                plt.title(label, fontsize=8)
                plt.axis("off")
        plt.tight_layout()
        plt.show()

    # data = TrainDataset("landmarks_data/train_dataset", transform= make_train_transform(224))
    (query, index), itoclass = make_query_index_split("landmarks_data/test_dataset", num_of_queries_per_class=2)
    data = TestDataset(index, transform=make_inference_transform(224))
    plot_data(data)