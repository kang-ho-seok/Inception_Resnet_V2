import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, models
from torch.utils.data import TensorDataset, DataLoader
import torch.optim.lr_scheduler as lrs
import random
import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.LRN = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)


    def forward(self, x):
        out = self.conv(x)
        out = self.LRN(out)


        return out

# @title inception module
class InceptionModule(nn.Module):
  def __init__(self, in_channels,
               reduce_3_out, reduce_5_out,
               conv_1_out, conv_3_out, conv_5_out,
               proj_out):

    super(InceptionModule, self).__init__()

    self.relu = nn.ReLU(inplace=True)

    self.reduce_3 = nn.Conv2d(in_channels = in_channels,
                         out_channels = reduce_3_out,
                         kernel_size = (1, 1),
                         stride = 1,)

    self.conv_3 = nn.Conv2d(in_channels = reduce_3_out,
                       out_channels = conv_3_out,
                       kernel_size = (3, 3),
                       stride = 1,
                       padding = 1)

    self.reduce_5 = nn.Conv2d(in_channels = in_channels,
                         out_channels = reduce_5_out,
                         kernel_size = (1, 1),
                         stride = 1)
    self.conv_5 = nn.Conv2d(in_channels = reduce_5_out,
                       out_channels = conv_5_out,
                       kernel_size =(5, 5),
                       stride = 1,
                       padding = 2)

    self.pool = nn.MaxPool2d(kernel_size = (3, 3),
                             stride = (1, 1),
                             padding = 1)
    self.proj = nn.Conv2d(in_channels = in_channels,
                          out_channels = proj_out,
                          kernel_size = (1, 1),
                          stride=1)

    self.conv_1 = nn.Conv2d(in_channels = in_channels,
                          out_channels = conv_1_out,
                          kernel_size = (1, 1),
                          stride=1)

  def forward(self, x):
    out1 = self.relu(self.conv_1(x))

    out2 = self.reduce_3(x)
    out2 = self.relu(out2)
    out2 = self.conv_3(out2)
    out2 = self.relu(out2)

    out3 = self.reduce_5(x)
    out3 = self.relu(out3)
    out3 = self.conv_5(out3)
    out3 = self.relu(out3)

    out4 = self.pool(x)
    out4 = self.proj(out4)
    out4 = self.relu(out4)

    return torch.cat((out1, out2, out3, out4), 1)

# @title Auxiliary Classifier

class AuxiliaryClassifier(nn.Module):
  def __init__(self, in_channels, num_classes):
    super(AuxiliaryClassifier, self).__init__()

    self.Auxiliary_Classifier = nn.Sequential(nn.AvgPool2d(kernel_size=(5, 5), stride=3),
                                              nn.Conv2d(in_channels=in_channels, out_channels=128,
                                                        kernel_size=(1, 1), stride=1),
                                              nn.ReLU(inplace=True),
                                              nn.Flatten(),
                                              nn.Linear(in_features=128*4*4,
                                                        out_features= 1024),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(in_features=1024,
                                                        out_features= num_classes))

  def forward(self, x):
    return self.Auxiliary_Classifier(x)

# @title GoogleNet

class GoogleNet(nn.Module):
  def __init__(self, img_channels = 3, num_classes = 1000):
    super(GoogleNet, self).__init__()
    self.prev_inception = nn.Sequential(
        nn.Conv2d(in_channels = img_channels,
                  out_channels = 64,
                  kernel_size = (7, 7),
                  stride = 2,
                  padding = 3),
        nn.MaxPool2d(kernel_size = (3, 3),
                     stride = 2,
                     padding = 1),
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        
        nn.Conv2d(in_channels = 64,
                  out_channels = 64,
                  kernel_size = (1, 1),
                  stride = 1),
        
        nn.Conv2d(in_channels = 64,
                  out_channels = 192,
                  kernel_size =(3, 3),
                  stride = 1,
                  padding = 1),
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        nn.MaxPool2d(kernel_size = (3, 3),
                     stride= 2,
                     padding = 1)
    )

    self.inception3a = InceptionModule(in_channels = 192,
                         reduce_3_out = 96,
                         reduce_5_out = 16,
                         conv_1_out = 64,
                         conv_3_out = 128,
                         conv_5_out = 32,
                         proj_out = 32)

    self.inception3b = InceptionModule(in_channels = 256,
                         conv_1_out = 128,
                         reduce_3_out = 128,
                         conv_3_out = 192,
                         reduce_5_out = 32,
                         conv_5_out = 96,
                         proj_out = 64)

    self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

    self.inception4a = InceptionModule(in_channels = 480,
                         conv_1_out = 192,
                         reduce_3_out = 96,
                         conv_3_out = 208,
                         reduce_5_out = 16,
                         conv_5_out = 48,
                         proj_out = 64)

    self.aux1 =  AuxiliaryClassifier(in_channels = 512,
                                              num_classes = num_classes)

    self.inception4b = InceptionModule(in_channels = 512,
                         conv_1_out = 160,
                         reduce_3_out = 112,
                         conv_3_out = 224,
                         reduce_5_out = 24,
                         conv_5_out = 64,
                         proj_out = 64)

    self.inception4c = InceptionModule(in_channels = 512,
                         conv_1_out = 128,
                         reduce_3_out = 128,
                         conv_3_out = 256,
                         reduce_5_out = 24,
                         conv_5_out = 64,
                         proj_out = 64)

    self.inception4d = InceptionModule(in_channels = 512,
                         conv_1_out = 112,
                         reduce_3_out = 144,
                         conv_3_out = 288,
                         reduce_5_out = 32,
                         conv_5_out = 64,
                         proj_out = 64)

    self.aux2 =  AuxiliaryClassifier(in_channels = 528,
                                              num_classes = num_classes)

    self.inception4e = InceptionModule(in_channels = 528,
                         conv_1_out = 256,
                         reduce_3_out = 160,
                         conv_3_out = 320,
                         reduce_5_out = 32,
                         conv_5_out = 128,
                         proj_out = 128)

    self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

    self.inception5a = InceptionModule(in_channels = 832,
                         conv_1_out = 256,
                         reduce_3_out = 160,
                         conv_3_out = 320,
                         reduce_5_out = 32,
                         conv_5_out = 128,
                         proj_out = 128)

    self.inception5b = InceptionModule(in_channels = 832,
                         conv_1_out = 384,
                         reduce_3_out = 192,
                         conv_3_out = 384,
                         reduce_5_out = 48,
                         conv_5_out = 128,
                         proj_out = 128)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.drop = nn.Dropout(p=0.4)
    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = 1024,
                  out_features = num_classes)
    )

  def forward(self, x):
    out = self.prev_inception(x)
    out = self.inception3a(out)
    out = self.inception3b(out)
    out = self.maxpool3(out)

    out = self.inception4a(out)
    out_1 = self.aux1(out)

    out = self.inception4b(out)
    out = self.inception4c(out)
    out = self.inception4d(out)
    out_2 = self.aux2(out)

    out = self.inception4e(out)
    out = self.maxpool4(out)

    out = self.inception5a(out)
    out = self.inception5b(out)
    out = self.avgpool(out)
    out = self.drop(out)
    out = self.fc(out)

    return out_1, out_2, out

def train_model(model, train_dataloader, val_dataloader, criterion, scheduler, optimizer, num_epochs) :
    model.to(device)
    torch.backends.cudnn.benchmark = True

    train_accuracy_list = []
    val_accuracy_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs) :
        print(f'Epoch {epoch + 1}/ {num_epochs}')
        print('*' * 30)

        # ====== 학습(Training) 단계 ======
        model.train() # 모델을 학습 모드로 설정

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm.tqdm(train_dataloader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            out1, out2, out = model(inputs)
            loss_main = criterion(out, labels)
            loss_aux1 = criterion(out1, labels)
            loss_aux2 = criterion(out2, labels)

            loss = loss_main + 0.3*loss_aux1 + 0.3*loss_aux2

            _, preds = torch.max(out, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_dataloader.dataset)

        scheduler.step()

        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc.double():.4f}')

        # ====== 검증(Validation) 단계 ======
        model.eval() # 모델을 평가 모드로 설정

        best_val_acc = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad(): # 검증 단계에서는 그라디언트 계산 비활성화
            for inputs, labels in tqdm.tqdm(val_dataloader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                out1, out2, out = model(inputs)
                loss = criterion(out, labels)
                _, preds = torch.max(out, 1)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels)

        epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
        epoch_val_acc = val_running_corrects.double() / len(val_dataloader.dataset)

        print(f'Validation Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc.double():.4f}')
        print('*' * 30)

        train_accuracy_list.append(epoch_train_acc.item())
        train_loss_list.append(epoch_train_loss)
        val_accuracy_list.append(epoch_val_acc.item())
        val_loss_list.append(epoch_val_loss)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f'New best model found at epoch {epoch + 1} with Validation Accuracy: {best_val_acc:.4f}. Saving checkpoint...')
            torch.save(model.state_dict(), 'best_model_checkpoint.pth')

    return train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list

# 학습과 검증 결과를 함께 시각화

# ===== test_model 함수 호출 수정 =====
def test_model(model, dataloader) :
    model.to(device)
    model.eval()
    accuracy_list = []
    loss_list = []
    with torch.no_grad() :
        for inputs, labels in tqdm.tqdm(dataloader) :
            inputs = inputs.to(device)
            labels = labels.to(device)
            out1, out2, out = model(inputs)
            loss = criterion(out, labels)
            _, preds = torch.max(out, 1)
            accuracy_list.append(torch.sum(preds == labels).item())
            loss_list.append(loss.item())
    return accuracy_list, loss_list

batch_size = 100
num_epochs = 5
learning_rate = 0.001
input_size = (32, 32)

root = './dataset'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ====== 학습 데이터셋을 위한 변환 (Data Augmentation 포함) ======
#train은 환경 상 imagenet을 사용할 수 없어서... test만 사용함..
#patch sampling, photometric distortions기법 사용...

train_transforms = transforms.Compose([
   transforms.Resize(224),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# multiscale_sizes = [256, 288, 320, 352]
#각 이미지를 multiscale_sizes로 변환
#변환된 4이미지를 3가지 정사각형(오른쪽, 중앙, 왼쪽)으로
#만들어진 12개의 각 이미지들의 각 코너 + 중앙 --> 5개의 crop + 224로 리사이즈한 1개
# 72개의 이미지 좌우반전 ---> 총 144개 이미지...
#솔직히 좀 복잡해서 단일 crop만 사용하자..

test_val_transforms = transforms.Compose([
      transforms.Resize(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
train_data = datasets.CIFAR10(root='../dataset', train=True, download=True, transform=train_transforms)
test_data = datasets.CIFAR10(root='../dataset', train=False, download=True, transform=test_val_transforms)
# train_data = dset.ImageNet(root=root, train=True, transform=train_transforms, download=True)
# test_data = dset.ImageNet(root=root, train=False, transform=test_val_transforms, download=True)

train_size = int(0.9 * len(train_data))
val_size = len(train_data) - train_size
dataset_train, dataset_val = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

model = GoogleNet().to(device)
pre_trained_model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
# print(pre_trained_model)
pre_trained_weights = pre_trained_model.state_dict()
model_state_dict = model.state_dict()

# 키 맞춰서 업데이트 !!layer 이름이 동일해야 함
pretrained_weights = {k: v for k, v in pre_trained_weights.items() if k in model_state_dict}
model_state_dict.update(pretrained_weights)

model.load_state_dict(model_state_dict)

for param in model.parameters():
    param.requires_grad = True  # 전체 학습
# for param in model.classifier[1].parameters():
#     param.requires_grad = True   # 학습되는 레이어 설정

# 마지막 레이어 교체
num_ftrs = model.fc[1].in_features
model.fc[1] = nn.Linear(num_ftrs, 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=learning_rate, weight_decay=0.0002)
lr_scheduler = lrs.StepLR(optimizer, step_size=8, gamma=0.96)

train_accuracy, val_accuracy, train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, lr_scheduler, optimizer, num_epochs=5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="w")

# 정확도(Accuracy) 그래프
ax1.plot(train_accuracy, label="Train Accuracy")
ax1.plot(val_accuracy, label="Validation Accuracy")
ax1.set_title("Accuracy over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

# 손실(Loss) 그래프
ax2.plot(train_loss, label="Train Loss")
ax2.plot(val_loss, label="Validation Loss")
ax2.set_title("Loss over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

test_accuracy_list, test_loss_list = test_model(model, test_loader)
print(f'Test Accuracy : {np.sum(test_accuracy_list) / len(test_data):.4f}')