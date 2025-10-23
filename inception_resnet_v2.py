from sched import scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchinfo
import torchvision.models as models
import timm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

class Reduction_A(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
      super(Reduction_A, self).__init__()
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
      self.conv1 = BasicConv2d(in_channels ,n, kernel_size=3, stride=2, padding=0, bias=False)
      self.conv2 = nn.Sequential(
          BasicConv2d(in_channels, k, kernel_size=1, padding=0, bias=False),
          BasicConv2d(k, l, kernel_size=3, padding=1, bias=False),
          BasicConv2d(l, m, kernel_size=3, stride=2, padding=0, bias=False)
      )

    def forward(self, x):
      x1 = self.maxpool(x)
      x2 = self.conv1(x)
      x3 = self.conv2(x)
      return torch.cat((x1, x2, x3), 1)
  
class inception_Resnet_v2_stem(nn.Module):
    def __init__(self, in_channels):
        super(inception_Resnet_v2_stem, self).__init__()
        self.feature = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=3, stride=2, padding=0, bias=False),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            BasicConv2d(64, 80, kernel_size=1, stride=1, padding=0, bias=False),
            BasicConv2d(80, 192, kernel_size=3, stride=1, padding=0, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )

        self.conv1 = BasicConv2d(192, 96, kernel_size=1, padding=0, bias=False)

        self.stem2 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, padding=0, stride=1, bias=False),
            BasicConv2d(48, 64, kernel_size=5, padding=2, stride=1, bias=False),
        )
        self.stem3 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=False),
            BasicConv2d(64, 96, kernel_size=3, padding=1, stride=1, bias=False),
            BasicConv2d(96, 96, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.avgpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=False)
        )

    def forward(self, x):
        feature = self.feature(x)
        x0 = self.conv1(feature)
        x1 = self.stem2(feature)
        x2 = self.stem3(feature)
        x3 = self.avgpool(feature)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out
    
class inception_Resnet_A_v2(nn.Module):
  def __init__(self, in_channels, scale_factor):
    super(inception_Resnet_A_v2, self).__init__()
    self.scale_factor = scale_factor
    self.conv0 = BasicConv2d(in_channels, 32, kernel_size=1, padding=0, stride=1, bias=False)
    self.conv1 = nn.Sequential(
        BasicConv2d(in_channels, 32, kernel_size=1, padding=0, stride=1, bias=False),
        BasicConv2d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
    )
    self.conv2 = nn.Sequential(
        BasicConv2d(in_channels, 32, kernel_size=1, padding=0, stride=1, bias=False),
        BasicConv2d(32, 48, kernel_size=3, padding=1, stride=1, bias=False),
        BasicConv2d(48, 64, kernel_size=3, padding=1, stride=1, bias=False)
    )
    self.conv3 = nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0, bias=True)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    identity = x
    x0 = self.conv0(x)
    x1 = self.conv1(x)
    x2 = self.conv2(x)
    out = torch.cat((x0, x1, x2), 1)
    out = self.conv3(out)
    #print(out.shape)
    return self.relu(out*self.scale_factor + identity)

class inception_Resnet_B_v2(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(inception_Resnet_B_v2, self).__init__()
        self.scale_factor = scale_factor
        self.conv0 =BasicConv2d(in_channels, 192, kernel_size=1, padding=0, bias=False)

        self.conv1 = nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size=1, padding=0, bias=False),
            BasicConv2d(128, 160, kernel_size=(1, 7), padding=(0, 3), bias=False),
            BasicConv2d(160, 192, kernel_size=(7, 1), padding=(3, 0), bias=False)
        )

        self.conv2 = nn.Conv2d(384, 1088, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
      identity = x
      x0 = self.conv0(x)
      x1 = self.conv1(x)
      out = torch.cat((x0, x1), 1)
      out = self.conv2(out)
      return self.relu(out*self.scale_factor + identity)

class inception_Resnet_C_v2(nn.Module):
    def __init__(self, in_channels, scale_factor=1.0, activation=True):
        super(inception_Resnet_C_v2, self).__init__()
        self.activation = activation
        self.scale_factor = scale_factor

        self.conv0 = BasicConv2d(in_channels, 192, kernel_size=1, padding=0, bias=False)

        self.conv1 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1, padding=0, bias=False),
            BasicConv2d(192, 224, kernel_size=(1, 3), padding=(0, 1), bias=False),
            BasicConv2d(224, 256, kernel_size=(3, 1), padding=(1, 0), bias=False),
        )

        self.conv2 = nn.Conv2d(448, 2080, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
      identity = x
      x0 = self.conv0(x)
      x1 = self.conv1(x)
      out = torch.cat((x0, x1), 1)
      out = self.conv2(out)
      if self.activation:
          out = self.relu(out*self.scale_factor + identity)
      return out*self.scale_factor + identity

class Reduction_Resnet_B_v2(nn.Module):
  def __init__(self, in_channels):
    super(Reduction_Resnet_B_v2, self).__init__()
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    self.conv0 = nn.Sequential(
        BasicConv2d(in_channels, 256, kernel_size=1, padding=0, bias=False),
        BasicConv2d(256, 384, kernel_size=3, stride=2, padding=0, bias=False),
    )

    self.conv1 = nn.Sequential(
        BasicConv2d(in_channels, 256, kernel_size=1, padding=0, bias=False),
        BasicConv2d(256, 288, kernel_size=3, stride=2, padding=0, bias=False),
    )

    self.conv2 = nn.Sequential(
        BasicConv2d(in_channels, 256, kernel_size=1, padding=0, bias=False),
        BasicConv2d(256, 288, kernel_size=3, padding=1, bias=False),
        BasicConv2d(288, 320, kernel_size=3, stride=2, padding=0, bias=False)
    )

  def forward(self, x):
    x0 = self.conv0(x)
    x1 = self.conv1(x)
    x2 = self.conv2(x)
    x3 = self.maxpool(x)
    return torch.cat((x0, x1, x2, x3), 1)  
 
  #Inception_Resnet_V2 residual scaling필요!!
class Inception_Resnet_V2(nn.Module):
  def __init__(self, num_classes, in_channels, k=256, l=256, m=384, n=384):
    super(Inception_Resnet_V2, self).__init__()
    self.features = nn.Sequential(
        inception_Resnet_v2_stem(in_channels),
        inception_Resnet_A_v2(320, 0.17),
        inception_Resnet_A_v2(320, 0.17),
        inception_Resnet_A_v2(320, 0.17),
        inception_Resnet_A_v2(320, 0.17),
        inception_Resnet_A_v2(320, 0.17),
        inception_Resnet_A_v2(320, 0.17),
        inception_Resnet_A_v2(320, 0.17),
        inception_Resnet_A_v2(320, 0.17),
        inception_Resnet_A_v2(320, 0.17),
        inception_Resnet_A_v2(320, 0.17),
        Reduction_A(320, k, l, m, n),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        inception_Resnet_B_v2(1088, 0.10),
        Reduction_Resnet_B_v2(1088),
        inception_Resnet_C_v2(2080, 0.20, True),
        inception_Resnet_C_v2(2080, 0.20, True),
        inception_Resnet_C_v2(2080, 0.20, True),
        inception_Resnet_C_v2(2080, 0.20, True),
        inception_Resnet_C_v2(2080, 0.20, True),
        inception_Resnet_C_v2(2080, 0.20, True),
        inception_Resnet_C_v2(2080, 0.20, True),
        inception_Resnet_C_v2(2080, 0.20, True),
        inception_Resnet_C_v2(2080, 0.20, True),
        inception_Resnet_C_v2(2080, False)
    )

    self.conv_final = BasicConv2d(2080, 1536, kernel_size=1, stride=1, padding=0, bias=False)
    self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout(0.2)
    self.linear = nn.Linear(1536, num_classes)

  def forward(self, x):
    out = self.features(x)
    out = self.conv_final(out)
    out = self.global_average_pooling(out)
    out = out.view(out.size(0), -1)
    out = self.dropout(out)
    out = self.linear(out)
    return out
              
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

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels)

        epoch_train_loss = running_loss / len(train_dataloader)
        epoch_train_acc = running_corrects.double() / len(train_dataloader.dataset)

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

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_running_loss += loss.item()*inputs.size(0)
                val_running_corrects += torch.sum(preds == labels)

        epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
        epoch_val_acc = val_running_corrects.double() / len(val_dataloader.dataset)
        scheduler.step(epoch_val_acc)

        print(f'Validation Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc.double():.4f}')
        print('*' * 30)

        train_accuracy_list.append(epoch_train_acc.item())
        train_loss_list.append(epoch_train_loss)
        val_accuracy_list.append(epoch_val_acc.item())
        val_loss_list.append(epoch_val_loss)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f'New best model found at epoch {epoch + 1} with Validation Accuracy: {best_val_acc:.4f}. Saving checkpoint...')
            torch.save(model.state_dict(), 'best_model_checkpoint1.pth')

    return train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list
  
  
def test_model(net, dataloader, criterion, num_epochs) :
  net.eval()
  accuracy_list = []
  loss_list = []

  for epoch in range(num_epochs) :
    print(f'Epoch {epoch + 1}/ {num_epochs}')
    print('*' * 30)

    epoch_test_loss = 0.0
    epoch_test_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            epoch_test_loss += loss.item() * inputs.size(0)
            epoch_test_corrects += torch.sum(preds == labels.data)

        epoch_test_loss = epoch_test_loss / len(dataloader.datasets)
        epoch_acc = epoch_test_corrects.double() / len(dataloader.datasets)

        print(f'Loss: {epoch_test_loss:.4f} Acc: {epoch_acc:.4f}')

        accuracy_list.append(epoch_acc.item())
        loss_list.append(epoch_test_loss)
  return accuracy_list, loss_list

batch_size = 32
num_epochs = 160
learning_rate = 0.045

root = './dataset'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
      transforms.RandomResizedCrop((299, 299)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

test_val_transforms = transforms.Compose([
      transforms.Resize((299, 299)),
      transforms.ToTensor(),
      transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

train_dataset = dset.CIFAR10(root='../dataset', train=True, download=True, transform=train_transforms)
valid_dataset = dset.CIFAR10(root='../dataset', train=False, download=True, transform=test_val_transforms)
test_dataset = dset.CIFAR10(root='../dataset', train=False, download=True, transform=test_val_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = Inception_Resnet_V2(num_classes=1000, in_channels=3)
pre_trained_model = timm.create_model('inception_resnet_v2', pretrained=False)

pre_trained_weights = pre_trained_model.state_dict()
model_state_dict = model.state_dict()

new_state_dict = {}
model_keys = list(model_state_dict.keys())
pre_trained_values = list(pre_trained_weights.values())

if len(model_keys) == len(pre_trained_values):
    for i in range(len(model_keys)):
        k_custom = model_keys[i]
        v_pre = pre_trained_values[i]
        if v_pre.shape == model_state_dict[k_custom].shape:
            new_state_dict[k_custom] = v_pre

model.load_state_dict(new_state_dict)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha = 0.9, eps = 1.0)
lr_scheduler = lrs.StepLR(optimizer, step_size=2, gamma=0.94)

train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list = train_model(model, train_dataloader, valid_dataloader, criterion, lr_scheduler, optimizer, num_epochs=num_epochs)

# 그래프 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="w")

# 정확도(Accuracy) 그래프
ax1.plot(train_accuracy_list, label="Train Accuracy")
ax1.plot(val_accuracy_list, label="Validation Accuracy")
ax1.set_title("Accuracy over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

# 손실(Loss) 그래프
ax2.plot(train_loss_list, label="Train Loss")
ax2.plot(val_loss_list, label="Validation Loss")
ax2.set_title("Loss over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('inception_resnet_v2_train_val_loss_plot1.png')
plt.show()
#torchinfo.summary(model, input_size=(1, 3, 224, 224))
test_acc, test_loss = test_model(model, test_dataloader, criterion, num_epochs=1)

model = Inception_Resnet_V2(num_classes=1000, in_channels=3)
print(len(model.state_dict().keys()))