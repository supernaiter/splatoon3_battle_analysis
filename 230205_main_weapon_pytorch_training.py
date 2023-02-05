# %% [markdown]
# ここに従ってpytorchの転移学習実装を作る． https://torch.classcat.com/category/transfer-learning/
# 
# convnet を再調整する : ランダム初期化の代わりに、imagenet 1000 データセット上で訓練された一つのような、事前訓練されたネットワークでネットワークを初期化します。訓練の残りは通常のようなものです。

# %%
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import copy
from PIL import Image
#import torchvision.transforms as transforms
from torch.nn import functional as F

# %% [markdown]
# データをロードするために torchvision と torch.utils.data パッケージを使用します。

# %% [markdown]
# https://reafnex.net/ai/pytorch-use-imagefolder/
# 
# 
# 次に、ImageFolderを使用して、先ほど作成したMNISTの手書き文字画像ファイルを取り込んでみます。最終的にバッチ分割されたデータローダーを作成します。
# 
# まずは、画像をテンソル化した後に、イメージ画像のデータ変換（標準化など）を行うクラスを定義します。

# %%

# %% [markdown]
# 次にImageFolderを使用して画像データの取り込みを行います。イメージ画像の格納状態は以下のようになっています。各数字の画像ファイルを格納しているディレクトリ名を、イメージ画像のラベルとして使用します。
# 
# 
# ・torchvision.datasets.ImageFolderは、イメージ画像ファイルを格納したディレクトリと画像変換設定を与えるだけなので、イメージ画像ファイルの整理さえできていれば、データ取込みがとても簡単にできてしまいます。PyTorchを使うならImageFolderを利用することで開発効率が格段にアップします。
# 
# 以下のコードでイメージ画像の取り込みを行っています。

# %%
#画像データをImageFolderを使って取込みする

class ImageTransform():
  def __init__(self, mean, std):
    self.data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

  def __call__(self, img):
    return self.data_transform(img)
mean = (0.5,)
std = (0.5,)
#images = torchvision.datasets.ImageFolder( "/content/drive/MyDrive/2023/splashlog/230204_main_weapons", transform = ImageTransform(mean, std))
images = torchvision.datasets.ImageFolder( "../230205_main_weapons_color_augmented", transform = ImageTransform(mean, std))

# %%
n_classes = len(images.classes)
print(n_classes)

class_names = images.classes

f = open('main_weapon_list.txt', 'w')
for x in class_names:
    f.write(str(x) + "\n")
f.close()


# %% [markdown]
# https://qiita.com/takurooo/items/ba8c509eaab080e2752c

# %%

trainval_dataset = images

n_samples = len(trainval_dataset) # n_samples is 60000
train_size = int(len(trainval_dataset) * 0.8) # train_size is 48000
val_size = n_samples - train_size # val_size is 48000

# shuffleしてから分割してくれる.
train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

print(len(train_dataset)) # 48000
print(len(val_dataset)) # 12000


test_dataset, val_dataset = torch.utils.data.random_split(val_dataset, [len(val_dataset)//2, len(val_dataset)-len(val_dataset)//2])

print(len(train_dataset)) # 48000
print(len(val_dataset)) # 12000
print(len(test_dataset)) # 12000

# %%
dataset_sizes = {"train":len(train_dataset), "val":len(val_dataset)}

dataloaders = {}
dataloaders["train"] = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                             shuffle=True, num_workers=4)
dataloaders["val"] = torch.utils.data.DataLoader(val_dataset, batch_size=4,
                                             shuffle=True, num_workers=4)

# %% [markdown]
# モデルを訓練する
# さて、モデルを訓練するための一般的な関数を書きましょう。ここでは、次を示します :
# 
# ・学習率をスケジューリングする
# 
# ・ベスト・モデルをセーブする
# 
# 以下で、パラメータ・スケジューラは torch.optim.lr_scheduler からの LR スケジューラ・オブジェクトです。
# 

# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
            #for inputs, labels in train_dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# %% [markdown]
# ConvNet を再調整する
# 事前訓練されたモデルをロードして最後の完全結合層をリセットします。

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
#model_ft = models.mobilenet_v2(pretrained=True)
#最後の完全結合層（全結合層）は、model_ft.fc = nn.Linear(num_ftrs, 2)という行でリセットされています。 この行では、元のモデルのfc層（出力層）が新しい線形層に置き換えられています。
#次に、出力層のノード数を2に設定するために、model_ft.fcを更新します。
num_ftrs = model.fc.in_features
# Alternatively, it can be generalized to nn.Linear(num_ftrs, n_classes).
model.fc = nn.Linear(num_ftrs, n_classes)

#num_ftrs = model.classifier[0].in_features
#model.classifier[0] = nn.Linear(num_ftrs, n_classes)

#model = torchvision.models.mobilenet_v2(pretrained=True)
#model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=n_classes)

#次に、学習させるために、GPUまたはCPUにモデルを送信します。
model = model.to(device)

#損失関数として、クロスエントロピー損失関数を設定します。
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# %%
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=20)

# %%
torch.save(model, 'main_weapons_classification_weight.pth')

# %%
