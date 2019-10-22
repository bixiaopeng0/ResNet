from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from myresnet import resnet50
from PIL import Image
# import cv2
batch_size = 16
learning_rate = 0.0005
epoch = 200
pre_epoch = 0

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    #水平翻转
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

train_dir = './mydataset/train'
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

val_dir = './mydataset/test'
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)


print(train_datasets,val_datasets)

# --------------------训练过程---------------------------------

#加载预训练模型，将不匹配的部分删掉
net = resnet50(pretrained=False)
model_dict = net.state_dict()
state_dict  = torch.load('resnet50-19c8e357.pth')
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
pretrained_dict.pop('fc.weight')
pretrained_dict.pop('fc.bias')
model_dict.update(pretrained_dict)
# 3. load the new state dict
net.load_state_dict(model_dict)

# state_dict = torch.load('./model/195xxxxx.pth')
if torch.cuda.is_available():
    net.cuda()
# net.load_state_dict(state_dict)
# fc_features = net.fc.in_features
# net.fc = nn.Linear(fc_features, 2)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# 训练
epochs = 200

#动态调整学习率，每30个epoch下降百分之10
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print("learning rate",lr)

def train():
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(50,epochs):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                adjust_learning_rate(optimizer,epoch)
                for i, data in enumerate(train_dataloader, 0):
                    # 准备数据
                    length = len(train_dataloader)
                    inputs, labels = data
                    print(labels)
                    # inputs, labels = inputs.to(device), labels.to(device)
                    # inputs, labels = inputs.to(device), l
                    inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    print(predicted)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print("batch Acc%.3f"%(predicted.eq(labels.data).cpu().sum().item()/batch_size))
                    if epoch % 5 == 0:
                        torch.save(net.state_dict(),'./model/'+str(epoch)+'xxxxx.pth')
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in val_dataloader:
                        net.eval()
                        inputs, labels = data
                        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                        outputs = net(inputs)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    # if acc > best_acc:
                    #     f3 = open("best_acc.txt", "w")
                    #     f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                    #     f3.close()
                    #     best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % epoch)

def run():
    img_class = ["cat","dog"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50()
    state_dict = torch.load('./model/100xxxxx.pth')
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(state_dict)
    model.eval()  # 把模型转为test模式

    # image = cv2.imread(img_id_path, cv2.IMREAD_COLOR)
    img_path = './mydataset/dog1.jpg'
    img = Image.open(img_path)
    plt.imshow(img)
    # img = cv2.resize(img,(256,256))
    trans = transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    img = trans(img)
    img = Variable(img).cuda()
    # img = img.to(device)
    img = img.unsqueeze(0)
    output = model(img)
    prob = F.softmax(output, dim=1)
    _, predicted = torch.max(output.data, 1)
    print(prob,predicted.item())
    print("predict is",img_class[predicted.item()])
    plt.show()

if __name__ == "__main__":
    run()
    # train()