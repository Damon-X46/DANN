import random
import os, sys

base_path = os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__)))
sys.path.append(base_path)

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset.data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from models.model_znr import CNNModel
import numpy as np
from test_znr import test




source_dataset_name = 'Source'
target_dataset_name = 'Target'
# source_image_root = os.path.join('.', 'dataset', source_dataset_name)
# target_image_root = os.path.join('.', 'dataset', target_dataset_name)
# model_root = os.path.join('.', 'models/save')
source_image_root = os.path.join('/emwuser/znr/code/DANN/dataset/Dataset_DANN', source_dataset_name)
target_image_root = os.path.join('/emwuser/znr/code/DANN/dataset/Dataset_DANN', target_dataset_name)
model_root = os.path.join('/emwuser/znr/code/DANN/dataset/Dataset_DANN/model_save')
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 128
image_size = 64
n_epoch = 100

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data

img_transform_source = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

img_transform_target = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# dataset_source = datasets.MNIST(                # 源域采用MNIST数据集，从网上下载
#     root='/emwuser/znr/code/DANN/dataset',
#     train=True, 
#     transform=img_transform_source,
#     download=True
# )

# dataloader_source = torch.utils.data.DataLoader(
#     dataset=dataset_source,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=8)

train_list = os.path.join(source_image_root, 'train.txt')

dataset_source = GetLoader(                     # 目标域采用mnist_m数据集
    data_root=os.path.join(source_image_root, 'train'),
    data_list=train_list,
    transform=img_transform_target
)

dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)


train_list = os.path.join(target_image_root, 'train.txt')

dataset_target = GetLoader(                     # 目标域采用mnist_m数据集
    data_root=os.path.join(target_image_root, 'train'),
    data_list=train_list,
    transform=img_transform_target
)

dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)

# load model

my_net = CNNModel()

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training

for epoch in range(n_epoch):

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data          # 用源域数据训练网络
        data_source = data_source_iter.next()
        s_img, s_label = data_source        # s_img[128, 1, 28, 28], s_label[128]

        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)    # [128, 3, 28, 28]
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)          # 源域当作0
        domain_label = domain_label.long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)            # [128, 1, 28, 28]
        class_label.resize_as_(s_label).copy_(s_label)

        class_output, domain_output = my_net(input_data=input_img, alpha=alpha)     # [128, 10], [128, 2]
        err_s_label = loss_class(class_output, class_label)             # 计算类别损失
        err_s_domain = loss_domain(domain_output, domain_label)         # 计算域损失

        # training model using target data                  # 用目标域数据训练网络
        data_target = data_target_iter.next()
        t_img, _ = data_target                              # [128, 3, 28, 28], [128]

        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        domain_label = torch.ones(batch_size)               # 目标域当作1
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)

        _, domain_output = my_net(input_data=input_img, alpha=alpha)    # [128, 10], [128, 2]
        err_t_domain = loss_domain(domain_output, domain_label)         # 计算目标域损失
        err = err_t_domain + err_s_domain + err_s_label                 # 总损失=目标域损失+源域损失+源域类别损失
        err.backward()
        optimizer.step()

        i += 1

        print ('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                 err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

    torch.save(my_net, '{0}/mnist_mnistm_model_epoch_{1}.pth'.format(model_root, epoch))
    test(source_dataset_name, epoch)
    test(target_dataset_name, epoch)

print ('done')
