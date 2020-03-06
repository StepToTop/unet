import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model import UNet
from data import CellDataset


def get_acc(output, label):
    # print(output, label)
    total = output.shape[2] * output.shape[3]
    num_correct = (output.squeeze(1) == label.squeeze(1)).sum().item()
    return num_correct / total


image_data = CellDataset(imgs_dir='./data/train/image/', masks_dir='./data/train/label/', scale=0.5)
train_Data = DataLoader(image_data, batch_size=2, shuffle=True)
criterion = torch.nn.BCEWithLogitsLoss().cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(n_channels=1, n_classes=1)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

#
# for batch in train_Data:
#     exit()


for epoch in range(300):
    net.train()
    epoch_loss = 0
    for batch in train_Data:
        imgs = batch['image']
        true_masks = batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)
        masks_pred = net(imgs)
        # print(masks_pred)
        # exit()
        loss = criterion(masks_pred, true_masks)
        # print(loss)
        # exit()
        # print(true_masks)
        # print(masks_pred)
        # acc = get_acc(true_masks, masks_pred)
        # epoch_loss += loss.item()
        acc = 0.0
        print("loss is {}, accuracy is {}".format(loss.item(), acc))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
