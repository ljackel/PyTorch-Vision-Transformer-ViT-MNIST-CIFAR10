import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from torch import optim
from model import VisionTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from termcolor import colored
from torchvision import datasets, transforms
use_less_data = True

from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'
# from helpers import plot

from typing import Any, Callable, Optional, List, Tuple

# class Augmented_Data(Dataset):
#     """
#     Augmented_Data is a class that is used to create a dataset from the
#     augmented data.  It is used to create the training and test datasets
#     for the neural networks.
#     """

#     def __init__(self, data, targets, transform: Optional[Callable] = None
#         ):
#         self.data = data
#         self.targets = targets

#         self.transform = transform

#     # This returns the total amount of samples in our dataset
#     def __len__(self):
#         return len(self.targets)

#     # This returns given an index to the i-th sample and label
#     def __getitem__(self, idx):
#         img, target = self.data[idx], int(self.targets[idx])

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img.numpy(), mode="L")

#         if self.transform is not None:
#             img = self.transform(img)

#         return img, target


def get_loader(args):
    train_transform = transforms.Compose([transforms.RandomCrop(args.image_size, padding=2, padding_mode='edge'), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.5], [0.5])])
    train = datasets.MNIST(os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_transform)

    test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    test = datasets.MNIST(os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_transform)


    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=args.batch_size * 2,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                drop_last=False)

    return train_loader, test_loader

class Solver(object):
    def __init__(self, args):
        self.args = args

        #self.train_loader, self.test_loader, self.transform = get_loader(args)
        self.train_loader, self.test_loader = get_loader(args)

        # ------ added by ldj
        # for i in self.args.num_examples:
        #     print("i=", i)
        #     current_num_examples = i
            # reduce the size of the training set
#            self.train_data = self.train_set.data[: self.args.num_examples]
            # self.train_loader.dataset.data = self.train_loader.dataset.data[: current_num_examples]
            #plot(self.train_data[0].cpu())
#            self.train_labels = self.train_set.targets[: self.args.num_examples]
            # self.train_loader.dataset.targets = self.train_loader.dataset.targets[: self.args.num_examples]
            # self.train_loader.dataset.targets = self.train_loader.dataset.targets[: current_num_examples]

            # self.train_set = Augmented_Data(self.train_data, self.train_labels, transform=self.transform)
            # self.train_loader = DataLoader(
            #     self.train_loader.dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
            # )

        # ---------

        self.model = VisionTransformer(
            n_channels=self.args.n_channels,
            embed_dim=self.args.embed_dim,
            n_layers=self.args.n_layers,
            n_attention_heads=self.args.n_attention_heads,
            forward_mul=self.args.forward_mul,
            image_size=self.args.image_size,
            patch_size=self.args.patch_size,
            n_classes=self.args.n_classes,
        )

        if self.args.is_cuda:
            print("Using GPU")
            self.model = self.model.cuda()
        else:
            print("Cuda not available.")

        # print('--------Network--------')
        # print(self.model)

        if args.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.model_path, "ViT_model.pt"))
            )

        self.ce = nn.CrossEntropyLoss()

    def test_dataset(self, loader):
        self.model.eval()

        actual = []
        pred = []

        for x, y in loader:
            if self.args.is_cuda:
                x = x.cuda()

            with torch.no_grad():
                logits = self.model(x)
            predicted = torch.max(logits, 1)[1]

            actual += y.tolist()
            pred += predicted.tolist()

        acc = accuracy_score(y_true=actual, y_pred=pred)
        cm = confusion_matrix(
            y_true=actual, y_pred=pred, labels=range(self.args.n_classes)
        )

        return acc, cm

    def test(self, train=True):  # ldj - stopped printing confusion matrix
        # added training accuracy

        train = True  # ldj   - don't know where train is set to False
        if train:
            acc, cm = self.test_dataset(self.train_loader)
            # print(f"Train acc: {acc:.2%}\nTrain Confusion Matrix:")
            # print(cm)
            print(f"    Train acc: {acc:.2%}", end="")

        acc, cm = self.test_dataset(self.test_loader)
        # print(f"Test acc: {acc:.2%}\nTest Confusion Matrix:")
        # print(cm)
        print(f"   Test acc: {acc:.2%}", end="")

        return acc

    def train(self,current_num_examples):

        print("current_num_examples=", current_num_examples)
        self.train_loader.dataset.targets = self.train_loader.dataset.targets[: current_num_examples]
        self.train_loader.dataset.data = self.train_loader.dataset.data[: current_num_examples]
        iter_per_epoch = len(self.train_loader)

        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=1e-3
        )
        linear_warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1 / self.args.warmup_epochs,
            end_factor=1.0,
            total_iters=self.args.warmup_epochs,
            last_epoch=-1,
            # verbose=True,
            verbose=False,
        )
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.args.epochs - self.args.warmup_epochs,
            eta_min=1e-5,
            # verbose=True,
            verbose=False,
        )

        best_acc = 0
        

        # the training loop
        for epoch in range(self.args.epochs):
            self.model.train()

            for i, (x, y) in enumerate(self.train_loader):
                # breakpoint()
                # plot([x[0], x[1]])
                # print(y)
                # plt.show()
                if self.args.is_cuda:
                    x, y = x.cuda(), y.cuda()

                logits = self.model(x)
                loss = self.ce(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # if i % 50 == 0 or i == (iter_per_epoch - 1):
            #     print(f'Ep: {epoch+1}/{self.args.epochs}, It: {i+1}/{iter_per_epoch}, loss: {loss:.4f}')
            print("epoch:", epoch + 1, end="")
            print(f"   loss: {loss:.4f}", end="")
            test_acc = self.test(
                train=((epoch + 1) % 25 == 0)
            )  # Test training set every 25 epochs
            # test_acc, cm = self.test_dataset(self.test_loader)
            if test_acc > best_acc:
                color = 'red'    
            else:
                color = 'white'
            best_acc = max(test_acc, best_acc)
            # print(colored(f"Test error: {100 * (1-test_acc):.2f}", color))
            print(colored(f"   Best test acc: {best_acc:.2%}", color))

            torch.save(
                self.model.state_dict(),
                os.path.join(self.args.model_path, "ViT_model.pt"),
            )

            if epoch < self.args.warmup_epochs:
                linear_warmup.step()
            else:
                cos_decay.step()
        # learning_curve.append([current_num_examples, best_acc])
        # print("learning_curve=", learning_curve)
