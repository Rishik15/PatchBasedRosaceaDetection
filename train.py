import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import numpy as np

from torchvision import transforms, datasets, models
from sklearn.metrics import roc_curve, auc
from PIL import Image

from utils import patch_configs, maskImages

new_seed = 42

g = torch.Generator()
g.manual_seed(new_seed)
torch.manual_seed(new_seed)
torch.cuda.manual_seed(new_seed)
torch.cuda.manual_seed_all(new_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(new_seed)

class ApplyMask:
    def __init__(self, mask_array):
        self.mask = mask_array

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)

        masked_np = img_np * self.mask

        masked_np = np.clip(masked_np, 0, 255).astype(np.uint8)

        return Image.fromarray(masked_np)

def train_model(model, criterion, optimizer, scheduler, num_epochs, model_save_name, dataloaders, dataset_sizes, device):
    since = time.time()

    ckdir = './checkpoints'
    os.makedirs(ckdir, exist_ok=True)
    best_model_params_path = os.path.join(ckdir, model_save_name)

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1) 

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  
                    loss = criterion(outputs, labels) 

                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            if phase == 'train':
                scheduler.step()

            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                print(f"Training Acc: {epoch_acc}, Loss: {epoch_loss}")
            else:
                print(f"Validation Acc: {epoch_acc}, Loss: {epoch_loss}")

            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model


def pipeline(patchname, booleanmask, patchsize):
    if booleanmask is None:
        data_transforms = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ])
            }
    else:
        points_config = patch_configs[patchsize]
        mask = maskImages(points_config, booleanmask[0], booleanmask[1], booleanmask[2], booleanmask[3])

        data_transforms = {
            'train': transforms.Compose([
                ApplyMask(mask),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]),
            'val': transforms.Compose([
                ApplyMask(mask),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]),
            'test': transforms.Compose([
                ApplyMask(mask),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        }

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    folder_map = {'train': 'cropped_train', 'val': 'cropped_val', 'test': 'cropped_test'}
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, folder_map[x]), data_transforms[x]) for x in ['train', 'val', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0, generator=g) for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(class_names)
    print("device: ", device)

    model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    model_ft = model_ft.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler,
        num_epochs=30,
        model_save_name="best_model.pt",
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device
    )

    model_ft.eval()

    val_running_corrects = 0
    val_y_true = []
    val_TP = val_TN = val_FP = val_FN = 0
    val_y_score_fraction = []

    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        with torch.no_grad():
            outputs = model_ft(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            val_running_corrects += torch.sum(preds == labels)

            for i in range(len(preds)):
                if preds[i] == 1 and labels[i] == 1:
                    val_TP += 1
                elif preds[i] == 0 and labels[i] == 0:
                    val_TN += 1
                elif preds[i] == 1 and labels[i] == 0:
                    val_FP += 1
                else:
                    val_FN += 1

            val_y_score_fraction.extend(probs.cpu().numpy().flatten())
            val_y_true.extend(labels.cpu().numpy().flatten())


    val_fpr, val_tpr, _ = roc_curve(val_y_true, val_y_score_fraction)
    val_auc = auc(val_fpr, val_tpr)

    print(f"[{patchname}] VAL Accuracy: {val_running_corrects.double() / dataset_sizes['val']:.4f}")
    print(f"[{patchname}] VAL AUC: {val_auc:.4f}")


    running_corrects = 0
    TP = TN = FP = FN = 0
    y_true = []
    y_score_fraction = []

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        with torch.no_grad():
            outputs = model_ft(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            running_corrects += torch.sum(preds == labels)

            for i in range(len(preds)):
                if preds[i] == 1 and labels[i] == 1:
                    TP += 1
                elif preds[i] == 0 and labels[i] == 0:
                    TN += 1
                elif preds[i] == 1 and labels[i] == 0:
                    FP += 1
                else:
                    FN += 1

            y_score_fraction.extend(probs.cpu().numpy().flatten())
            y_true.extend(labels.cpu().numpy().flatten())

    test_fpr, test_tpr, test_thresholds = roc_curve(y_true, y_score_fraction)
    test_auc = auc(test_fpr, test_tpr)

    print(f"[{patchname}] TEST Accuracy: {running_corrects.double() / dataset_sizes['test']:.4f}")
    print(f"[{patchname}] TEST AUC: {test_auc:.4f}")
    print("\n***************************************************************")
    print("****************************************************************\n")

    return val_fpr, val_tpr, val_auc, test_fpr, test_tpr, test_auc, test_thresholds, y_true, y_score_fraction