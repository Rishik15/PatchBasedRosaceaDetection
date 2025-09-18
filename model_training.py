import torch

import numpy as np
import matplotlib.pyplot as plt


import random
import argparse

from train import pipeline

new_seed = 42

g = torch.Generator()
g.manual_seed(new_seed)
torch.manual_seed(new_seed)
torch.cuda.manual_seed(new_seed)
torch.cuda.manual_seed_all(new_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(new_seed)
random.seed(new_seed)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--patch_size",
    type=str,
    choices=["small", "medium", "large"],
    required=True,
    help="Patch size must be one of: small, medium, large"
)
args = parser.parse_args()
patchsize = args.patch_size


selection_masks = {
    "F": [1, 0, 0, 0],
    "N": [0, 1, 0, 0],
    "LC": [0, 0, 1, 0],
    "RC": [0, 0, 0, 1],
    "F+N": [1, 1, 0, 0],
    "F+LC": [1, 0, 1, 0],
    "F+RC": [1, 0, 0, 1],
    "N+LC": [0, 1, 1, 0],
    "N+RC": [0, 1, 0, 1],
    "LC+RC": [0, 0, 1, 1],
    "F+N+LC": [1, 1, 1, 0],
    "F+N+RC": [1, 1, 0, 1],
    "F+LC+RC": [1, 0, 1, 1],
    "N+LC+RC": [0, 1, 1, 1],
    "F+N+LC+RC": [1, 1, 1, 1],
    "Full Face": None
}

val_rocs = []
test_rocs = []
per_patch_raw = []

for patchname, selection_mask in selection_masks.items():
    val_fpr, val_tpr, val_auc, test_fpr, test_tpr, test_auc, test_thresholds, y_true, y_score = pipeline(patchname, selection_mask, patchsize)
    val_rocs.append((patchname, val_fpr, val_tpr, val_auc))
    test_rocs.append((patchname, test_fpr, test_tpr, test_auc))
    per_patch_raw.append((patchname, test_fpr, test_tpr, test_thresholds, np.array(y_true), np.array(y_score)))

color_list = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#393b79', '#637939',
    '#8c6d31', '#843c39', '#7b4173', '#17a768'
]

plt.figure(figsize=(10, 6))
for idx, (patchname, fpr, tpr, auc_val) in enumerate(val_rocs):
    plt.plot(fpr, tpr, lw=2, label=f'{patchname} (AUC={auc_val:.2f})', color=color_list[idx % len(color_list)])
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Validation ROC curves for different patches configurations ({patchsize.capitalize()} Patch)')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(f'validation_rocs_{patchsize}.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for idx, (patchname, fpr, tpr, auc_test) in enumerate(test_rocs):
    plt.plot(fpr, tpr, lw=2, label=f'{patchname} (AUC={auc_test:.2f})', color=color_list[idx % len(color_list)])
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Test ROC curves for different patches configurations ({patchsize.capitalize()} Patch)')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(f'test_rocs_{patchsize}.png', dpi=300, bbox_inches='tight')

plt.show()