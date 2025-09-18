## Patch Based Rosacea Detection

## Files
- `preprocessing/get_coordinates.py`: Lets you click on the eyes of uncropped pictures to get their coordinates  
- `preprocessing/crop_face.ipynb`: Crops and aligns faces to 130x150 using the eye coordinates  
- `utils.py`: Stores patch configurations (small, medium, large) and the masking function  
- `train.py`: Contains the training loop, pipeline function, and evaluation logic  
- `model_training.py`: Main entry point to train and evaluate models with different patch sizes (`small`, `medium`, `large`)  

---

## Installing the required Libraries
Run the following in the terminal:
```bash
pip install -r requirements.txt
```

## How to use

### 1. Preprocessing

Run `get_coordinates.py`:

```bash
python preprocessing/get_coordinates.py
```

→ Click **left eye** then **right eye** for each image  
→ Coordinates are saved to `coord.dat`

Run `crop_face.ipynb` to crop and align faces using `coord.dat`.

---

### 2. Model training (patch-based evaluation)

Run:

```bash
python model_training.py --patch_size small
python model_training.py --patch_size medium
python model_training.py --patch_size large
```
→ Trains models on selected patch size  
→ Evaluates on validation and test sets  
→ Saves ROC curves:
- `validation_rocs_<patchsize>.png`
- `test_rocs_<patchsize>.png`

→ Saves best model checkpoint in `checkpoints/best_model.pt`

---

### 3. Final testing

The script automatically reports:
- Validation Accuracy & AUC  
- Test Accuracy & AUC  
- ROC curves for each patch combination and full face
