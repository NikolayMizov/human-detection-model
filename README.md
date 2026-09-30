# Human Detection Model

A small convolutional neural network that classifies a camera frame as **person** or **no person**. It was built as the safety signal for a semi-autonomous vacuum robot that has to work around people, with NVIDIA Jetson Nano / Xavier as the target hardware.

This is a solo learning and prototyping project.

## How it works

1. Images are loaded as grayscale and resized to 80 x 80 pixels.
2. A Keras `Sequential` CNN classifies them:
   - three `Conv2D` + `MaxPooling2D` blocks (32, 64, 128 filters, 3x3 kernels)
   - `Flatten`, a 128-unit dense layer, then a single sigmoid output
3. The dataset is split 80/20 into training and validation sets. Training uses Adam (learning rate 0.001), batch size 16 and 20 epochs.
4. The trained network is saved as `my_model.h5` (about 13 MB), small enough for a Jetson-class device.
5. A webcam module captures a single frame and releases the camera, and a small OpenCV window (`ui.py`) shows the live feed with a coloured overlay for the result.

## Files

| File | Purpose |
| --- | --- |
| `human_detection_model.py` | data loading, CNN definition, training and prediction functions |
| `train.py` | trains the model on a dataset folder and saves `my_model.h5` |
| `main.py` | loads `my_model.h5` and classifies one image |
| `take_photo.py` | captures a frame from the default webcam into `capture/` |
| `ui.py` | live camera preview with a result overlay (model call is still a stub) |
| `my_model.h5` | the trained model |

## Setup

Python 3.8 or newer.

```bash
pip install -r requirements.txt
```

**Classify an image**

```bash
mkdir capture
# put a test image at capture/image2.jpg, or capture one with take_photo.py
python main.py
```

**Train on your own data**

Put `.png` images in `Converted_Data/1/` (person) and `Converted_Data/0/` (no person), and any extra class folders under `data/`, then run:

```bash
python train.py
```

## Results

The model has not been benchmarked yet. There are no published accuracy figures and the training dataset is not included in this repository. Evaluating it on a held-out set is the first item below.

## Known issues and next steps

These come from reading the code and are the first things to fix:

- **Loss function.** The model has one sigmoid output but is compiled with `categorical_crossentropy`. `binary_crossentropy` is the correct loss for this setup.
- **Prediction decoding.** `predict_image` applies `argmax` to a single output, which always returns class 0. It should threshold the sigmoid value (for example at 0.5), and the labels in `main.py` should match the training labels (1 = person).
- **Preprocessing mismatch.** Inference divides pixels by 255, but training does not normalise. Both should use the same scaling.
- **Dataset layout.** `define_paths()` expects a `Converted_Data/` folder with `1/` (person) and `0/` (no person) subfolders, and `train_model` also scans the folder you pass it. Consolidating these two loaders would be cleaner.
- `ui.py` still uses a placeholder instead of calling the model.
- Add a validation report (accuracy, precision, recall, confusion matrix), then try a lightweight pre-trained detector such as MobileNet-SSD, and export to TensorFlow Lite for the Jetson.

## Tech stack

Python, TensorFlow / Keras, OpenCV, NumPy, scikit-learn, Matplotlib, Pillow.

## Author

Nikolay Mizov. Contact: nikolaymizov43@gmail.com
