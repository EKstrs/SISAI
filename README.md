# SISAI


This repository contains the AI component MVP for our course project. The system can identify humans from both RGB and thermal images. 


## 1.How the AI Component Works

The core of this MVP is Faster R-CNN with a ResNet-50 backbone. 

The model was trained in two phases using FLIR ADAS v2 Dataset: https://oem.flir.com/solutions/automotive/adas-dataset-form

Phase 1: 10 epochs of training on 10318 RGB images to learn high-resolution human features. 
Phase 2: 5 epochs of fine tuning on combined dataset (21060 images) of RGB and thermal signatures to ensure robustness. 


Based on the teachers suggestions I decided to move away from using the methods listed in our deliverable 1 and used CNNs. 
15 epochs of training total is not a whole lot and based on the amount of data available I would have liked to run it for 60 epochs. I tried to achieve this using Puhti supercomputer but the dataset got messed up during the uploading to Puhti making it not possible to train the model. 



## 2.Required Libraries & Tools

```console
pip install torch torchvision pillow matplotlib tqdm

```


## 3.Project structure

train_script.py is for training the model it handles the loading the dataset and weight saving. 
testing.py The inference script that takes an image path and visualizes detection.
flir_person_detector.pth The earlier version with only 10 epochs of training with RGB images
flir_person_detector_rgb_thermal.pth The latest version with 15 epochs of training with 10 on RGB and 5 on combined dataset.


## 4.Instructions on how to run

Ensure the paths in the train_script.py are correct: 

```python
    # RGB dataset
    RGB_IMAGES_DIR = "FLIR_ADAS_v2/images_rgb_train"
    RGB_JSON = "FLIR_ADAS_v2/images_rgb_train/coco.json"

    # Thermal dataset
    THERMAL_IMAGES_DIR = "FLIR_ADAS_v2/images_thermal_train"
    THERMAL_JSON = "FLIR_ADAS_v2/images_thermal_train/coco.json"
```


Run the training script: 

```console
python train_script.py
```

Test the MVP (ensure path to the model is correct): 

```python
print("Loading weights...")
model.load_state_dict(torch.load("flir_person_detector_rgb_thermal.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded.")
```
```console
python testing.py path/to/your/image.jpg
```


## 5.Ethical considerations

Privacy: Data collection restrictions for collecting data in real world scenarios
Bias mitigation: By training on combined RGB and thermal the model reduces bias against different clothing types and environmental conditions
Explicability: Clear visual indicator are in place with bounding boxes around detected humans as well as a confidence score given for detection so human operators can make informed decisions. 




