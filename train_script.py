import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
import json
import os

class FLIRPersonDataset(Dataset):
    def __init__(self, images_dir, annotation_file):
        self.images_dir = images_dir
        
        with open(annotation_file) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        self.ann_map = {}
        for ann in self.annotations:
            if ann["category_id"] == 1:  #This is the ID used for humans in the dataset
                img_id = ann["image_id"]
                if img_id not in self.ann_map:
                    self.ann_map[img_id] = []
                self.ann_map[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        img_id = img_info["id"]

        boxes = []
        labels = []

        if img_id in self.ann_map:
            for ann in self.ann_map[img_id]:
                x, y, w, h = ann["bbox"]

                boxes.append([x, y, x + w, y + h])
                labels.append(1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])

        return F.to_tensor(img), target
    


def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    num_classes = 2  # background + person
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    return model



def train_model(model, dataloader, device, epochs=5):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            progress_bar.set_postfix(avg_loss=total_loss / (progress_bar.n + 1))

        print(f"Epoch {epoch+1} Total Loss: {total_loss:.4f}")


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # RGB dataset
    RGB_IMAGES_DIR = "FLIR_ADAS_v2/images_rgb_train"
    RGB_JSON = "FLIR_ADAS_v2/images_rgb_train/coco.json"

    # Thermal dataset
    THERMAL_IMAGES_DIR = "FLIR_ADAS_v2/images_thermal_train"
    THERMAL_JSON = "FLIR_ADAS_v2/images_thermal_train/coco.json"

    rgb_dataset = FLIRPersonDataset(RGB_IMAGES_DIR, RGB_JSON)
    thermal_dataset = FLIRPersonDataset(THERMAL_IMAGES_DIR, THERMAL_JSON)

    print("RGB samples:", len(rgb_dataset))
    print("Thermal samples:", len(thermal_dataset))

    # Combine the RGB and Thermal Datasets
    combined_dataset = ConcatDataset([rgb_dataset, thermal_dataset])

    print("Combined samples:", len(combined_dataset))

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    #Load previous model (trained for 10 epochs on RGB images)
    model = get_model()
    model.load_state_dict(torch.load("flir_person_detector.pth"))
    print("Loaded previous RGB model.")

    #Train for 5 more epochs with combined dataset
    train_model(model, combined_loader, device, epochs=5)

    # Save final RGB + Thermal model
    torch.save(model.state_dict(), "flir_person_detector_rgb_thermal.pth")
    print("Saved combined RGB + Thermal model.")
