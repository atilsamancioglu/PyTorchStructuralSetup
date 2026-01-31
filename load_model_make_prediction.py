import model_creation, setup_data
from torchvision import transforms
import torchvision
import torch

HIDDEN_UNITS = 32
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MODEL_SAVE_PATH = 'models/desert_classifier.pth'

train_dir = "data/desert101/train"
test_dir = "data/desert101/test"

data_transform = transforms.Compose([
      transforms.Resize((64, 64)),
      transforms.ToTensor()
    ])

train_dataloader, test_dataloader, class_names = setup_data.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)
# you can obviously omit the previous codes and just get the class names from main

loaded_model = model_creation.DesertClassifier(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    )

# Load model state dict
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))


# use loaded model to predict one image
from pathlib import Path
data_path = Path("data/")
online_image_path = data_path / "baklava-online.jpg"
single_image = torchvision.io.read_image(str(online_image_path)).type(torch.float32)
single_image = single_image / 255
single_image_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
])
single_image = single_image_transform(single_image)
single_image = single_image.unsqueeze(dim=0)
loaded_model.eval()
with torch.inference_mode():
    logits = loaded_model(single_image)
    probs = torch.softmax(logits, dim=1)
    pred_idx = probs.argmax(dim=1).item()

print("Predicted class:", class_names[pred_idx])