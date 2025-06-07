# programmer : Layan
# Date : 2/1/2024

import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 102),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    img = Image.open(image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = preprocess(img)
    img_np = np.array(img)
    img_np = img_np.transpose((0, 1, 2))
    return img_np

def predict(image_path, model, topk=5):
    img = process_image(image_path)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    img_tensor.unsqueeze_(0)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs, indices = torch.topk(torch.exp(output), topk)
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[idx.item()] for idx in indices[0]]
    return probs.numpy()[0], classes

def main():
    parser = argparse.ArgumentParser(description='Predict the flower class of an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the trained model checkpoint file')
    parser.add_argument('--topk', type=int, default=5, help='Number of top classes to predict')
    args = parser.parse_args()

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.image_path, model, topk=args.topk)
    
    class_names = [cat_to_name[cls] for cls in classes]
    print("Predictions:")
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {probs[i]*100:.2f}%")

if __name__ == "__main__":
    main()
