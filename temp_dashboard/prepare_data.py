import os
import torch
import torchvision
import numpy as np
from PIL import Image
import json
from pathlib import Path
import sys

# Add demo repo to path to import the model
sys.path.append('/Users/erph/Documents/maee-demo-repo')
from models.baseline_cnn import BaselineCNN

# Constants
DEMO_REPO_PATH = os.getenv('MAEE_REPO_PATH', '/Users/erph/Documents/maee-demo-repo')
MODEL_PATH = os.path.join(DEMO_REPO_PATH, 'models/checkpoints/best_model.pth')
DATA_PATH = os.path.join(DEMO_REPO_PATH, 'data')
OUTPUT_DIR = 'preprocessed_data'

def prepare_data():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load MNIST test dataset
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaselineCNN(
        input_shape=[1, 28, 28],
        hidden_dims=[32, 64],
        num_classes=10
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Prepare regular examples
    regular_examples = []
    for i in range(len(test_dataset)):
        image, true_label = test_dataset[i]
        image = image.to(device)
        
        with torch.no_grad():
            pred = model.predict(image.unsqueeze(0)).item()
        
        if pred != true_label:
            # Save image
            img_array = (image.squeeze().cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(OUTPUT_DIR, f'regular_{i}.png')
            img.save(img_path)
            
            regular_examples.append({
                'image_path': img_path,
                'true_label': true_label,
                'pred_label': pred
            })
            
            if len(regular_examples) >= 10:
                break
    
    # Prepare adversarial examples
    adversarial_examples = []
    for i in range(len(test_dataset)):
        image, true_label = test_dataset[i]
        image = image.to(device)
        
        # Generate adversarial example
        epsilon = 0.3
        image_adv = image.clone().detach().requires_grad_(True)
        
        model.train()
        output = model(image_adv.unsqueeze(0))
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([true_label]).to(device))
        loss.backward()
        data_grad = image_adv.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        with torch.no_grad():
            pred = model.predict(perturbed_image.unsqueeze(0)).item()
        
        if pred != true_label:
            # Save image
            img_array = (perturbed_image.squeeze().cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(OUTPUT_DIR, f'adversarial_{i}.png')
            img.save(img_path)
            
            adversarial_examples.append({
                'image_path': img_path,
                'true_label': true_label,
                'pred_label': pred
            })
            
            if len(adversarial_examples) >= 10:
                break
    
    # Save metadata
    metadata = {
        'regular_examples': regular_examples,
        'adversarial_examples': adversarial_examples
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Preprocessed data saved to {OUTPUT_DIR}")
    print(f"Regular examples: {len(regular_examples)}")
    print(f"Adversarial examples: {len(adversarial_examples)}")

if __name__ == "__main__":
    prepare_data() 