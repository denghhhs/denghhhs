import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import time
import os
import sys
import shlex
from datetime import datetime


class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]  

def resnet18():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.Dropout(0.2),
        nn.Linear(256, 10)
    )
    return model

def test_model(model_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    print(f'Model loaded on: {device}')
    return model, device

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2022,0.1994,0.2010])
    ])
    image =Image.open(image_path).convert('RGB')
    original_size = image.size
    tensor = transform(image).unsqueeze(0)
    return tensor, original_size

def predict(model, device, image_tensor):
    image_tensor = image_tensor.to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    elapsed_time = time.time() - start_time
    return probabilities.squeeze().cpu().numpy(), elapsed_time

def print_results(class_probs, elapsed_time, image_path, original_size):
    sorted_indices = class_probs.argsort()[::-1]
    print('\n' + '='*60)
    print(f'Input image: {os.path.basename(image_path)} (original size: {original_size[0]}x{original_size[1]})')
    print(f'time:{elapsed_time * 1000:.2f}ms')
    print('='*60)
    
    print(f"{'class':>15s} {'probability':>12s} bar")
    print('-'*60)
    
    for idx in sorted_indices:
        class_name = cifar10_classes[idx]
        prob = class_probs[idx]
        bar_length = int(prob*20)
        bar = '*' * bar_length
        print(f'{class_name:>15s} {prob:10.2%} /{bar}')
    print('='*60)
    top_idx = sorted_indices[0]
    print(f'\n predicted: **{cifar10_classes[top_idx]}** (confidence:{class_probs[top_idx]:.2%})\n')

    
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, f'result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    tee = Tee(log_path)
    sys.stdout = tee

    print('=' * 60)
    print(' cifar-10 image classifier (resnet-18)')
    print(f' Log file: {log_path}')
    print('=' * 60)

    model_path = os.path.join(script_dir, 'best_model.pth')

    if not os.path.exists(model_path):
       print(f'Error: Model file not found at {model_path}')
       tee.close()
       sys.exit(1)

    model, device = test_model(model_path)

    try:
        while True:
            print('\n' + '-' * 60)
            raw_input = input(
                'Enter image path(s), multiple images separated by comma/space.\n'
                'Drag & drop is supported. Type "quit" to exit:\n> '
            ).strip()
            if raw_input.lower() == 'quit':
                print('Exiting...')
                break
            if ',' in raw_input or ';' in raw_input:
                paths = [p.strip().strip('"').strip("'") for p in raw_input.replace(';', ',').split(',') if p.strip()]
            else:
                try:
                    paths = [p.strip('"').strip("'") for p in shlex.split(raw_input) if p.strip()]
                except ValueError:
                    paths = [p.strip().strip('"').strip("'") for p in raw_input.split() if p.strip()]

            if not paths:
                continue

            print(f'\nProcessing {len(paths)} image(s)...')
            success_count = 0
            for i, image_path in enumerate(paths):
                if len(paths) > 1:
                    print(f'\n[{i+1}/{len(paths)}] {os.path.basename(image_path)}')
                if not os.path.exists(image_path):
                    print(f'  [Skip] Path not found: {image_path}')
                    continue
                try:
                    image_tensor, original_size = preprocess_image(image_path)
                    class_probs, elapsed = predict(model, device, image_tensor)
                    print_results(class_probs, elapsed, image_path, original_size)
                    success_count += 1
                except Exception as e:
                    import traceback
                    print(f'  [Error] {image_path}: {e}')
                    traceback.print_exc()

            if len(paths) > 1:
                print(f'\nDone: {success_count}/{len(paths)} image(s) processed successfully.')
    finally:
        tee.close()
        sys.stdout = tee.stdout
            
if __name__ == '__main__':
    main()