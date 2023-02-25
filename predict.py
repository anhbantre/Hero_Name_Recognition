import torch
from torch.utils.data import DataLoader

import argparse
import numpy as np

from utils.dataloader import HeroNameDatasetTest
from model.model import HeroModel
from utils.utils import set_seed, label_dict


device = "cuda" if torch.cuda.is_available() else "cpu"

def get_args():
    parser = argparse.ArgumentParser(description='Inference arguments for Hero Name Recognition model')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Model name from timm to be used as backbone')
    parser.add_argument('--data', '--d', type=str, default='data/test', help='Path to folder containing test data')
    parser.add_argument('--checkpoint', '--c', type=str, help='Path to checkpoint')
    parser.add_argument('--batch-size', '--b', type=int, default=64, help='Batch size')
    parser.add_argument('--label', '--lb', type=str, default='data/hero_names.txt', help='Path to label file')
    parser.add_argument('--output', '--o', type=str, default='output.txt', help='Path to output file')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # Initialize a result list
    result = []
    imgname_result = []

    # Load data
    test_dataset = HeroNameDatasetTest(data_path = args.data)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)

    # Load model
    model = HeroModel(args.backbone)
    model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(device)

    # Predict
    with torch.no_grad():
        model.eval()
        for data in test_dataloader:
            img, imgname = data
            img = img.to(device)
            output = model(img)       # (batch_sie, 64)
            # Conver tensor to array
            output = output.detach().cpu().numpy()
            # Merge output into the result
            result.extend(output)
            imgname_result.extend(imgname)
    
    # Get id_class of these results
    list_id_class = [np.argmax(i) for i in result]
    # Conver the results from id class to the hero names
    name_to_id, id_to_name = label_dict(args.label)
    list_hero_name = [id_to_name[id_class] for id_class in list_id_class]
    # Write into output.txt file
    with open(args.output, 'w') as file:
      for i in range(len(imgname_result)):
        file.write(f'{imgname_result[i]}\t{list_hero_name[i]}\n')
    print(f'Done! Check your output in {args.output} file')
