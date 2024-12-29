# Copyright (c) OpenMMLab. All rights reserved.
import os
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
from enet import ENet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Resize
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor
import torch.nn.functional as F

seed = 42
NUM_CHANNELS = 3
NUM_CLASSES = 20
IMG_SIZE = (512, 1024)

# Initialize transforms
input_transform = Compose([Resize(IMG_SIZE, Image.BILINEAR), ToTensor()])
target_transform = Compose([Resize(IMG_SIZE, Image.NEAREST)])

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def initialize_model(weights_path, use_cpu, model_name):
    print ("Loading model: " + model_name)
    print ("Loading weights: " + weights_path)

    if model_name == "erfnet":
        model = ERFNet(NUM_CLASSES)
    elif model_name == "enet":
        model = ENet(NUM_CLASSES)

    if (not use_cpu):
        model = torch.nn.DataParallel(model).cuda()
    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements   
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model
    if model_name == "enet":
        model = load_my_state_dict(model.module, torch.load(weights_path))
    else:
        model = load_my_state_dict(model, torch.load(weights_path, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    return model
    
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).float()

def preprocess_mask(mask_path, transform):
    mask = Image.open(mask_path)
    return transform(mask)

def adjust_labels(path):
    pathGT = path.replace('images', 'labels_masks')
    if 'RoadObsticle21' in pathGT:
        pathGT = pathGT.replace('webp', 'png')
    if 'fs_static' in pathGT:
        pathGT = pathGT.replace('jpg', 'png')
    if 'RoadAnomaly' in pathGT:
        pathGT = pathGT.replace('jpg', 'png')

    mask = preprocess_mask(pathGT, target_transform)
    ood_gts = np.array(mask)

    if 'RoadAnomaly' in pathGT:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)
    if 'LostAndFound' in pathGT:
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

    if 'Streethazard' in pathGT:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts)
    return mask, ood_gts

def evaluate_model(model, input_paths, args):
    anomaly_score_list = []
    ood_gts_list = []
    for path in input_paths:
        print(path) 
        images = preprocess_image(path, input_transform)
        if not args.cpu:
            images = images.cuda()
        with torch.no_grad():
            result = model(images).squeeze(0)  
        if args.classifier == "void":
            anomaly_result = F.softmax(result, dim=0)[-1]
        else:
            # Discard the void class        
            result = result[:-1]
            if args.metric == "msp":
                anomaly_result = 1 - torch.max(F.softmax(result / args.temperature, dim=0), dim=0)[0]
            elif args.metric == "maxLogit":
                anomaly_result = - torch.max(result, dim=0)[0]
            elif args.metric == "maxEntropy":
                anomaly_result = torch.sum(
                    -F.softmax(result, dim=0) * F.log_softmax(result, dim=0),
                    dim=0
                ) / torch.log(torch.tensor(result.size(0), dtype=torch.float32))

        anomaly_result = anomaly_result.data.cpu().numpy()
        mask, ood_gts = adjust_labels(path)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()
    return np.array(ood_gts_list), np.array(anomaly_score_list)

def calculate_metrics(ood_gts, anomaly_scores):
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    return prc_auc, fpr

def extract_dataset_name(path):
    if 'RoadAnomaly21' in path:
        dataset_name = 'RoadAnomaly21'
    elif 'RoadAnomaly' in path:
        dataset_name = 'RoadAnomaly'
    elif 'RoadObsticle21' in path:
        dataset_name = 'RoadObsticle21'
    elif 'fs_static' in path:
        dataset_name = 'fs_static'
    elif 'FS_LostFound_full' in path:
        dataset_name = 'FS_LostFound_full'
    else:
        dataset_name = 'Unknown'
    return dataset_name

# Define color codes using ANSI escape sequences
class ConsoleColors:
    YELLOW = '\033[33;1m'
    PURPLE = '\033[35;1m'
    BLUE = '\033[34;1m'
    RED = '\033[31;1m'
    RESET = '\033[0m'

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    ) 
    parser.add_argument('--model', default="erfnet") 
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--metric', type=str, default='msp')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--classifier', type=str, default=None)
    args = parser.parse_args()

    # Load the model
    weightspath = args.loadDir + args.loadWeights
    model = initialize_model(weightspath, args.cpu, args.model)

    model.eval()
    
    # Load input images
    input_paths = glob.glob(os.path.expanduser(str(args.input[0])))

    # Evaluate model
    ood_gts, anomaly_scores = evaluate_model(model, input_paths, args)

    # Calculate metrics
    prc_auc, fpr = calculate_metrics(ood_gts, anomaly_scores)

    # Write the results
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')
    file.write( "\n\n")
    # Log the results
    dataset_name = extract_dataset_name(input_paths[0])
    print(f'{ConsoleColors.RED}Model: {ConsoleColors.RESET}{args.model.upper()}')
    print(f'{ConsoleColors.YELLOW}Dataset name: {ConsoleColors.RESET}{dataset_name}')
    print(f'{ConsoleColors.PURPLE}Metric: {ConsoleColors.RESET}{args.metric}')
    file.write('Weights: '+ args.loadDir + args.loadWeights)
    file.write('\nModel: ' + args.model.upper() )
    file.write('\nDataset name: '+dataset_name)
    file.write('\n\tMetric: '+args.metric)
    if args.metric == "msp":
        print(f'{ConsoleColors.PURPLE}Temperature: {ConsoleColors.RESET}{args.temperature}')
        file.write(' Temperature: '+ str(args.temperature))
    print(f'{ConsoleColors.BLUE}AUPRC score: {ConsoleColors.RESET}{prc_auc * 100.0} %')
    print(f'{ConsoleColors.BLUE}FPR@TPR95: {ConsoleColors.RESET}{fpr * 100.0} %')
    file.write(('\n\tAUPRC score: ' + str(prc_auc*100.0) + '   FPR@TPR95: ' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()