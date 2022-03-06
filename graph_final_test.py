import argparse
import datetime
from pathlib import Path
import time
import yaml
import yamlloader
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from model_graph import load_model
from data_loader import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from pyramid import pyramid


def L1Loss(pred, gt, mask=None):
    assert (pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        distence = distence * mask
    return distence.sum() / mask.sum()

def focal_loss(pred, gt):
    return (-(1 - pred) * gt * torch.log(pred) - pred * (1 - gt) * torch.log(1 - pred)).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph Resnet landmark detection network")
    parser.add_argument("--tag", default='', help="name of the run")
    parser.add_argument("--config_file", default="config_graph.yaml", help="default configs")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)

    # Create Logger
    logger = get_mylogger()
    logger.info(config)
    torch.cuda.set_device(config['cuda_device'])

    # Create runs dir
    tag = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-') if args.tag == '' else args.tag
    runs_dir = "./runs/" + tag
    runs_path = Path(runs_dir)
    config['runs_dir'] = runs_dir

    dataset = Cephalometric(config['dataset_pth'], 'Train', augment=False, num_landmark=config['num_landmarks'],
                            resized=[2400, 1935])
    pnts = np.stack([dataset.__getitem__(i)[3] for i in range(dataset.__len__())])
    device = 'cuda'
    means = torch.tensor(pnts.mean(0, keepdims=True), device=device, dtype=torch.float32)
    stddevs = torch.tensor(pnts.std(0, keepdims=True), device=device, dtype=torch.float32)

    train_dataset = Cephalometric(config['dataset_pth'], 'Train', augment=True, num_landmark=config['num_landmarks'],
                                  resized=[2400, 1935])
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=config['num_workers'])
    val_dataset = Cephalometric(config['dataset_pth'], 'Test1', augment=False, num_landmark=config['num_landmarks'],
                                resized=[2400, 1935]) ## note!!!
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'],
                                shuffle=False, num_workers=config['num_workers'])

    net = load_model(config['levels'], config['model_path'], config['resume'])
    print(torch.cuda.is_available())
    net = net.cuda()
    net.eval()
    errors = []
    time_list = []
    n=1
    for img, _, _, landmarks in tqdm(val_dataloader):
        inputs_tensor, labels_tensor = img.cuda(), landmarks.cuda()
        pym = pyramid(inputs_tensor.unsqueeze(1), config['levels'])
        labels_tensor = labels_tensor.unsqueeze(1).to(torch.float32)
        guess = means.expand(config['batch_size'], 1, 19, 2)

        with torch.no_grad():
            start = time.time()
            tmp = 10.
            for j in range(config['test_iterations']):
                outputs = guess + net(pym, guess, train=False)  #
                loss_metric = F.mse_loss(outputs, labels_tensor, reduction='none')
                guess = outputs.detach()
            time_list.append(time.time() - start)
            error = loss_metric.detach().sum(dim=3).sqrt()
            errors.append(error)
        n += 1
    print("time elapsed",np.mean(time_list))
    errors = torch.cat(errors, 0).detach().cpu().numpy() / 2 * 192
    error = errors.mean()
    radii = [2, 2.5, 3, 4]
    inliers = np.array([np.count_nonzero(np.less_equal(errors, x)) for x in radii]) / (config['num_landmarks'] * len(errors)) * 100
    msg = '#correct_id <= {}: ({:.2f}%)\n'
    print(('').join(msg.format(i, x) for i, x in zip(radii, inliers)))
    logger.info("Test mean radial error {0:.4f} std error {1:.4f} ".format(errors.mean(), errors.std()))
