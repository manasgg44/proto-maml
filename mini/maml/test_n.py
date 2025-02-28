#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 10:04:16 2022

@author: localadmin
"""

import torch
from tqdm import tqdm
import logging
import random
import numpy as np
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
import torch.nn.functional as F
from model1 import PrototypicalNetwork_new
from utils import get_accuracy
from torchmeta.utils.gradient_based import gradient_update_parameters
import os
logger = logging.getLogger(__name__)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def simple_triplet_loss(embeddings, targets, margin=1.0):
    n = embeddings.size(0)
    loss = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if torch.all(targets[i] == targets[j]) and torch.all(targets[i] != targets[k]):
                    # Calculate Euclidean distances
                    dist_ij = F.pairwise_distance(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
                    dist_ik = F.pairwise_distance(embeddings[i].unsqueeze(0), embeddings[k].unsqueeze(0))

                    # Triplet loss calculation
                    loss += F.relu(dist_ij - dist_ik + margin).pow(2)

    # Normalize the loss

    loss /= (n * (n - 1) * (n - 2) / 2)
    #mean_triplet_loss = torch.mean(loss.clone().detach().requires_grad_(True))
    mean_triplet_loss = torch.mean(loss)

    return mean_triplet_loss



def simple_contrastive_loss(embeddings, targets, margin=1.0):
    n = embeddings.size(0)
    loss = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = (embeddings[i] - embeddings[j]).pow(2).sum().sqrt()
            same_class =torch.all(targets[i] == targets[j])

            if same_class:

                loss += dist.pow(2)
            else:
                loss += F.relu(margin - dist).pow(2)

    loss /= (n * (n - 1) / 2)
    return loss

def find_tsne(embeddings, classes, save_path):
    tsne = TSNE(n_components=2, perplexity=3, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10,8))
    unique_classes = np.unique(classes)
    for cls in unique_classes :
        idx = np.where(classes == cls)
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=cls)
    plt.title('Visualization')
    plt.legend()
    plt.savefig(save_path)



def test(args):
    logger.warning('This script is an example to showcase the extensions and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested.')
    dataset = miniimagenet(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=True,
                       test_shots=15,
                       meta_test=True,
                       download=args.download)
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)

   
                                    
    

    model = PrototypicalNetwork_new(3,args.embedding_size,hidden_size=args.hidden_size)                                
    #                               
    #Save model
    if args.output_folder is not None:
        filename1 = os.path.join(args.output_folder, 'maml_omniglot_'
            '{0}shot_{1}way.th'.format(args.num_shots, args.num_ways))

    with open(filename1,'r+b') as f:
        state_dict_enc = torch.load(f, map_location='cuda:0')
        model.load_state_dict(state_dict_enc)

    model.to(device=args.device)

    model.eval()
    
    total_accuracy = torch.tensor(0., device = args.device)
    total_accuracy_original = torch.tensor(0., device = args.device)


    # Training loop
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= args.num_batches:
                break
            else:
            
            
            #with torch.no_grad():
                train_inputs, train_targets = batch['train']
                train_inputs = train_inputs.to(device=args.device)
                train_targets = train_targets.to(device=args.device)
                    
                test_inputs, test_targets = batch['test']
                test_inputs = test_inputs.to(device=args.device)
                test_targets = test_targets.to(device=args.device)
                accuracy = torch.tensor(0., device = args.device)
                for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
                    params=None
                    train_embeddings_original = model.forward_enc(train_input.unsqueeze(0))

                    for i in range(args.num_steps):
                        train_embeddings = model.forward_enc(train_input.unsqueeze(0), params=params)
                        tl = simple_triplet_loss(train_embeddings.squeeze(), train_target)
                        cl = simple_contrastive_loss(train_embeddings.squeeze(), train_target)
                        inner_loss = cl+tl
                        params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=args.step_size,
                                                    first_order= not args.first_order,params=params)

                    train_embeddings = model.forward_enc(train_input.unsqueeze(0), params=params)

                    if (batch_idx == 1 or batch_idx == 2 or batch_idx == 3) and task_idx ==1 :
                        find_tsne(train_embeddings_original.squeeze().clone().detach().cpu(), train_target.clone().detach().cpu(), f'train_embeddings_original_{batch_idx}.png')
                        find_tsne(train_embeddings.squeeze().clone().detach().cpu(), train_target.clone().detach().cpu(), f'train_embeddings_{batch_idx}.png')


                    prototypes = get_prototypes(train_embeddings, train_target.unsqueeze(0),dataset.num_classes_per_task)
                     
                    test_embeddings = model.forward_enc(test_input.unsqueeze(0), params=params)
                    acc, preds = get_accuracy(prototypes, test_embeddings, test_target.unsqueeze(0))
                    #if acc.item()<=0.5:
                    #print("Task accuracy",acc)
                    accuracy += acc
                total_accuracy += accuracy
                
        print(f"\nTotal accuracy ours {(total_accuracy/args.batch_size)/args.num_batches}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Prototypical Networks')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of steps in inner loop).')

    parser.add_argument('--folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first-order approximation of MAML.')
    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--step-size', type=float, default=0.4,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=16,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batches the prototypical network is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')
    args.device = torch.device('cuda')
    test(args)
    #seed = torch.randint(0,100000,(1,1)).squeeze()

    #filename = os.path.join(args.output_folder, 'seed.txt')
    #with open(filename, 'r') as f:
        #seed = int(f.read())
    
    #for i in range(10) :
        #seed = torch.randint(0,100000,(1,1)).squeeze()
        #seed_torch(seed)
        #test(args)
    #f.close()
