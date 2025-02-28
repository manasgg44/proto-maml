import os
import torch
from tqdm import tqdm
import logging
import random
import numpy as np
import torch.nn.functional as F
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from torchmeta.utils.gradient_based import gradient_update_parameters

from model2 import PrototypicalNetwork_new
from utils import get_accuracy

import gc
gc.collect()
torch.cuda.empty_cache()
logger = logging.getLogger(__name__)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
logger = logging.getLogger(__name__)

def train(args):
    logger.warning('This script is an example to showcase the extensions and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested.')

    dataset = miniimagenet(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=True,
                       test_shots=15,
                       meta_train=True,
                       download=args.download)
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)

    model_enc = PrototypicalNetwork_new(3,
                                args.embedding_size,
                                hidden_size=args.hidden_size)
                                
    #model_full = PrototypicalNetwork_new(3,
    #                            args.embedding_size,
    #                            args.num_ways,
    #                            hidden_size=args.hidden_size)
                                
    #model = torch.nn.DataParallel(model, device_ids=[0,1,2])
    model_enc.to(device=args.device)
    model_enc.train()
    #model_full.to(device=args.device)
    #model_full.train()

    optimizer_enc = torch.optim.Adam(model_enc.parameters(), lr=1e-3)
    #optimizer_full = torch.optim.Adam(model_full.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=2000, gamma=0.5)
    
    # Training loop
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= args.num_batches:
                break
            model_enc.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)
            train_embeddings = model_enc(train_inputs)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)
            test_embeddings = model_enc(test_inputs)

            prototypes = get_prototypes(train_embeddings, train_targets,
                dataset.num_classes_per_task)
            loss = prototypical_loss(prototypes, test_embeddings, test_targets)

            loss.backward()
            optimizer_enc.step()
            scheduler.step()

            #with torch.no_grad():
             #   accuracy = get_accuracy_proto(prototypes, test_embeddings, test_targets)
             #   pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))

            #model_full.zero_grad()

            #train_inputs, train_targets = batch['train']
            #train_inputs = train_inputs.to(device=args.device)
            #train_targets = train_targets.to(device=args.device)

            #test_inputs, test_targets = batch['test']
            #test_inputs = test_inputs.to(device=args.device)
            #test_targets = test_targets.to(device=args.device)

            #outer_loss = torch.tensor(0., device=args.device)
            #accuracy = torch.tensor(0., device=args.device)
            #for task_idx, (train_input, train_target, test_input,
            #        test_target) in enumerate(zip(train_inputs, train_targets,
            #        test_inputs, test_targets)):
            #    train_logit = model_full(train_input)
            #    inner_loss = F.cross_entropy(train_logit, train_target)

            #    model_full.zero_grad()
            #    params = gradient_update_parameters(model_full,
            #                                        inner_loss,
            #                                        step_size=args.step_size,
            #                                        first_order=args.first_order)

            #    test_logit = model_full(test_input, params=params)
             #   outer_loss += F.cross_entropy(test_logit, test_target)

             #   with torch.no_grad():
             #       accuracy += get_accuracy_anil(test_logit, test_target)

            #outer_loss.div_(args.batch_size)
            #accuracy.div_(args.batch_size)

            #outer_loss.backward()
            #optimizer_full.step()

    # Save model
    if args.output_folder is not None:
        #filename1 = os.path.join(args.output_folder, 'protonet_full_miniimagenet_'
        #    '{0}shot_{1}way.pt'.format(args.num_shots, args.num_ways))
        filename2 = os.path.join(args.output_folder, 'protonet_enc_miniimagenet_'
            '{0}shot_{1}way.pt'.format(args.num_shots, args.num_ways))

        #with open(filename1, 'wb') as f:
        #    state_dict = model_full.state_dict()
        #    torch.save(state_dict, f)
 
        with open(filename2, 'wb') as f:
            state_dict = model_enc.state_dict()
            torch.save(state_dict, f)
 

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Prototypical Networks')

    parser.add_argument('--folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

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
    args.step_size = 0.4
    args.first_order = True
    args.device = torch.device('cuda:1' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')

    #seed = torch.randint(0,100000,(1,1)).squeeze()
    #seed_torch(seed)
    train(args)

    #filename = os.path.join(args.output_folder, 'seed.txt')
    #with open(filename, 'w') as f:
    #    f.write(f'{seed}')
    #f.close()
