import os
import torch
from tqdm import tqdm
import logging

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes, prototypical_loss

from model import PrototypicalNetwork
from utils import get_accuracy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

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


def train(args):
    logger.warning('This script is an example to showcase the extensions and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested.')

    dataset = omniglot(args.folder,
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
    
    model = PrototypicalNetwork(1,
                                args.embedding_size,
                                hidden_size=args.hidden_size)


    if args.output_folder is not None:
        filename1 = os.path.join(args.output_folder, 'protonet_omniglot_'
                    '{0}shot_{1}way.pt'.format(args.num_shots, args.num_ways))
        with open(filename1,'r+b') as f:
            state_dict = torch.load(f, map_location='cuda:0')    
            model.load_state_dict(state_dict)

    model.to(device=args.device)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)
            train_embeddings = model(train_inputs)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)
            test_embeddings = model(test_inputs)
            
            if (batch_idx == 1 or batch_idx == 2 or batch_idx == 3) :
                find_tsne(train_embeddings[0].clone().detach().cpu(), train_targets[0].clone().detach().cpu(), f'train_embeddings_original_{batch_idx}.png')

            prototypes = get_prototypes(train_embeddings, train_targets,
                dataset.num_classes_per_task)
            #loss = prototypical_loss(prototypes, test_embeddings, test_targets)

            with torch.no_grad():
                accuracy = get_accuracy(prototypes, test_embeddings, test_targets)
                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))

            if batch_idx >= args.num_batches:
                break


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
    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')

    train(args)
