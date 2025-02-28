import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters
from torchmeta.utils.prototype import get_prototypes, prototypical_loss

from model1 import PrototypicalNetwork_new
from utils import get_accuracy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

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
    mean_triplet_loss = torch.mean(torch.tensor(loss))

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



def train(args):
    logger.warning('This script is an example to showcase the MetaModule and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested. For a better tested implementation of '
                   'Model-Agnostic Meta-Learning (MAML) using Torchmeta with '
                   'more features (including multi-step adaptation and '
                   'different datasets), please check `https://github.com/'
                   'tristandeleu/pytorch-maml`.')

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

    model = PrototypicalNetwork_new(1,
                                       args.embedding_size,
                                       hidden_size=args.hidden_size)
    model.to(device=args.device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
                params=None
                #model.zero_grad()
                
                #task_path = os.makedirs(os.path.join(os.getcwd(),logs,task_idx))
                #print(task_path)
 
                train_embeddings_original = model.forward_enc(train_input, params=params)
                for i in range(args.num_steps):
                    train_embeddings = model.forward_enc(train_input,params=params)
                    print(train_embeddings.shape)
                    inner_loss = simple_contrastive_loss(train_embeddings, train_target)+simple_triplet_loss(train_embeddings, train_target)

                    params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=args.step_size,
                                                    first_order= not args.first_order,params=params)

                train_embeddings = model.forward_enc(train_input,params=params)
                if (batch_idx == 50 or batch_idx == 99) and task_idx ==1 :
                    #find_tsne(train_embeddings_original.clone().detach().cpu(), train_target.clone().detach().cpu(), f'train_embeddings_original_{batch_idx}.png')
                    find_tsne(train_embeddings.clone().detach().cpu(), train_target.clone().detach().cpu(), f'train_embeddings_{batch_idx}.png')
                
                #print(train_embeddings.shape, train_embeddings.unsqueeze(0).shape, train_target.unsqueeze(0).shape)
                prototypes = get_prototypes(train_embeddings.unsqueeze(0), train_target.unsqueeze(0),dataset.num_classes_per_task)
                #loss = prototypical_loss(prototypes, test_embeddings, test_targets)
                test_embeddings = model.forward_enc(test_input, params=params)
                outer_loss += prototypical_loss(prototypes,test_embeddings.unsqueeze(0), test_target.unsqueeze(0))
		#loss.backward()
                #optimizer.step()
                with torch.no_grad(): 
                    acc, preds = get_accuracy(prototypes, test_embeddings.unsqueeze(0), test_target.unsqueeze(0))
                #print(train_input.shape)
                #print(train_input.shape)
                    if acc.item()<=0.5:
                        print("Task accuracy",acc)
                        print("Predictions", preds)
                    accuracy += acc

            outer_loss.div_(args.batch_size)
            accuracy.div_(args.batch_size)
            outer_loss.backward()
            meta_optimizer.step()
            model.zero_grad()

            #print(accuracy.item())
            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx >= args.num_batches:
                break

    # Save model
    if args.output_folder is not None:
        filename = os.path.join(args.output_folder, 'maml_omniglot_'
            '{0}shot_{1}way.th'.format(args.num_shots, args.num_ways))
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Pro-Model-Agnostic Meta-Learning (MAML)')

    parser.add_argument('--folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of steps in inner loop).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.4,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=16,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batches the model is trained over (default: 500).')
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
