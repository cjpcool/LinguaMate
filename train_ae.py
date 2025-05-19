import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import LatticeModulus
from modules.geom_autoencoder import GeomEncoder, GeomDecoder, GeomVAE
from torch_geometric.loader import DataLoader
from modules.submodules import LatticeNormalizer
from utils.ldm_utils import unpadding_max_num_node2graph
from visualization.vis import visualizeLattice
import numpy as np
import datetime
import wandb


MAX_NODE_NUM = 100
sample_max_num_nodes = 30
train_size = 8000
valid_size = 40
batch_size=512
condition_dim=12
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 5000
# device='cpu'
is_condition=True
is_disent_variational=True
disentagle_same_layer=True
load_name = 'vae_cond_128_beta001_dis_same_100_frac'
save_name = 'vae_cond_128_beta001_dis_same_100_frac_'
root = '.'
use_wandb=True
latent_dim=128



# --------------
#   Load data
# --------------
dataset = LatticeModulus('/home/jianpengc/datasets/metamaterial/LatticeModulus', file_name='data')
# dataset = LatticeModulus('D:\\项目\\Material design\\code_data\\data\\LatticeModulus',file_name='data_new')
indices = []
for i, data in enumerate(tqdm(dataset)):
    if data.num_atoms <= sample_max_num_nodes and data.num_edges <= sample_max_num_nodes * 2:
        indices.append(i)
dataset = dataset[indices]
print('All data size', len(dataset))
split_idx = dataset.get_idx_split(len(dataset), train_size=train_size, valid_size=valid_size, seed=42)
print(split_idx.keys())
print(dataset[split_idx['train']])
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
    split_idx['test']]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# --------------
#   Init model
# --------------
normalizer = LatticeNormalizer(train_dataset.lengths, train_dataset.angles)


model = GeomVAE(normalizer, max_node_num=MAX_NODE_NUM, latent_dim=latent_dim, edge_sample_threshold=0.5, is_variational=True,
                 is_disent_variational=is_disent_variational, is_condition=is_condition, condition_dim=condition_dim, disentangle_same_layer=disentagle_same_layer)
model = model.to(device)
normalizer = normalizer.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=.8)

print("Start training...")
# model.load_state_dict(torch.load(root+f'/checkpoints/{load_name}/best_ae_model.pt', map_location=device))
if use_wandb:
    wandb.init(
            entity='jianpengc',
            project='GeomVAE_VAE',
            name=save_name+'-'+datetime.datetime.now().strftime('%Y-%m-%d--%H:%M'),
        )

model.train()
model.train_model(train_loader, optimizer, device=device, num_epochs=epoch, beta=1, beta_geo=0.001, scheduler=scheduler,
                  checkpoint_dir=root+f'/checkpoints/{save_name}/', save_every=100, use_wandb=use_wandb)

# model.load_state_dict(torch.load(root+'/checkpoints/testae_128/best_ae_more.pt', map_location=device))

model.eval()

node_num_logit, node_num_pred, lengths_pred, angles_pred, latent = model.sample_lattice(num_samples=3, device=device)

lengths_pred, angles_pred = normalizer.denormalize(lengths_pred, angles_pred)

print("Sampled node numbers: ", node_num_pred)
print("Sampled lengths: ", lengths_pred)
print("Sampled angles: ", angles_pred)




