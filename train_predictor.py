import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import LatticeModulus
from modules.geom_autoencoder import GeomEncoder, GeomDecoder, GeomVAE, EncoderwithPredictionHead
from torch_geometric.loader import DataLoader
from modules.submodules import LatticeNormalizer
from utils.ldm_utils import unpadding_max_num_node2graph
from visualization.vis import visualizeLattice
import numpy as np
import datetime
import wandb


MAX_NODE_NUM = 100
sample_max_num_nodes = 100
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
save_name = 'vae_cond_128_beta001_dis_same_100_frac'
root = '.'
use_wandb=False
latent_dim=128



# --------------
#   Load data
# --------------
dataset = LatticeModulus('[your data path]/LatticeModulus', file_name='data')
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
normalizer = LatticeNormalizer()


AEmodel = GeomVAE(normalizer, max_node_num=MAX_NODE_NUM, latent_dim=latent_dim, edge_sample_threshold=0.5, is_variational=True,
                 is_disent_variational=is_disent_variational, is_condition=is_condition, condition_dim=condition_dim, disentangle_same_layer=disentagle_same_layer)


print("Start training...")
AEmodel.load_state_dict(torch.load(root+f'/checkpoints/{load_name}/best_ae_model.pt', map_location=device))
normalizer = AEmodel.normalizer

y_mean = train_dataset.y.mean(dim=0).to(device).view(1, -1)
y_std = train_dataset.y.std(dim=0).to(device).view(1, -1)
model = EncoderwithPredictionHead(AEmodel, latent_dim=latent_dim, condition_dim=condition_dim, y_mean=y_mean, y_std=y_std)
model.load_state_dict(torch.load(root+f'/checkpoints/{load_name}/best_predictor_model.pt', map_location=device))
model = model.to(device)
normalizer = normalizer.to(device)
if use_wandb:
    wandb.init(
            entity='',
            project='',
            name=save_name+'-'+datetime.datetime.now().strftime('%Y-%m-%d--%H:%M'),
        )


model.train()
optimizer = torch.optim.Adam(model.Predictor.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=.8)


best_loss = np.inf
for epoch in range(epoch):
    model.train()
    train_loss = 0.0
    for batch_data in tqdm(train_loader):

        z, coords, edge_index, batch, lengths, angles, num_atoms = batch_data.node_type, batch_data.frac_coords, batch_data.edge_index, batch_data.batch, batch_data.lengths, batch_data.angles, batch_data.num_atoms
        z = z.to(device)
        coords = coords.to(device)
        edge_index = edge_index.to(device)
        lengths = lengths.to(device)
        angles = angles.to(device)
        num_atoms = num_atoms.to(device)
        batch = batch.to(device)
        y = batch_data.y.to(device)
        batch_size = y.shape[0]

        lengths_normed, angles_normed = normalizer(lengths, angles)


        optimizer.zero_grad()
        y_pred = model(z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms, denormalize=False)
        # normalize y
        y = (y - y_mean) / y_std
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size

    train_loss /= len(train_loader.dataset)
    scheduler.step()

    # save the model
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), root+f'/checkpoints/{save_name}/best_predictor_model.pt')
        print(f"Best model saved at epoch {epoch} with loss {best_loss:.4f}")

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

    if use_wandb:
        wandb.log({"train_loss": train_loss})
        wandb.log({"epoch": epoch})





