import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import LatticeModulus
from modules.geom_autoencoder import GeomEncoder, GeomDecoder, GeomVAE
from modules.submodules import LatticeNormalizer, DisentangledDenoise, DisentangledScore
# from modules.ldm.ddim import LatentDiffusionDDIM_GeomVAE, DenoiseTransformer, DenoiseGPS
from modules.ldm.ddim_disentangle import LatentDiffusionDDIM_GeomVAE
from modules.ldm.scheduler import DDPM_Scheduler, ScoreScheduler
from torch_geometric.loader import DataLoader
from diffusers import DDPMScheduler, DDIMScheduler, KarrasVeScheduler

from utils.ldm_utils import unpadding_max_num_node2graph
from visualization.vis import visualizeLattice
import numpy as np
import datetime
import wandb



MAX_NODE_NUM = 15
sample_max_num_nodes = 15
train_size = 1
valid_size = 1
batch_size=1
epoch = 20000
is_condition=True
condition_dim=12
is_diff_on_coords = True
use_wandb=False
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# save_name = 'ldm_disentangle_wovae_denoisediff_decp_cond_128'
save_name = 'disent_score_cond_128_vae'
# device='cpu'

root = '.'

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
dataset = dataset[indices][:4]
print("Total number of samples: ", len(dataset))

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


AEmodel = GeomVAE(normalizer, max_node_num=MAX_NODE_NUM, latent_dim=latent_dim, edge_sample_threshold=0.5, is_variational=True, is_disent_variational=True, is_condition=is_condition, condition_dim=condition_dim)
AEmodel = AEmodel.to(device)
AEmodel.load_state_dict(torch.load(root+'/checkpoints/fintune_vae_cond_6_beta0/best_ae_model.pt', map_location=device))

# DiffModel = DenoiseTransformer(
#     latent_dim=latent_dim,
#     time_emb_dim=latent_dim,
#     is_condition=is_condition,
#     condition_dim=condition_dim
# )
# DiffModel = DisenangledDenoise(latent_dim=latent_dim, time_emb_dim=128, is_condition=is_condition, condition_dim=condition_dim, is_diff_on_coords=is_diff_on_coords)
# print('DiffModel parameters:', sum(p.numel() for p in DiffModel.parameters() if p.requires_grad))
DiffModel = DisentangledScore(latent_dim=latent_dim, time_emb_dim=128, is_condition=is_condition, condition_dim=condition_dim, is_diff_on_coords=is_diff_on_coords)
# DiffModel.load_state_dict(torch.load(root+f'/checkpoints/{save_name}/best_diff_model.pt', map_location=device))
# diffscheduler = DDPM_Scheduler(
#     num_timesteps=1000,
#     beta_start=1e-4,
#     beta_end=2e-2,
#     schedule_type='cosine'
# )
# diffscheduler = DDPMScheduler(num_train_timesteps=500,
#                       beta_schedule="squaredcos_cap_v2",
#                       variance_type="fixed_small_log",
#                       prediction_type="v_prediction"
#                       )
# diffscheduler = DDPMScheduler(
#     num_train_timesteps=1000,
#     beta_start=1e-7,
#     beta_end=5e-4,
#     beta_schedule="sigmoid")
# diffscheduler = DDPMScheduler(
#     num_train_timesteps=1000,
#     beta_schedule="linear",   # 已内置于 diffusers
#     beta_start=1e-4,           # β_min
#     beta_end=1e-2,             # β_max
#     # sigmoid_temp=4.0,          # λ，决定过渡陡峭程度
#     prediction_type="v_prediction",
#     # rescale_betas_zero_snr=True  # 保证末步 SNR≈0
# )
# diffscheduler = DDPMScheduler(
#     num_train_timesteps = 1000,
#     beta_schedule="linear",   # 已内置于 diffusers
#     beta_start=1e-4,           # β_min
#     beta_end= 2e-2,             # β_max
#     # sigmoid_temp=4.0,          # λ，决定过渡陡峭程度
#     clip_sample=False,
#     prediction_type="epsilon",
#     # rescale_betas_zero_snr=True  # 保证末步 SNR≈0
# )


diffscheduler = ScoreScheduler(
        sigma_min = 0.002,
        sigma_max = 80.0,
        rho       = 7,              # log-linear exponent
        sigma_data= 0.5,            # latent std-dev
        prediction_type = "score",  # we'll output ∇log p
        device=device,
        num_train_timesteps=1000,
)

LDM = LatentDiffusionDDIM_GeomVAE(AEmodel, DiffModel, scheduler=diffscheduler, lr_vae=5e-4, lr_diffusion=1e-3, device=device, is_condition=is_condition,condition_dim=condition_dim, is_diff_on_coords=is_diff_on_coords)



# --------------
#   Train model
# --------------
print("Start training...")
if use_wandb:
    wandb.init(
            entity='jianpengc',
            project='GeomVAE_Diffusion',
            name=save_name+'-'+datetime.datetime.now().strftime('%Y-%m-%d--%H:%M'),
        )
LDM.train(train_loader, root+f'/checkpoints/{save_name}/', num_epochs=epoch, train_vae=False, use_wandb=use_wandb,
          lambda_e=1.0, lambda_p=1.0, lambda_s=1.0, beta_kl=1.0)


# --------------
#   Sampling
# --------------
lengths_pred, angles_pred, node_num_pred, coords_pred_list, edge_index_list, y_pred = LDM.sample_ddim(num_samples=3, ddim_steps=500, eta=0., is_recon=False)




val_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
batch_data = next(iter(val_loader))
(z, coords, edge_index, batch,
                 lengths, angles, num_atoms, num_edges) = (batch_data.node_type,batch_data.cart_coords, batch_data.edge_index,
                                                batch_data.batch, batch_data.lengths, batch_data.angles, batch_data.num_atoms, batch_data.num_edges)

z = z.to(device)
coords = coords.to(device)
condition = batch_data.y.to(device)
edge_index = edge_index.to(device)
batch = batch.to(device)
lengths = lengths.to(device)
angles = angles.to(device)
num_atoms = num_atoms.to(device)
num_edges = num_edges.to(device)
y_true = batch_data.y.to(device)

visualizeLattice(coords.cpu().numpy(), edge_index.cpu().numpy())



num_samples=1
LDM.geomvae.decoder.edge_sample_threshold = 0.5
with torch.no_grad():
    # encode
    lengths_normed, angles_normed = AEmodel.normalizer(lengths, angles)
    semantic_latent, geo_coords_latent = AEmodel.encoder(z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms,)
    node_num_logit, node_num_pred, lengths_pred, angles_pred, z_latent = AEmodel.sample_lattice(num_samples=num_samples, z_latent=semantic_latent, device=device, random_scale_latent=1.)

    lengths_pred, angles_pred, node_num_pred, coords_pred_list, edge_index_list, y_pred = LDM.sample_ddim(num_samples=num_samples, ddim_steps=50, eta=1e-5, z_lattice=z_latent, z_g=geo_coords_latent, batch=batch, is_recon=True)

visualizeLattice(coords_pred_list[0].cpu().numpy(), edge_index_list[0].cpu().numpy())
print('node_num_pred', node_num_pred)
lengths_pred, angles_pred = normalizer.denormalize(lengths_pred, angles_pred)
print('pred', lengths_pred, angles_pred)
print('gt', lengths, angles)
print('pred', y_pred, '; gt', y_true)

