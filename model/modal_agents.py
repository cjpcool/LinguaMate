import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')


import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI
from google import genai
from google.genai import types
import re
import numpy as np

from utils.llm_utils import parse_graph, graph_to_text, construct_supervisor_input, classify_nodes_with_geometry, find_closest_structure
from modules.geom_autoencoder import GeomEncoder, GeomDecoder, GeomVAE, EncoderwithPredictionHead
from modules.submodules import LatticeNormalizer, DisentangledDenoise
# from modules.ldm.ddim import LatentDiffusionDDIM_GeomVAE, DenoiseTransformer, DenoiseGPS
from modules.ldm.ddim_disentangle import LatentDiffusionDDIM_GeomVAE
from modules.ldm.scheduler import DDPM_Scheduler
from torch_geometric.loader import DataLoader
from visualization.vis import visualizeLattice
from model.symb_logi_ops import pairwise_l2, sinkhorn, kl_gaussian_diag, node_logic_loss, poe, qoe, mixture_mm, \
    nodes_to_keep, gather_nodes_and_edges, sinkhorn_log, remove_close_nodes, check_all_nodes_connected, node_logic_keep_original, gaussian_negation
import copy

from modules.submodules import LatticeNormalizer

from datasets import LatticeModulus
from tqdm import tqdm
from utils.mat_utils import frac_to_cart_coords

import warnings
warnings.filterwarnings("ignore")


INSTRUCTIONS_TRANSLATOR = """
You are a metamaterial scientist specializing in structural design and mechanical characterization. You have expert knowledge of canonical 3‑D architectures (octet‑truss, BCC, SC, Kelvin cell, Diamond, TPMS, etc.) and their typical mechanical responses.

Task
-----

Given a single *design requirement*, locate in the metamaterial literature the simplest existing basic substructure (motif) that meets the requirement. Describe this motif as an undirected graph:

- **Nodes** — 3‑D fractional coordinates.  
- **Edges** — pairs of node indices.

Output the graph in a code block exactly as shown below; provide **no additional text, commentary, or reasoning**.

Input
-----

Design prompt (free text).

Output format
-------------
~~~
Node number: <N>
Node coordinates:
(x0, y0, z0)
...
(xN-1, yN-1, zN-1)

Edges:
(i0, j0)
...
(iM-1, jM-1)
~~~

Constraints
-----------

- Keep the motif as simple as possible (minimal nodes/edges).  
- Return the output *only* in the specified layout and code‑block format.  
- Do not include any other information.
"""

INSTRUCTIONS_SUPERVISOR = """
You are a metamaterial scientist specializing in structural design and mechanical characterization. You are fluent in the geometry and typical property ranges of canonical 3‑D architectures such as octet‑truss, BCC, SC, Kelvin cell, Diamond, and TPMS families.

Task
-----

Given one *design prompt* and a corresponding *metamaterial structure* with its mechanical properties (Young's modulus, Shear modulus, and Poisson's ratio), output:

1. **Score** — a single real number in **[0, 1]** evaluating how well the structure and the provided properties (if have) fulfills the design prompt (0 = poor, 1 = perfect).
2. **Improved Prompt** — the original design requirement rewritten with clearer, more specific engineering details.
3. **Predicted Properties** — your best estimate of the structure’s mechanical response:  
   • *Young’s modulus* (Ex, Ey, Ez)  
   • *Shear modulus* (Gxy, Gyz, Gzx)  
   • *Poisson ratio* (νxy, νyx, νxz, νzx, νyz, νzy)

Input Format
------------

Prompt: <free‑text design requirement>

Structure:
~~~
Node number: <N>
Node coordinates:
(x1, y1, z1)
...
(xN, yN, zN)

Edges:
(i0, j0)
...
(iM, jM)
~~~

Lattice lengths: [a, b, c]
Lattice angles: [α, β, γ]

Properties:
Young's modulus: [Ex, Ey, Ez]
Shear modulus: [Gxy, Gyz, Gzx]
Poisson ratio: [νxy, νyx, νxz, νzx, νyz, νzy]

Output Format
-------------

Score: <float in [0,1]>
Improved Prompt: <refined design requirement>
Improved Properties:
Young's modulus: [Ex, Ey, Ez]
Shear modulus: [Gxy, Gyz, Gzx]
Poisson ratio: [νxy, νyx, νxz, νzx, νyz, νzy]

Constraints
-----------

- Return *only* the fields specified above, in the exact order and layout.  
- Provide no additional commentary, explanations, or reasoning steps.
- For mechanical properties, their value has scales: Ex, Ey, Ez in [0, 1e-2]; Gxy, Gyz, Gzx in [0, 1e-2]; νxy, νyx, νxz, νzx, νyz, νzy in [-20, +20].
"""

MAX_NODE_NUM = 100

class MetamatGenAgents(nn.Module):
    def __init__(self, root='../', ckpt_dir='/checkpoints/testae/', device='cuda', backbone='vae', designer_client='gpt-4o-mini', supervisor_client='gpt-4.1', api_key='', \
                 evaluation_threshold=0.5, max_evaluate_num=10, Generator=None, latent_dim=128):
        super(MetamatGenAgents, self).__init__()
        self.designer_client = designer_client
        self.supervisor_client = supervisor_client
        self.api_key = api_key
        self.backbone = backbone
        self.device=device
        self.latent_dim = latent_dim
        self.condition_dim = 12
        self.root = root
        self.ckpt_dir = ckpt_dir
        self.normalizer = LatticeNormalizer()
        self.evaluation_threshold = evaluation_threshold
        self.max_evaluate_num = max_evaluate_num

        self.Translator = self.load_translator()
        self.Generator = self.load_generator() if Generator is None else Generator
        self.Predictor = None
        self.Supervisor = self.load_supervisor()

    def init_normalizer(self, train_dataset):
            self.normalizer.fit(train_dataset.lengths, train_dataset.angles)


    def translate(self, input_text):
        while True:

            if 'gpt' in self.designer_client or 'o4-mini' in self.designer_client:
                response = self.Translator.responses.create(
                    model=self.designer_client,
                    instructions=INSTRUCTIONS_TRANSLATOR,
                    input=input_text,
                ) 
                output = response.output_text
            elif 'gemini' in self.designer_client:
                response = self.Translator.models.generate_content(
                model=self.designer_client,
                config=types.GenerateContentConfig(
                    system_instruction=INSTRUCTIONS_TRANSLATOR),
                contents=input_text
                )
                output = response.text

            try:
                z, coords, edge_index, batch, lengths, angles, num_atoms = parse_graph(output)
            except ValueError as e:
                print(f"Error parsing graph: {e}")
                print(f"Output: {output}")
                # If parsing fails, try again with a different input
                continue
            break

        return z, coords, edge_index, batch, lengths, angles, num_atoms

    def load_translator(self):
        if 'gpt' in self.designer_client or 'o4-mini' in self.designer_client:
            client = OpenAI(
                api_key=self.api_key
                )
        elif 'gemini' in self.designer_client:
            client = genai.designer_client(api_key=self.api_key)
        return client

    def evaluate(self, input_text):
        if 'gpt' in self.supervisor_client or 'o4-mini' in self.supervisor_client:
            response = self.Supervisor.responses.create(
                model=self.supervisor_client,
                instructions=INSTRUCTIONS_SUPERVISOR,
                input=input_text,
            ) 
            output = response.output_text
        elif 'gemini' in self.supervisor_client:
            response = self.Supervisor.models.generate_content(
            model=self.supervisor_client,
            config=types.GenerateContentConfig(
                system_instruction=INSTRUCTIONS_SUPERVISOR),
            contents=input_text
            )
            output = response.text

        score = output.split("Score: ")[1].split("\n")[0]
        score = float(score)
        improved_prompt = output.split("Improved Prompt: ")[1].split("\n")[0]

        young = output.split("Young's modulus: ")[1].split("\n")[0]
        young = torch.from_numpy(np.array([float(v) for v in re.findall(r"[0-9]*\.?[0-9]+", young)], dtype=float))
        shear = output.split("Shear modulus: ")[1].split("\n")[0]
        shear = torch.from_numpy(np.array([float(v) for v in re.findall(r"[0-9]*\.?[0-9]+", shear)], dtype=float))
        poisson = output.split("Poisson ratio: ")[1].split("\n")[0]
        poisson = torch.from_numpy(np.array([float(v) for v in re.findall(r"[0-9]*\.?[0-9]+", poisson)], dtype=float))

        properties = torch.cat([young, shear, poisson], dim=-1)


        return score, improved_prompt, properties

    def load_generator(self, latent_dim=128, condition_dim=12, edge_sample_threshold=0.5):
        AEmodel = GeomVAE(self.normalizer, max_node_num=MAX_NODE_NUM, latent_dim=latent_dim, edge_sample_threshold=0.5, is_variational=True, is_disent_variational=True, is_condition=True, condition_dim=condition_dim,
                  disentangle_same_layer=True)    
        AEmodel = AEmodel.to(self.device)
        self.normalizer = AEmodel.normalizer    
        AEmodel.load_state_dict(torch.load(os.path.join(self.root, self.ckpt_dir, 'best_ae_model.pt'), map_location=self.device))
        AEmodel.eval()
        if self.backbone == 'LDM':
            DiffModel = DisentangledDenoise(latent_dim=latent_dim, time_emb_dim=128, is_condition=False, condition_dim=condition_dim)

            DiffModel.load_state_dict(torch.load(os.path.join(self.root,self.ckpt_dir, 'best_diff_model.pt'), map_location=self.device))
            diffscheduler = DDPM_Scheduler(
                num_timesteps=1000,
                beta_start=1e-4,
                beta_end=2e-3,
                schedule_type='linear'
            )

            LDM = LatentDiffusionDDIM_GeomVAE(AEmodel, DiffModel, scheduler=diffscheduler, lr_vae=1e-3, lr_diffusion=1e-3, device=self.device, is_condition=False,condition_dim=condition_dim)
            LDM.eval()
            return LDM
        return AEmodel


    def generate_from_scratch(self, gen_num=1, ddim_steps=50, eta=1e-8):
        lengths_pred, angles_pred, node_num_pred, coords_pred_list, edge_index_list, y_pred = self.LDM.sample_ddim(num_samples=gen_num, ddim_steps=ddim_steps, eta=eta, is_recon=False)
        # visualizeLattice(coords_pred_list[i].cpu().numpy(), edge_index_list[i].cpu().numpy())
        # print(normalizer.denormalize(lengths_pred[i], angles_pred[i]))
        return lengths_pred, angles_pred, node_num_pred, coords_pred_list, edge_index_list, y_pred

    def load_supervisor(self):
        self.Predictor = EncoderwithPredictionHead(GeomVAE(self.normalizer, max_node_num=MAX_NODE_NUM, latent_dim=self.latent_dim, edge_sample_threshold=0.5, is_variational=True,
                                                            is_disent_variational=True, is_condition=True, condition_dim=self.condition_dim,disentangle_same_layer=True),
                                                             self.latent_dim, self.condition_dim)
        self.Predictor.load_state_dict(torch.load(os.path.join(self.root, self.ckpt_dir, 'best_predictor_model.pt'), map_location=self.device))
        self.Predictor = self.Predictor.to(self.device)
        self.Predictor.eval()
        if 'gpt' in self.supervisor_client or 'o4-mini' in self.supervisor_client:
            client = OpenAI(
                api_key=self.api_key
                )
        elif 'gemini' in self.supervisor_client:
            client = genai.Client(api_key=self.api_key)
        return client


    def reconstruct_from_input(self, node_type, frac_coords, edge_index, batch, lengths, angles, num_atoms, num_edges, num_samples):
        (z, coords, edge_index, batch,
                 lengths, angles, num_atoms, num_edges) = (node_type,frac_coords, edge_index,
                                                batch, lengths, angles, num_atoms, num_edges)
        z = z.to(self.device)
        coords = coords.to(self.device)
        # condition = y.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)
        lengths = lengths.to(self.device)
        angles = angles.to(self.device)
        num_atoms = num_atoms.to(self.device)
        num_edges = num_edges.to(self.device)



        with torch.no_grad():
            # encode
            lengths_normed, angles_normed = self.Generator.geomvae.normalizer(lengths, angles)
            semantic_latent, geo_coords_latent = self.Generator.geomvae.encoder(z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms,)
            node_num_logit, node_num_pred, lengths_pred, angles_pred, z_latent = self.Generator.geomvae.sample_lattice(num_samples=num_samples, z_latent=semantic_latent, device=self.device, random_scale_latent=1.)

            lengths_pred, angles_pred, node_num_pred, coords_pred_list, edge_index_list, y_pred = self.Generator.sample_ddim(num_samples=num_samples, ddim_steps=50, eta=1e-5, z_lattice=z_latent, z_g=geo_coords_latent, batch=batch, is_recon=True)

        print(coords_pred_list[0].shape)
        # visualizeLattice(coords_pred_list[0].cpu().numpy(), edge_index_list[0].cpu().numpy())
        print('node_num_pred', node_num_pred)
        lengths_pred, angles_pred = self.normalizer.denormalize(lengths_pred, angles_pred)

        lengths_pred, angles_pred, node_num_pred, coords_pred_list, edge_index_list, y_pred = self.Generator.sample_ddim(num_samples=3, ddim_steps=50, eta=1e-8, is_recon=False)
        # i=0
        # visualizeLattice(coords_pred_list[i].cpu().numpy(), edge_index_list[i].cpu().numpy())
        # print(normalizer.denormalize(lengths_pred[i], angles_pred[i]))


    def collaborate_between_agents_13(self, input_text, visualize_results=False):
        """
        Generate a structure from the input text using the translator and evaluate it using the supervisor.
        Args:
            input_text (str): The input text describing the desired structure.
        Returns:
            tuple: A tuple containing the generated structure's parameters (z, coords, edge_index, batch, lengths, angles, num_atoms).
        """
        init_prompt = input_text
        evaluation_result = 0
        eval_num=0
        while evaluation_result < self.evaluation_threshold:
            # 1. Translate the input text to a graph structure
            z, coords, edge_index, batch, lengths, angles, num_atoms = self.translate(input_text)
            if (edge_index is not None) and (len(edge_index) > 0) and (edge_index.max() > coords.shape[0]-1):
                # raise ValueError(f"Edge index {edge_index.max()} is out of bounds for coords shape {coords.shape[0]}")
                print(f"Edge index {edge_index.max()} is out of bounds for coords shape {coords.shape[0]}")
                continue
            # Property prediction
            z, coords, edge_index, batch, lengths, angles, num_atoms = z.to(self.device), coords.to(self.device), edge_index.to(self.device), \
                batch.to(self.device), lengths.to(self.device), angles.to(self.device), num_atoms.to(self.device)
            lengths_normed, angles_normed = self.Generator.normalizer(lengths, angles)
            pred_y = self.Predictor(z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms, denormalize=True)
            properties = {}
            properties["Young's modulus"] = pred_y[:, :3]
            properties["Shear modulus"] = pred_y[:, 3:6]
            properties["Poisson ratio"] = pred_y[:, 6:]


            structure_prompt  = graph_to_text(coords, edge_index)
            supervisor_input_text = construct_supervisor_input(prompt=init_prompt, structure=structure_prompt, properties=properties)
            # 2. Evaluate the generated structure against the original prompt
            evaluation_result, improved_prompt, improved_properties = self.evaluate(supervisor_input_text)
            print(f"\n\nDesign trial {eval_num+1}\nPrompt: {input_text}\nEvaluation_result: {evaluation_result}\nImproved Prompt: {improved_prompt}")
            input_text = init_prompt+improved_prompt
            if visualize_results:
                visualizeLattice(coords.cpu().numpy(), edge_index.cpu().numpy())

            eval_num += 1
            if eval_num >= self.max_evaluate_num:
                break
        return z, coords, edge_index, batch, lengths, angles, num_atoms, evaluation_result

    def collaborate_between_agents_23(self, dataset, scaffold, input_text, logic_mode='union', num_steps_lattice=30, num_steps_geo=300, \
                                      optimize_network_params=False, fusion_thresh=0.1, mix_lam=0.5, condition=None, visualize_results=False, eval_thr=None):
        """
        Generate a structure from the input text using the translator and evaluate it using the supervisor.
        Args:
            input_text (str): The input text describing the desired structure.
        Returns:
            tuple: A tuple containing the generated structure's parameters (z, coords, edge_index, batch, lengths, angles, num_atoms).
        """
        evaluation_result = 0
        eval_num=0
        dataiter = iter(DataLoader(dataset, batch_size=1, shuffle=True))
        improved_properties = None
        batch_data = next(dataiter)
        if eval_thr is None:
            eval_thr = self.evaluation_threshold
        while evaluation_result < eval_thr:
            print(f"\nGeneratioin Trial:{eval_num+1}")
            if True:
                if improved_properties is not None:
                    if len(improved_properties) == 12:
                        new_data_idx, _ = find_closest_structure(dataset, improved_properties, )
                        batch_data =next(iter(DataLoader(dataset[new_data_idx], batch_size=1, shuffle=True)))
                    else:
                        print(f"Improved properties {improved_properties.shape} is not 12D, using the original data")
                        batch_data = next(dataiter)
                else:
                    batch_data = next(dataiter)
            else:
                new_data_idx, _ = find_closest_structure(dataset, condition)
                batch_data =next(iter(DataLoader(dataset[new_data_idx], batch_size=1, shuffle=True)))
            # 1. Generate a structure using generator
            final_node_num, lengths, angles, coords, edge_index = \
                self.collaborate_between_agents_12(batch_data, scaffold, logic_mode=logic_mode, num_steps_lattice=num_steps_lattice, num_steps_geo=num_steps_geo,\
                                                    optimize_network_params=optimize_network_params, thresh=fusion_thresh, mix_lam=mix_lam, condition=condition)

            try:
                z = classify_nodes_with_geometry(coords, edge_index)
                z = torch.LongTensor(torch.argmax(z,dim=-1)+1).to(self.device)
            except:
                Warning("Error computing geometry information, using default node labels")
                z = torch.ones(coords.shape[0], dtype=torch.long)
            # 2. Property prediction
            z, coords, edge_index, lengths, angles, num_atoms = z.to(self.device), coords.to(self.device), edge_index.to(self.device), \
                lengths.to(self.device), angles.to(self.device), torch.LongTensor([final_node_num]).to(self.device)
            lengths_normed, angles_normed = self.Generator.normalizer(lengths, angles)
            with torch.no_grad():
                batch = torch.zeros(coords.shape[0], dtype=torch.long, device=self.device)
                try:
                    pred_y = self.Predictor(z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms, denormalize=True)
                except:
                    print(f"Error in generation results, regenerating the structure")
                    continue
            properties = {} 
            properties["Young's modulus"] = pred_y[:, :3]
            properties["Shear modulus"] = pred_y[:, 3:6]
            properties["Poisson ratio"] = pred_y[:, 6:]

            structure_prompt  = graph_to_text(coords, edge_index)
            supervisor_input_text = construct_supervisor_input(prompt=input_text, structure=structure_prompt, lattice_lengths=lengths, lattice_angles=angles, properties=properties) 
            # 2. Evaluate the generated structure against the original prompt
            evaluation_result, improved_prompt, improved_properties  = self.evaluate(supervisor_input_text)
            final_output_prompt = f"Prompt: {input_text}\nEvaluation_result: {evaluation_result}\nImproved Prompt: {improved_prompt}\nImproved Properties: {improved_properties}"
            print(f"\nPrompt: {input_text}\nEvaluation_result: {evaluation_result}\nImproved Prompt: {improved_prompt}\nImproved Properties: {improved_properties}")
            if visualize_results:
                visualizeLattice(coords.cpu().numpy(), edge_index.cpu().numpy(), title='Final Structure')
            eval_num += 1
            if eval_num >= self.max_evaluate_num:
                break
        return z, coords, edge_index, lengths, angles, evaluation_result, pred_y, batch_data.y, improved_prompt, final_output_prompt

    def collaborate_between_agents_12(self, batch_data, scaffold, logic_mode='mix', num_steps_lattice=30, num_steps_geo=300, \
                                      optimize_network_params=False, thresh=0.5, mix_lam=0.1, condition=None):
        """
       logic_mode: poe/qoe/mix
        """
        self.Generator.eval()

        ##  1. Encode scaffold
        device = self.device
        z_pre, coords_pre, edge_index_pre, batch_pre, lengths_pre, angles_pre, num_atoms_pre = scaffold

        z_pre = torch.tensor(z_pre).to(device)
        coords_pre = torch.tensor(coords_pre).to(device)
        edge_index_pre = torch.tensor(edge_index_pre).to(device)
        batch_pre = torch.tensor(batch_pre).to(device)
        lengths_pre = torch.tensor(lengths_pre).to(device)
        angles_pre = torch.tensor(angles_pre).to(device)
        num_atoms_pre = torch.tensor(num_atoms_pre).to(device)

        lengths_pre, angles_pre = self.normalizer(lengths_pre, angles_pre)
        
        with torch.no_grad():
            lattice_latent_pre, geo_latent_pre = self.Generator.encode(z_pre, coords_pre, edge_index_pre, batch_pre, lengths_pre, angles_pre, num_atoms_pre)

        # 2. Encode training data
        (z, coords, edge_index, batch,
                 lengths, angles, num_atoms, num_edges) = (batch_data.node_type,batch_data.frac_coords, batch_data.edge_index,
                                                batch_data.batch, batch_data.lengths, batch_data.angles, batch_data.num_atoms, batch_data.num_edges)
        z = z.to(device)
        coords = coords.to(device)
        if condition is None:
            # use the original condition for reconstruction
            condition = batch_data.y.to(device)
        else:
            condition = condition.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        lengths = lengths.to(device)
        angles = angles.to(device)
        num_atoms = num_atoms.to(device)
        num_edges = num_edges.to(device)
        
        # visualizeLattice(coords_pre.cpu().numpy(), edge_index_pre.cpu().numpy(), title='Scaffold')
        # visualizeLattice(coords.cpu().numpy(), edge_index.cpu().numpy(), title='Init. data')
        
        # coords_pre = frac_to_cart_coords(coords_pre, lengths, angles, num_atoms = num_atoms_pre)
        lengths, angles = self.normalizer(lengths, angles)
        lattice_latent_pre, geo_latent_pre = self.Generator.encode(z_pre, coords_pre, edge_index_pre, batch_pre, lengths, angles, num_atoms_pre)

        lattice_latent, geo_latent = self.Generator.encode(z, coords, edge_index, batch, lengths, angles, num_atoms)
                
        
        if logic_mode == 'union':
            optimized_geo_latent, newAEmodel = self.optimize_geo_latent(geo_latent, batch, geo_latent_pre, batch_pre, lr=0.01, num_steps=num_steps_geo, \
                                                                        logic_mode=logic_mode, lam=mix_lam, optimize_network_params=optimize_network_params,thr=0.5)
            optimized_lattice_latent = lattice_latent
            # newAEmodel = self.Generator
            # return node_num_pred, lengths_pred, angles_pred, coords_pred_list[0], edge_index_list[0]
        else:        
            # condition = 0.001 * torch.randn((num_samples, AEmodel.condition_dim), device=device)
            optimized_lattice_latent, _, _ = self.optimize_lattice_latent(lattice_latent, lattice_latent_pre, condition, lr=0.01, num_steps=num_steps_lattice, logic_mode=logic_mode, lam_mix=0.5)
            optimized_geo_latent, newAEmodel = self.optimize_geo_latent(geo_latent, batch, geo_latent_pre, batch_pre, lr=0.01, num_steps=num_steps_geo, \
                                                                        logic_mode=logic_mode, lam=mix_lam, optimize_network_params=optimize_network_params,thr=0.2)

        # 4. Decode the optimized latent variables  
        node_num_pred, lengths_pred, angles_pred, coords_pred_list, edge_index_list, y_pred = \
            newAEmodel.decode_from_encoded(new_geo_latent=optimized_geo_latent, new_lattice_latent=optimized_lattice_latent, batch=batch, condition=condition, random_scale_latent=1.0, random_scale_geo=1.0)
        # visualizeLattice(coords_pred_list[0].cpu().numpy(), edge_index_list[0].cpu().numpy())
        # Decode the target latent variables
        node_num_pred_pre, lengths_pred_pre, angles_pred_pre, coords_pred_list_pre, edge_index_list_pre, y_pred_pre = \
            newAEmodel.decode_from_encoded(new_geo_latent=geo_latent_pre, new_lattice_latent=optimized_lattice_latent, batch=batch_pre, condition=condition, random_scale_latent=1.0, random_scale_geo=1.0)
        # visualizeLattice(coords_pred_list_pre[0].cpu().numpy(), edge_index_list_pre[0].cpu().numpy())

        # compute the Sinkhorn distance
        cost = pairwise_l2(coords_pred_list[0].detach(), coords_pred_list_pre[0].detach())
        cost = cost / (cost.max() + 1e-8)  
        P = sinkhorn_log(cost, epsilon=.1, verbose=False)
        cooreds_all, edge_index_all = gather_nodes_and_edges(coords_pred_list[0], edge_index_list[0], coords_pred_list_pre[0], edge_index_list_pre[0], P, logic_mode, thresh=thresh)

        # visualizeLattice(cooreds_all.cpu().numpy(), edge_index_all.cpu().numpy())

        cooreds_all, edge_index_all = remove_close_nodes(cooreds_all, edge_index_all, 0.2)

        cooreds_all, edge_index_all = check_all_nodes_connected(cooreds_all, edge_index_all)
        final_node_num = cooreds_all.shape[0]
        # print('final_node_num', final_node_num)
        return final_node_num, lengths_pred.detach(), angles_pred.detach(), cooreds_all.detach(), edge_index_all.detach()
    
    def optimize_geo_latent(self, latent, batch, target_latent, batch_target, lr=0.01, num_steps=300, eps=1e-8,logic_mode='mix', lam=0.5, optimize_network_params=False\
                        , lam_node=1.0, lam_edge=1e-2, lam_prior=1e-4, thr=0.1):
        print('optimizing geo latent...')
        AEmodel = copy.deepcopy(self.Generator)

        # ---------- build optimizer ----------
        latent0 = latent.clone().detach().requires_grad_(True)
        params_dict = [{'params': [latent0]}]

        if optimize_network_params:
            train_modules = [AEmodel.proj_in_semantic,
                            AEmodel.proj_in_edge,
                            AEmodel.proj_in_pos]
            for m in train_modules:
                m.requires_grad_(True)
                params_dict.append({'params': m.parameters()})
        else:
            for p in AEmodel.parameters():
                p.requires_grad_(False)

        opt_latent = torch.optim.Adam(params_dict, lr=lr)

        # ---------- target (compute once) ----------
        with torch.no_grad():
            z_e_t, z_p_t, z_s_t = AEmodel.disentangle_latent(target_latent,
                                                            batch=batch_target)
            _, mu_sem_t0, log_sem_t0 = AEmodel.reparameterize(z_s_t, 'semantic', 1.0, batch_target)
            _, mu_node_t, log_node_t = AEmodel.reparameterize(z_p_t, 'coords',   1.0)
            _, mu_edge_t, log_edge_t = AEmodel.reparameterize(z_e_t, 'edge',   1.0)
            # edge regularizer refs
            z_e_r, z_p_r, _ = AEmodel.disentangle_latent(latent, batch=batch)
            _, mu_edge_old, log_edge_old = AEmodel.reparameterize(
                                            z_e_r, 'edge', 1.0)
            _, mu_node_old, log_node_old = AEmodel.reparameterize(
                                            z_p_r, 'coords', 1.0)


                # ---------- loop ----------
        for step in range(num_steps):
            opt_latent.zero_grad()

            # Current latent0 → disentangle
            z_e, z_p, z_s = AEmodel.disentangle_latent(latent0, batch=batch)
            _, mu_sem, log_sem = AEmodel.reparameterize(z_s, 'semantic', 1.0, batch=batch)
            _, mu_node, log_node = AEmodel.reparameterize(z_p, 'coords', 1.0)
            _, mu_edge, log_edge = AEmodel.reparameterize(z_e, 'edge', 1.0)

            # Semantic target (PoE/QoE/Mix) —— internal detach handled
            if logic_mode == 'int':
                mu_sem_t, log_sem_t = poe(mu_sem, log_sem,
                                        mu_sem_t0, log_sem_t0)

            elif logic_mode == 'neg':
                # mu_sem_t, log_sem_t = qoe(mu_sem, log_sem,
                #                         mu_sem_t0, log_sem_t0)
                mu_sem_t, log_sem_t = gaussian_negation(mu_sem, log_sem, mu_sem_t0, log_sem_t0,
                                       alpha=1.0, beta=0.01, eps=eps)
            else:
                mu_sem_t, log_sem_t = mixture_mm(mu_sem, log_sem,
                                                mu_sem_t0, log_sem_t0,
                                                lam=lam)

            L_sem = kl_gaussian_diag(mu_sem, log_sem,
                                    mu_sem_t.detach(), log_sem_t.detach()).mean()

            if logic_mode  == 'union':
                loss = L_sem
            else:
                # Sinkhorn on node μ (detached)
                cost = pairwise_l2(mu_node.detach(), mu_node_t.detach())
                cost = cost / (cost.max() + 1e-8)  
                P = sinkhorn_log(cost,
                            epsilon=0.1)

                L_node = node_logic_loss(P,
                                        mu_node,  log_node,
                                        mu_node_t.detach(), log_node_t.detach(),
                                        logic_mode=logic_mode, lam=lam,thr=thr)
                L_node_keep = node_logic_keep_original(P, mu_node, log_node, mu_node_old, log_node_old,thr=thr)
                L_node += L_node_keep

                # Edge regularizer (keep old topology)
                L_edge = node_logic_loss(P,
                                        mu_edge,  log_edge,
                                        mu_edge_t.detach(), log_edge_t.detach(),
                                        logic_mode=logic_mode, lam=lam,thr=thr)
                L_edge_keep = node_logic_keep_original(P, mu_edge, log_edge, mu_edge_old, log_edge_old,thr=thr)
                L_edge += L_edge_keep

                loss = L_sem + lam_node*L_node + lam_edge*L_edge \
                        + lam_prior*latent0.pow(2).mean()
            loss.backward()
            opt_latent.step()

            # if step % 10 == 0 or step == num_steps-1:
            #     print(f"[{step:03d}]  loss={loss.item():.4f}  Lsem={L_sem.item():.4f}   Lnode={L_node.item():.4f}   Ledge={L_edge.item():.4f}   Lprior={lam_prior*latent0.pow(2).mean().item():.4f}")



        return latent0.detach(), AEmodel

    def optimize_lattice_latent(
            self,
            latent_src,            
            latent_tgt,            
            condition,             
            lr=0.01,
            num_steps=30,
            logic_mode="union",      
            lam_mix=0.5,           
            eps=1e-8):

        device   = self.device
        AE       = self.Generator
        print(f"optimizing lattice latent...")
        with torch.no_grad():
            _, mu_tgt, log_tgt = AE.reparameterize(latent_tgt)
            _, mu_init, log_init = AE.reparameterize(latent_src)

            if logic_mode == "int":
                mu_teacher, log_teacher = poe(mu_init,  log_init,
                                            mu_tgt,   log_tgt)
            elif logic_mode == "neg":
                # mu_teacher, log_teacher = qoe(mu_init,  log_init,
                #                             mu_tgt,   log_tgt)
                mu_teacher, log_teacher = gaussian_negation(mu_init, log_init, mu_tgt, log_tgt,
                                       alpha=1.0, beta=0.01, eps=eps)
            elif logic_mode == "mix":
                mu_teacher, log_teacher = mixture_mm(mu_init, log_init,
                                                mu_tgt,  log_tgt,
                                                lam=lam_mix)
            else:
                raise ValueError("logic_mode must be int/neg/mix")

            # detach 
            mu_teacher  = mu_teacher.detach()
            log_teacher = log_teacher.detach()

        # ---------- (1) latent ----------
        latent = latent_src.clone().detach().requires_grad_(True).to(device)
        opt_latent = torch.optim.Adam([latent], lr=lr)

        for step in range(num_steps):
            opt_latent.zero_grad()

            _, mu_cur, log_cur = AE.reparameterize(latent)

            # KL 
            kl_align = kl_gaussian_diag(mu_cur, log_cur,
                                        mu_teacher, log_teacher, eps=eps)
            kl_loss  = kl_align.mean()            # scalar

            kl_loss.backward()
            opt_latent.step()

            # ---------- print ----------
            if step % 10 == 0 or step == num_steps - 1:
                with torch.no_grad():
                    sample_lat, _, _ = AE.reparameterize(latent, tag='lattice',
                                                        noise_scale=1.0)
                    if AE.is_condition and condition is not None:
                        sample_lat = torch.cat([sample_lat, condition], dim=-1)

                    node_logit, node_pred, len_pred, ang_pred = \
                            AE.decoder.decode_latent_global(sample_lat)

                # print(f"[{logic_mode}] step {step:2d}  loss={kl_loss.item():.4f} "
                    # f"node_num_pred: {node_pred}")

        return latent.detach(), mu_cur.detach(), log_cur.detach()


    def collaborative_end_to_end_generation(self, dataset, input_text, logic_mode='union', num_steps_lattice=30, num_steps_geo=300, max_collaboration_num=5, \
                                            optimize_network_params=False, fusion_thresh=0.1, mix_lam=0.2, condition=None,select_max_node_num=30, verbose=False, save_dir=None):
        
        indices = []
        for i, data in enumerate(tqdm(dataset)):
            if data.num_atoms <= select_max_node_num and data.num_edges <= select_max_node_num * 2:
                indices.append(i)
        select_init_data = dataset[indices]

        init_prompt = input_text
        if condition is not None:
            young = condition[:, :3]
            shear = condition[:, 3:6]
            poisson = condition[:, 6:]

        for j in range(max_collaboration_num):
            print(f'====================> Generate trail index:{j}/{max_collaboration_num}')
            z, coords, edge_index, batch, lengths, angles, num_atoms, evaluation_results_A1 = self.collaborate_between_agents_13(input_text, visualize_results=verbose)
            if verbose:
                visualizeLattice(coords.cpu().numpy(), edge_index.cpu().numpy(), title=f'scaffold in {j}th iter') 
            
            if condition is not None:
                prompt_w_cond = init_prompt + f" Properties: Young's modulus: {young.tolist()[0]}, Shear modulus: {shear.tolist()[0]}, Poisson ratio: {poisson.tolist()[0]}."
            else:
                prompt_w_cond = init_prompt
            scaffold = z, coords, edge_index, batch, lengths, angles, num_atoms
            z, coords, edge_index, lengths_pred, angles_pred, evaluation_result, y_pred, y_cond, improved_prompt, final_output_prompt = \
                self.collaborate_between_agents_23(select_init_data, scaffold, prompt_w_cond,\
                                                          logic_mode=logic_mode, num_steps_lattice=num_steps_lattice, num_steps_geo=num_steps_geo, \
                                                            optimize_network_params=optimize_network_params, fusion_thresh=fusion_thresh, mix_lam=mix_lam, condition=condition)
            if verbose:
                visualizeLattice(coords.cpu().numpy(), edge_index.cpu().numpy(), title=f'structure in {j}th iter')
            print('lengths_pred', lengths_pred)
            print('angles_pred', angles_pred)
            print('y_pred', y_pred)
            print('y_cond', condition)
            print('final_output_prompt', final_output_prompt)
            input_text = improved_prompt
            if evaluation_result >= self.evaluation_threshold:
                print(f"{j}th iter, Evaluation result: {evaluation_result} is greater than the threshold, stop the generation, and save the result.")
                break
            
        if save_dir is not None:
            print(f"Saving the generated structure to {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            lattice_name = os.path.join(save_dir, f'{logic_mode}_{i}_{evaluation_result}.npz')

            np.savez(lattice_name,
                    atom_types=z.cpu().numpy(),
                    lengths=lengths_pred.cpu().view(-1).numpy(),
                    angles=angles_pred.cpu().view(-1).numpy(),
                    frac_coords=coords.cpu().numpy(),
                    edge_index=edge_index.cpu().numpy(),
                    prop_list=condition.cpu().numpy(),
                    )



def test():
    pass



if __name__ == "__main__":
    is_test = False
    if is_test:
        test()
    else:
        pass
