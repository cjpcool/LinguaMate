import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')


import torch

from openai import OpenAI
from google import genai
from google.genai import types
import re

from tqdm import tqdm
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

def parse_graph_llm(text: str):
    """
    Parse a node/edge description like the prompt example and return
    coords  – Tensor[N, 3]  (float32)
    edges   – Tensor[M, 2]  (long)
    """
    # --- split the text into its three sections -----------------------------
    #     1) header with node number (optional for parsing)
    #     2) node coordinates lines
    #     3) edge lines
    #
    # find where the coordinates section starts and where the edges start
    coord_start = text.index("Node coordinates:") + len("Node coordinates:")
    edge_start  = text.index("Edges:")

    lengths_start = text.index("Lattice Lengths:")
    angles_start = text.index("Lattice Angles:")
    
    coord_block = text[coord_start:edge_start].strip().splitlines()
    edge_block  = text[edge_start + len("Edges:"):lengths_start].strip().splitlines()
    lengths_block  = text[lengths_start + len("Lattice Lengths:"):angles_start].strip().splitlines()
    angles_block  = text[angles_start + len("Lattice Angles:"):].strip().splitlines()
    
    # --- helper: grab all numbers in a line ---------------------------------
    num_pat = re.compile(r'[-+]?\d*\.\d+|\d+')   # floats or ints

    # Parse coordinates -------------------------------------------------------
    coords = [
        [float(x) for x in num_pat.findall(line)]
        for line in coord_block if num_pat.search(line)
    ]
    coords = torch.tensor(coords, dtype=torch.float32)  # → [N, 3]

    # Parse edges -------------------------------------------------------------
    edges = [
        [int(float(x)) for x in num_pat.findall(line)]
        for line in edge_block if num_pat.search(line)
    ]
    edges = torch.tensor(edges, dtype=torch.long)       # → [M, 2]
    

    # Construct other elements:z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms,
    num_atoms = torch.LongTensor([coords.shape[0]])
    edge_index = edges.T
    try:
        node_labels = classify_nodes_with_geometry(coords, edge_index)
        z = torch.LongTensor(torch.argmax(node_labels,dim=-1)+1)
    except:
        Warning("No geometry information, using default node labels")
        z = torch.ones(coords.shape[0], dtype=torch.long)
    batch = torch.zeros(num_atoms, dtype=torch.long)
    lengths = [
        [float(x) for x in num_pat.findall(line)]
        for line in lengths_block if num_pat.search(line)
    ]
    angles = [
        [float(x) for x in num_pat.findall(line)]
        for line in angles_block if num_pat.search(line)
    ]
    lengths = torch.FloatTensor([[1,1,1]])
    angles = torch.FloatTensor([[90,90,90]])

    return z, coords, edge_index, batch, lengths, angles, num_atoms

def parse_graph(text):
    """
    Parse a node/edge description like the prompt example and return
    coords  – Tensor[N, 3]  (float32)
    edges   – Tensor[M, 2]  (long)
    """
    # --- split the text into its three sections -----------------------------
    #     1) header with node number (optional for parsing)
    #     2) node coordinates lines
    #     3) edge lines
    #
    # find where the coordinates section starts and where the edges start
    coord_start = text.index("Node coordinates:") + len("Node coordinates:")
    edge_start  = text.index("Edges:")
    # properties_start = text.index("Properties:")
    
    coord_block = text[coord_start:edge_start].strip().splitlines()
    edge_block  = text[edge_start + len("Edges:"):].strip().splitlines()
    # properties_block  = text[properties_start + len("Properties:"):].strip()
    # --- helper: grab all numbers in a line ---------------------------------
    num_pat = re.compile(r'[-+]?\d*\.\d+|\d+')   # floats or ints

    # Parse coordinates -------------------------------------------------------
    coords = [
        [float(x) for x in num_pat.findall(line)]
        for line in coord_block if num_pat.search(line)
    ]
    coords = torch.tensor(coords, dtype=torch.float32)  # → [N, 3]

    # Parse edges -------------------------------------------------------------
    edges = [
        [int(float(x)) for x in num_pat.findall(line)]
        for line in edge_block if num_pat.search(line)
    ]
    edges = torch.tensor(edges, dtype=torch.long)       # → [M, 2]
    

    # Construct other elements:z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms,
    num_atoms = torch.LongTensor([coords.shape[0]])
    edge_index = edges.T
    try:
        node_labels = classify_nodes_with_geometry(coords, edge_index)
        z = torch.LongTensor(torch.argmax(node_labels,dim=-1)+1)

    except:
        Warning("No geometry information, using default node labels")
        z = torch.ones(coords.shape[0], dtype=torch.long)
    batch = torch.zeros(num_atoms, dtype=torch.long)
    lengths = torch.FloatTensor([[1,1,1]])
    angles = torch.FloatTensor([[90,90,90]])


    return z, coords, edge_index, batch, lengths, angles, num_atoms




INSTRUCTIONS_BASELINE_LLM="""
You are a metamaterial scientist specializing in structural design and mechanical characterization. You have expert knowledge of canonical 3‑D architectures (octet‑truss, BCC, SC, Kelvin cell, Diamond, TPMS, etc.) and their typical mechanical responses.

Task
-----

Given a single *design requirement*, your task is to generate a possible memtamaterial structures, described as a graph. The graph should be defined by:

- **Nodes** — 3‑D fractional coordinates.  
- **Edges** — pairs of node indices.
- **Lattice Lengths** - lengths of the unit cell in 3D.
- **Lattice Angles** - angles of the unit cell in 3D.

Output the graph in a code block exactly as shown below; provide **no additional text, commentary, or reasoning**.

Input
-----

Design prompt (free text).

Output format
-------------
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

Lattice Lengths:
(L1, L2, L3)

Lattice Angles:
(A1, A2, A3)
~~~

Constraints
-----------

- Return the output *only* in the specified layout and code‑block format.  
- Do not include any other information.
"""






API_KEY='AIzaSyB1gA-mwQEa1KzwJ99X8k_OzUJeUhp6yTY'
MODEL_CLINET_VERSION="gpt-4o-mini"
# MODEL_CLINET_VERSION="gemini-1.5-flash"
MAX_NODE_NUM = 100



save_dir = 'D:/ModalAgent/evaluation/'
save_name_ae = 'checkpoints/vae_cond_128_beta001_dis_same_100_frac'
prompt_file = 'D:/ModalAgent/metamaterial_design_prompts.csv'
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device='cpu'
root = os.path.dirname(os.path.abspath(__file__)) + '/../'
print('root', root)    

client_model_name = 'gpt-4o-mini'
if 'gpt' in client_model_name:
    client = OpenAI(
        api_key=API_KEY
        )
elif 'gemini' in client_model_name:
    client = genai.Client(api_key=API_KEY)

results = {'Prompt': [], 'Output': []}  

input_texts = pd.read_csv(prompt_file, header=None)
input_texts = input_texts.values.tolist()
for i, input_text in enumerate(tqdm(input_texts)):
    input_text = input_text[0]
    if 'gpt' in client_model_name:
        response = client.responses.create(
            model=client_model_name,
            instructions=INSTRUCTIONS_BASELINE_LLM,
            input=input_text,
        ) 
        output = response.output_text
    elif 'gemini' in client_model_name:
        response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCTIONS_BASELINE_LLM),
            contents=input_text,
            # temperature=1
        )
        output = response.text
    results['Output'].append(output)
    results['Prompt'].append(input_text)

results_df = pd.DataFrame(results)
results_df.to_csv(save_dir+'/results_gpt-4o-mini.csv', index=False)

results_df = pd.read_csv(save_dir+'/results_gpt-4o-mini.csv')

    # ModalAgent = MetamatGenAgent12(root=root, ckpt_dir=save_name_ae, device=device, backbone='vae',client_model=MODEL_CLINET_VERSION, evaluation_threshold=0.6, max_evaluate_num=10)
    # score = []

    # output_text = results_df['Output'].values.tolist()
    # for i, output in enumerate(tqdm(output_text)):
    #     input_text = results_df['Prompt'].values.tolist()[i]
    #     # print('output', output)
    #     # Parse the graph from the output
    #     # z, coords, edge_index, batch, lengths, angles, num_atoms = parse_graph(output)
    #     z, coords, edge_index, batch, lengths, angles, num_atoms = parse_graph_llm(output)
    #     # visualizeLattice(coords.cpu().numpy(), edge_index.cpu().numpy())
    #     evaluation_result = ModalAgent.get_agent2_score(input_text, num_atoms, lengths, angles, coords, edge_index)
    #     print(f"Evaluation result: {evaluation_result}")
    #     score.append(evaluation_result)
    #     # Save the results to a CSV file

    # results_df['Score'] = score
    # results_df.to_csv(save_dir+'/results_gpt-4o-mini_score_gpt-4.1.csv', index=False)
    # print('mean score', results_df['Score'].mean())


