# LinguaMate: Language-Guided Metamaterial Discovery via Symbolic-Driven Latent Optimization

This script performs **MetaSymbO** using a end-to-end collaborative method.  
It enables **end-to-end structure generation** driven by symbolic logic and latent optimization through interactions among:

* **Designer Agent** – formulates prompts (e.g., *gpt-4o-mini*)  
* **Generator Model** – VAE (diffusion TBD)  
* **Supervisor Agent** – evaluates and feeds back (e.g., *gpt-4.1*)


---

## Requirements

* Python ≥ 3.8  
* PyTorch  == torch2.4
* torch_geometric and its dependencies for torch2.4
* OpenAI API key (for the chosen GPT models). Alternatively, use other AI APIs and revise related code.

---
## Dataset
To download dataset, please revise "datasets/dataset_truss.py" 
```
root = '[Your data path]/LatticeModulus'
```
to 
```
root = 'your dataset path'
```
Then, run datasets/dataset_truss.py,  the data will be automatically downloaded and preprocessed.

Note: Sometimes there will be network error. Please try again after waiting for several minutes.

## Checkpoints
We will update the checkpoints via public links after the paper is accepted. 

In the current phase, please (1) set the root path in scripts "train_ae.py" and "train_predictor.py", and (2) run "train_ae.py" and "train_predictor.py" to make a VAE checkpoint of Agent 2 (generator), and the predictor in Agent 3 (Supervisor).


## Usage

```bash
python run_modal_agent.py \
  -prompt "Design a structure with high stiffness, with less nodes." \
  -dataset_path "[Your path]/LatticeModulus" \
  -api_key <your_openai_api_key> \
  -save_dir results/lattices \
  -verbose True   # optional. if do not want to visualize, omit this param. 
  # --condition_vec "[0.001, 0.002, 0.01, 0.05, 0.1, 0.2, 0.19, 0.25, 0.2, 0.16, 0.05, 0.08]" # optional. for conditional generation.
```


Runing example:
```bash
python run_modal_agent.py   --prompt "Design a structure with high stiffness, with less nodes."   --dataset_path "[Your path]/LatticeModulus"   --api_key "" --save_dir "results/lattices" --verbose "True"
```


---

## Key Arguments

| Argument                       | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `--prompt`                     | Natural-language text guiding structure generation                          |
| `--api_key`                    | OpenAI API key for GPT-based agents                                         |
| `--cuda`                       | GPU ID (`-1` for CPU)                                                       |
| `--root`                       | Project root directory                                                      |
| `--designer_client`            | LLM for prompt design (default `gpt-4o-mini`)                               |
| `--supervisor_client`          | LLM for evaluation (default `gpt-4.1`)                                      |
| `--backbone`                   | Generator type: `vae` or `diffusion`                                        |
| `--save_name`                  | Identifier for checkpoints/output                                           |
| `--latent_dim`                 | Dimensionality of the latent space                                          |
| `--dataset_path`               | Path to metamaterial dataset                                               |
| `--condition_vec`              | Target property vector (must match `data.y` dimension)                      |
| `--logic_mode`                 | Fusion logic: `union`, `mix`, `int`, or `neg`                               |
| `--max_collaboration_num`      | Maximum collaboration iterations                                            |
| `--optimize_network_params`    | Fine-tune model weights during optimization (flag)                          |
| `--save_dir`                   | Directory for generated results                                             |

---

## Output

* **Generated structures** saved in `save_dir`  
* **Logs** detailing collaboration loops and evaluations  
* Structures satisfy both **semantic** (prompt) and **physical** (property) constraints



