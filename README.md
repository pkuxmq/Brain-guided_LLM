# Brain-guided language models for robust reasoning

This repo is the official implementation of ["Beyond representational alignment with brain-guided language models for robust reasoning"](https://www.nature.com/articles/s42256-026-01278-w) (Nature Machine Intelligence, 2026).

---

## Table of contents

- [Dependencies](#dependencies)
- [Data](#data)
- [Supported models](#supported-models)
- [Project structure](#project-structure)
- [Pipeline](#pipeline)
  - [Extract model activations](#extract-model-activations)
  - [Behaviour analysis](#behaviour-analysis)
  - [Neural predictivity analysis](#neural-predictivity-analysis)
  - [NARI: Intervention on fMRI questions](#nari-intervention-on-fmri-questions)
  - [NARI (gen.): Generalization to new questions](#nari-gen-generalization-to-new-questions)
  - [NARF: Fine-tuning with brain guidance](#narf-fine-tuning-with-brain-guidance)
  - [Testing](#testing)
- [HCP relational processing experiment](#hcp-relational-processing-experiment)

---

## Dependencies

* Python 3.8+
* PyTorch ≥ 2.0
* CUDA-compatible GPU(s) recommended

Install Python packages:

```bash
pip install timm scipy scikit-learn transformers peft nnsight huggingface_hub datasets pandas jsonlines matplotlib seaborn accelerate
```

---

## Data

### fMRI data

We provide preprocessed fMRI data in `fmri_data/`. The data contains brain responses from human participants solving deductive reasoning problems during neuroimaging. Each subject has a response matrix of shape `(n_questions, n_voxels)` containing beta values extracted via GLMSingle from relevant brain regions (top 10% most responsive voxels within ROIs).

```
fmri_data/
└── preprocessed_data_glmsinglesep_newdrroi_topksep/
    └── top-10%/
        └── all_extracted_beta_{analyze_type}/
            └── {sub}/{sub}.mat    # Per-subject fMRI response matrix
```

Three ROI types are available, controlled by the `-analyze_type` argument throughout the pipeline:
- `deductive_reasoning`: Voxels from deductive reasoning-related ROIs
- `language`: Voxels from language network ROIs
- `md`: Voxels from multiple demand network ROIs

The raw fMRI data was preprocessed using MATLAB with SPM12 and GLMSingle. Preprocessing scripts are provided in `matlab/`.

### Neuroimaging task information

`neuroimaging_info/` contains preprocessed problem items and human behavioural results:
- `task_items.pkl`: Problem items used in the fMRI experiment `{task_run: {premise1, premise2, premise3, conclusion}}`
- `events.pkl`: Human behavioural data `{sub: {task_run: DataFrame with accuracy, RT, etc.}}`

### Generated reasoning datasets

`data/` contains generated test/train/validation datasets and their generation scripts:

| File | Description |
|------|-------------|
| `deductive_reasoning_data_test.json` | Test set (3 premises, syllogisms + transitive) |
| `deductive_reasoning_data_test_4premises.json` | Test set (4 premises) |
| `deductive_reasoning_data_test_5premises.json` | Test set (5 premises) |
| `deductive_reasoning_data_test_6premises.json` | Test set (6 premises) |
| `deductive_reasoning_data_test_propositional.json` | Propositional reasoning test set |
| `deductive_reasoning_data_val_new.json` | Validation set |
| `deductive_reasoning_data_train_{32,64,...,192}.json` | Training sets of varying sizes for the optional experiment with synthetic data |
| `relational_test_set.jsonl` / `relational_val_set.jsonl` | Relational reasoning data (for HCP experiment) |

To regenerate or customize datasets:

```bash
cd data
python generate_data.py                    # Standard test set (3 premises)
python generate_data_4premises.py          # 4-premise test set
python generate_data_5premises.py          # 5-premise test set
python generate_data_6premises.py          # 6-premise test set
python generate_data_propositional.py      # Propositional reasoning
python generate_data_train_latest.py       # Training data
python generate_relational_data.py         # Relational reasoning data
```

### HCP fMRI data

`hcp_fmri_results/` contains preprocessed fMRI data (extracted responses) for the HCP Relational Processing task, used for cross-dataset validation. See [HCP relational processing experiment](#hcp-relational-processing-experiment).

---

## Supported models

The code supports the following LLM families (both instruction-tuned and base variants):

| Model | model_type | Example model_path |
|-------|------------|-------------------|
| Qwen2-1.5B-Instruct | `qwen1-5b` | `Qwen/Qwen2-1.5B-Instruct` |
| Qwen2-7B-Instruct | `qwen7b` | `Qwen/Qwen2-7B-Instruct` |
| Qwen2-72B-Instruct | `qwen72b` | `Qwen/Qwen2-72B-Instruct` |
| Qwen3-4B | `qwen3_4b` | `Qwen/Qwen3-4B` |
| Mistral-7B-Instruct-v0.2 | `mistral` | `mistralai/Mistral-7B-Instruct-v0.2` |
| LLaMA-2-7B-Chat | `llama2` | `meta-llama/Llama-2-7b-chat-hf` |
| LLaMA-3-8B-Instruct | `llama3` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| LLaMA-3.3-70B-Instruct | `llama3-3_70b` | `meta-llama/Llama-3.3-70B-Instruct` |
| Phi-4-mini-instruct | `phi4-mini` | `microsoft/Phi-4-mini-instruct` |
| Gemma-2-9B-IT | `gemma2_9b` | `google/gemma-2-9b-it` |
| DeepSeek-R1-Distill-Qwen-1.5B | `deepseekqwen1-5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Qwen2-7B (base) | `qwen7b_base` | `Qwen/Qwen2-7B` |
| LLaMA-2-7B (base) | `llama2_base` | `meta-llama/Llama-2-7b-hf` |

---

## Project structure

```
.
├── LM.py                          # LLM wrapper classes (LM_normal, LM_nnsight, LM_nnsight_base, etc.)
├── LM_finetune.py                 # Model definition for fine-tuning (ModelwithAttentionSupervision)
├── utils.py                       # Shared utilities (regression, intervention direction computation, testing, etc.)
│
├── get_activations.py             # Extract activations for instruction-tuned models on fMRI task items
├── get_activations_base.py        # Extract activations for base (non-instruction-tuned) models
├── get_activations_localized.py   # Extract activations with functional localization (task vs. control)
├── get_activations_base_localized.py
│
├── analyze_behaviour_results.py   # Analyze model vs. human behavioural accuracy and consistency
├── calculate_ceiling.py           # Calculate predictivity ceiling for fMRI responses
│
├── make_analysis.py               # Neural predictivity analysis (brain score per layer)
├── make_analysis_localized.py     # Neural predictivity with functionally localized model units
│
├── make_intervention_sep.py       # NARI: Make intervention on fMRI task items
├── make_intervention_generaldir.py# NARI (gen.): Apply general directions to new test problems
│
├── finetune_model_sep.py          # NARF / NARF+Label fine-tuning
├── finetune_model_sep_onlylabel.py# Label-only fine-tuning baseline
├── finetune_model_sep_syntheticdata.py     # NARF+Label with synthetic training data
├── finetune_model_sep_onlylabel_syntheticdata.py  # Label-only with synthetic data
│
├── test_model.py                  # Test models on generated deductive reasoning datasets
├── test_model_analysis_propositional.py    # Test on propositional reasoning dataset
├── test_finetune_model_cosine.py  # Test with cosine similarity analysis
│
├── compare_model_trajectories.py  # Compare representation trajectories across layers/models
├── visualize_brain_rep.py         # Visualize fMRI representations (PCA)
├── visualize_model_rep.py         # Visualize LLM representations (PCA)
├── visualize_intv_rep_dir.py      # Visualize intervention directions
├── visualize_finetune_rep_test.py # Visualize fine-tuned vs. original representations
│
├── hcp_get_activations.py         # HCP: Extract activations for relational processing task
├── hcp_make_intervention.py       # HCP: Make intervention on HCP task items
├── hcp_make_intervention_generaldir.py  # HCP: Apply directions to new relational problems
├── hcp_finetune_model.py          # HCP: Fine-tune with brain guidance
├── hcp_utils.py                   # HCP: Shared utilities
│
├── stimuli_mapping_final.jsonl    # Mapping between HCP fMRI stimuli and problem items
│
├── run_get_activations.sh         # Shell scripts with example commands for each stage
├── run_get_activations_localized.sh
├── run_analysis.sh
├── run_analysis_localized.sh
├── run_ceiling.sh
├── run_intervention.sh
├── run_intervention_generaldir.sh
├── run_finetune.sh
├── run_test.sh
├── hcp_run_get_activations.sh
├── hcp_run_intervention.sh
├── hcp_run_intervention_generaldir.sh
├── hcp_run_finetune.sh
│
├── data/                          # Generated datasets and generation scripts (see Data section)
├── fmri_data/                     # Preprocessed fMRI data (see Data section)
├── hcp_fmri_results/              # HCP fMRI data (see Data section)
├── neuroimaging_info/             # Task items and human behavioural results (see Data section)
└── matlab/                        # MATLAB code for fMRI preprocessing (SPM12 + GLMSingle)
```

### Output directory structure (auto-generated)

```
results/
├── activations_results/{model_type}/       # Extract model activations outputs
│   ├── {task_run}_hidden.npy               #   Hidden states: (n_questions, n_layers, hidden_dim)
│   ├── {task_run}_attention.npy            #   Attention states: (n_questions, n_layers, n_heads, head_dim)
│   └── behaviour_results.pkl               #   Model answers and correctness
│
├── activations_results_localized/{model_type}/  # Localized activations outputs
│
├── score_results/{model_type}/{suffix}/    # Neural predictivity outputs
│   ├── analysis.log                        #   Per-subject, per-layer brain scores
│   └── scores.npy                          #   Per-voxel scores at best layer
│
├── ceiling_results/                        # Noise ceiling values
│
├── intervention_sep_results/{model_type}/{suffix}/  # NARI outputs
│   ├── results.log
│   └── directions_{sub}.npy                #   Saved intervention directions per subject
│
├── intervention_sumdir_test/{model_type}/{suffix}/  # NARI (gen.) outputs
│   └── results.log                         #   Accuracy at different intervention scales
│
├── finetune_results/{model_type}/{suffix}/ # NARF outputs
│   ├── model/                              #   Best checkpoint (saved model weights)
│   └── train.log
│
└── test_results/{model_type}_{suffix}/     # Testing outputs
    └── behaviour_results.pkl
```

---

## Pipeline

The overall pipeline is as follows:

```
fMRI Data + LLM
       │
       ▼
  Extract LLM activations on fMRI task items
       │
       ├──► Behaviour analysis
       │
       ├──► Neural predictivity analysis (brain score)
       │
       ├──► NARI: Brain-guided intervention on fMRI task items
       │         │
       │         ▼
       │    NARI (gen.): Apply general directions to new test problems
       │
       └──► NARF: Fine-tune LLM with brain-guided supervision
                 │
                 ▼
            Testing on generated datasets
```

### Extract model activations

Extract intermediate representations (hidden states and attention outputs) and behavioral results of LLMs for the problems in the fMRI dataset. This is the prerequisite for all downstream analyses.

**Standard extraction:**

```bash
# Instruction-tuned model
CUDA_VISIBLE_DEVICES='0' python get_activations.py \
    -model_type qwen1-5b \
    -model_path Qwen/Qwen2-1.5B-Instruct \
    -device cuda

# Base model
CUDA_VISIBLE_DEVICES='0,1' python get_activations_base.py \
    -model_type llama2_base \
    -model_path meta-llama/Llama-2-7b-hf

# Untrained model (random initialization)
CUDA_VISIBLE_DEVICES='0' python get_activations.py \
    -model_type qwen1-5b \
    -model_path Qwen/Qwen2-1.5B-Instruct \
    -device cuda -untrained
```

**Extraction with functional localization:**

`get_activations_localized.py` additionally runs a control condition (reading premises without reasoning) to identify task-relevant model units via statistical comparison (t-test between task and control activations). The localized representations are used in `make_analysis_localized.py`.

```bash
CUDA_VISIBLE_DEVICES='0' python get_activations_localized.py \
    -model_type qwen1-5b \
    -model_path Qwen/Qwen2-1.5B-Instruct \
    -device cuda
```

**Key arguments:**
- `-model_type`: Model identifier (see [Supported models](#supported-models))
- `-model_path`: Path to HuggingFace model (local path or hub ID)
- `-device`: `cuda` or `cpu`
- `-untrained`: Use randomly initialized weights
- `-bf16`: Use bfloat16 precision

**Outputs** → `results/activations_results/{model_type}/` (or `results/activations_results_localized/`)

See `run_get_activations.sh` and `run_get_activations_localized.sh` for all model commands.

---

### Behaviour analysis

Compare model and human behavioral accuracy and consistency:

```bash
python analyze_behaviour_results.py -model_type qwen1-5b
```

Requires activations extraction output. Reports model accuracy (overall, syllogisms, transitive), human accuracy, and model-human consistency per subject.

---

### Neural predictivity analysis

Compute brain scores by fitting ridge regression from LLM representations to fMRI responses (cross-validated across problem items). The noise ceiling for fMRI responses can also be computed using `calculate_ceiling.py`.

**Layer-wise analysis (each layer; finally take the max layer):**

```bash
python make_analysis.py \
    -model_type qwen1-5b \
    -analyze_type deductive_reasoning \
    -use_ridgecv
```

**Localized analysis (using functionally localized model units):**

```bash
python make_analysis_localized.py \
    -model_type qwen1-5b \
    -analyze_type deductive_reasoning \
    -use_ridgecv \
    -LLM_rep_type all  # options: all, hidden, attention
```

**Predictivity ceiling calculation:**

```bash
python calculate_ceiling.py -analyze_type deductive_reasoning -use_ridgecv
```

**Key arguments:**
- `-analyze_type`: fMRI ROI type (`deductive_reasoning`, `language`, `md`)
- `-use_ridgecv`: Use cross-validated ridge alpha selection
- `-only_syllogisms` / `-only_transitive`: Restrict to one task type
- `-rand_feature`: Use random features as a baseline control
- `-LLM_rep_type` (localized only): `all`, `hidden`, or `attention`

**Outputs** → `results/score_results/{model_type}/{suffix}/` and `results/ceiling_results/`

See `run_analysis.sh`, `run_analysis_localized.sh`, and `run_ceiling.sh` for batch commands.

---

### NARI: Intervention on fMRI questions

Compute brain-guided intervention directions using the NARI method.

```bash
CUDA_VISIBLE_DEVICES='0' python make_intervention_sep.py \
    -device cuda \
    -iteration 200 -lr 0.1 -iter_interval 5 \
    -model_type qwen1-5b \
    -model_path Qwen/Qwen2-1.5B-Instruct \
    -loss_type cosine \
    -analyze_type deductive_reasoning \
    -proj_alpha 10. \
    -use_ridgecv \
    -save_rep
```

**Key arguments:**
- `-loss_type`: Loss for direction computation (`cosine`, `mse`, `pearsonr`)
- `-proj_alpha`: Intervention scale factor
- `-iteration`: Number of optimization iterations
- `-lr`: Learning rate for iterative optimization
- `-iter_interval`: Save directions every N iterations
- `-save_rep`: Save intervened representations for downstream use
- `-random`: Use random directions as control
- `-random_fmri`: Replace fMRI with random signals (structure-only control); can further use `-with_fmri_mean` to keep the mean value the same as fMRI data
- `-add_intercept`: Include intercept term in the ridge regression mapping
- `-manual_seed`: Specify seed for randomness

**Outputs** → `results/intervention_sep_results/{model_type}/{suffix}/`

See `run_intervention.sh` for all commands.

---

### NARI (gen.): Generalization to new questions

Apply the saved intervention directions (from NARI) to new test problems. The directions are aggregated across fMRI questions and applied at varying scales.

```bash
CUDA_VISIBLE_DEVICES='0' python make_intervention_generaldir.py \
    -device cuda \
    -iteration 200 -lr 0.1 -iter_interval 5 \
    -model_type qwen1-5b \
    -model_path Qwen/Qwen2-1.5B-Instruct \
    -loss_type cosine \
    -analyze_type deductive_reasoning \
    -proj_alpha 10. \
    -use_ridgecv \
    -dir_scale_min 0. -dir_scale_max 1. -dir_scale_itv 0.1
```

**Key arguments:**
- `-dir_scale_min`, `-dir_scale_max`, `-dir_scale_itv`: Range and step for intervention scale search
- `-select_nearest`: Select intervention direction from the nearest fMRI problem
- `-normalize_dir`: Normalize direction vectors before aggregation
- `-DATA`: Path to test dataset (default: `./data/deductive_reasoning_data_test.json`)
- `-random`: Use random directions as control

**Outputs** → `results/intervention_sumdir_test/{model_type}/{suffix}/`

See `run_intervention_generaldir.sh` for all commands.

---

### NARF: Fine-tuning with brain guidance

Fine-tune the LLM using brain-guided representational supervision. Three training modes are supported:

#### NARF (representation supervision only)

Uses pre-computed intervention directions to supervise the model's attention representations:

```bash
CUDA_VISIBLE_DEVICES='0' python finetune_model_sep.py \
    -model_type qwen1-5b \
    -model_path Qwen/Qwen2-1.5B-Instruct \
    -use_intervention_info \
    -loss_type cosine -use_ridgecv \
    -reg_weight 10. -max_epoch 100 -val_epoch 10 \
    -lr 1e-6 -select_nearest -val_all_order \
    -device cuda
```

#### NARF+Label (representation + language supervision)

Combines brain-guided representation supervision with language label loss (using LoRA):

```bash
CUDA_VISIBLE_DEVICES='0' python finetune_model_sep.py \
    -model_type qwen1-5b \
    -model_path Qwen/Qwen2-1.5B-Instruct \
    -loss_type cosine -use_ridgecv \
    -train_weight 0.01 -reg_weight 0. \
    -use_label -label_loss_weight 1. \
    -max_epoch 100 -val_epoch 1 \
    -lora -lr 1e-4 -bf16 \
    -grad_accumulate_step 70 -val_all_order \
    -device cuda -delete_checkpoint
```

#### Label only (baseline)

Standard supervised fine-tuning with language labels only:

```bash
CUDA_VISIBLE_DEVICES='0' python finetune_model_sep_onlylabel.py \
    -model_type qwen1-5b \
    -model_path Qwen/Qwen2-1.5B-Instruct \
    -epoch 100 -val_epoch 1 \
    -lora -lr 1e-4 -bf16 \
    -grad_accumulate_step 70 -val_all_order \
    -device cuda -delete_checkpoint
```

#### Synthetic data experiments

Fine-tuning can also be run with synthetic training data using `finetune_model_sep_syntheticdata.py` and `finetune_model_sep_onlylabel_syntheticdata.py`. Specify `-synthetic_data_file` to point to the desired training set.

**Key arguments for fine-tuning:**
- `-use_intervention_info`: Use NARI directions (NARF-only mode)
- `-use_label`: Add language label supervision (NARF+Label)
- `-lora`: Use LoRA for parameter-efficient fine-tuning (r=8, targeting q/k/v/o projections)
- `-train_weight`: Weight for brain representation loss
- `-reg_weight`: Regularization weight
- `-label_loss_weight`: Weight for language label cross-entropy loss
- `-grad_accumulate_step`: Gradient accumulation steps
- `-bf16`: Use bfloat16 mixed precision
- `-forward_partial`: Accelerate by only computing forward pass up to the last supervised layer (depends on transformers version)
- `-val_all_order`: Validate on all premise orderings
- `-delete_checkpoint`: Remove checkpoint after training
- `-seed`: Random seed
- `-use_random_fmri`: Replace fMRI with random signals (ablation)
- `-use_label_as_fmri`: Replace fMRI with one-hot labels (ablation)

**Outputs** → `results/finetune_results/{model_type}/{suffix}/`

See `run_finetune.sh` for all commands including ablation experiments.

---

### Testing

Evaluate original or fine-tuned models on generated deductive reasoning test sets.

**Standard test (3 premises):**

```bash
# Test original model
CUDA_VISIBLE_DEVICES='0' python test_model.py \
    -model_type llama3 \
    -model_path meta-llama/Meta-Llama-3-8B-Instruct \
    -device cuda -test_all_order -suffix original

# Test fine-tuned model (use the saved model path)
CUDA_VISIBLE_DEVICES='0' python test_model.py \
    -model_type llama3 \
    -model_path ./results/finetune_results/llama3/useintvinfo_*/model/ \
    -device cuda -test_all_order -suffix narf
```

**Generalization to more premises:**

```bash
CUDA_VISIBLE_DEVICES='0' python test_model.py \
    -model_type llama3 \
    -model_path meta-llama/Meta-Llama-3-8B-Instruct \
    -device cuda -test_all_order -suffix original \
    -DATA ./data/deductive_reasoning_data_test_4premises.json -num_premises 4
```

Supports `-num_premises 4`, `5`, or `6` with the corresponding data files.

**Propositional reasoning:**

```bash
CUDA_VISIBLE_DEVICES='0' python test_model_analysis_propositional.py \
    -model_type llama3 \
    -model_path meta-llama/Meta-Llama-3-8B-Instruct \
    -device cuda
```

**Key arguments:**
- `-test_all_order`: Test all permutations of premise ordering (robustness evaluation)
- `-num_premises`: Number of premises (3, 4, 5, or 6)
- `-DATA`: Path to test dataset
- `-test_batch_size`: Batch size for inference
- `-suffix`: Identifier for the output directory

**Outputs** → `results/test_results/{model_type}_{suffix}/`

See `run_test.sh` for all commands.

---

## HCP relational processing experiment

The HCP (Human Connectome Project) experiment validates our approach on an independent fMRI dataset using relational processing tasks. The pipeline mirrors the main experiment:

```bash
# 1. Extract activations
CUDA_VISIBLE_DEVICES='0' python hcp_get_activations.py \
    -model_type phi4-mini \
    -model_path microsoft/Phi-4-mini-instruct \
    -device cuda

# 2. NARI: Make intervention on fMRI questions
CUDA_VISIBLE_DEVICES='0' python hcp_make_intervention.py \
    -device cuda -iteration 50 -lr 0.1 -iter_interval 5 \
    -model_type phi4-mini \
    -model_path microsoft/Phi-4-mini-instruct \
    -loss_type cosine -proj_alpha 5 -use_ridgecv \
    -save_rep -save_index

# 3. NARI (gen.): Apply directions to new relational problems
CUDA_VISIBLE_DEVICES='0' python hcp_make_intervention_generaldir.py \
    -device cuda -iteration 50 -lr 0.1 -iter_interval 5 \
    -model_type phi4-mini \
    -model_path microsoft/Phi-4-mini-instruct \
    -loss_type cosine -proj_alpha 5.0 -use_ridgecv \
    -dir_scale_min 0. -dir_scale_max 10.0 -dir_scale_itv 0.5 \
    -normalize_dir

# 4. NARF+Label fine-tuning
python hcp_finetune_model.py \
    -model_type phi4-mini \
    -model_path microsoft/Phi-4-mini-instruct \
    -device cuda \
    -max_epoch 100 -val_epoch 1 -val_all_order \
    -train_weight 0.1 -test_batch_size 20 \
    -use_label -use_fmri -loss_type cosine -use_ridgecv \
    -use_all_stimuli -log_stdout \
    -lora -bf16 -lr 1e-4 -grad_accumulate_step 96 \
    -filter_subject \
    -use_intervention_filter -INTERVENTION_RESULTS_DIR ./hcp_results/intervention_results/phi4-mini/*/
```

See `hcp_run_*.sh` scripts for complete commands for all models.

## Contact
If you have any questions, please contact <mingqing_xiao@pku.edu.cn>.
