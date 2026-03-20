# Qwen2-7B
model="qwen7b"
model_path="Qwen/Qwen2-7B-Instruct"
seed=0
# only label
python hcp_finetune_model.py -model_type $model -model_path $model_path -device cuda \
    -use_label -max_epoch 100 -val_epoch 1 -val_all_order -train_weight 0.0 -test_batch_size 20 -use_all_stimuli \
    -log_stdout -lora -bf16 -lr 1e-4 -grad_accumulate_step 96 -seed $seed;

# NARF+Label
python hcp_finetune_model.py -model_type $model -model_path $model_path -device cuda \
    -max_epoch 100 -val_epoch 1 -val_all_order -train_weight 0.05 -test_batch_size 20 \
    -use_label -use_fmri -loss_type cosine -use_ridgecv -use_all_stimuli \
    -log_stdout -lora -bf16 -lr 1e-4 -grad_accumulate_step 96 -seed $seed -filter_subject \
    -use_intervention_filter -INTERVENTION_RESULTS_DIR ./hcp_results/intervention_results/$model/*/;


# Mistral-7B
model="mistral"
model_path="mistralai/Mistral-7B-Instruct-v0.2"
seed=0
# only label
python hcp_finetune_model.py -model_type $model -model_path $model_path -device cuda \
    -use_label -max_epoch 100 -val_epoch 1 -val_all_order -train_weight 0.0 -test_batch_size 20 -use_all_stimuli \
    -log_stdout -lora -bf16 -lr 1e-4 -grad_accumulate_step 96 -seed $seed;

# NARF+Label
python hcp_finetune_model.py -model_type $model -model_path $model_path -device cuda \
    -max_epoch 100 -val_epoch 1 -val_all_order -train_weight 0.01 -test_batch_size 20 \
    -use_label -use_fmri -loss_type cosine -use_ridgecv -use_all_stimuli \
    -log_stdout -lora -bf16 -lr 1e-4 -grad_accumulate_step 96 -seed $seed -filter_subject \
    -use_intervention_filter -INTERVENTION_RESULTS_DIR ./hcp_results/intervention_results/$model/*/;


# Llama3-8B
model="llama3"
model_path="meta-llama/Meta-Llama-3-8B-Instruct"
seed=0
# only label
python hcp_finetune_model.py -model_type $model -model_path $model_path -device cuda \
    -use_label -max_epoch 100 -val_epoch 1 -val_all_order -train_weight 0.0 -test_batch_size 20 -use_all_stimuli \
    -log_stdout -lora -bf16 -lr 1e-4 -grad_accumulate_step 96 -seed $seed;

# NARF+Label
python hcp_finetune_model.py -model_type $model -model_path $model_path -device cuda \
    -max_epoch 100 -val_epoch 1 -val_all_order -train_weight 0.1 -test_batch_size 20 \
    -use_label -use_fmri -loss_type cosine -use_ridgecv -use_all_stimuli \
    -log_stdout -lora -bf16 -lr 1e-4 -grad_accumulate_step 96 -seed $seed -filter_subject \
    -use_intervention_filter -INTERVENTION_RESULTS_DIR ./hcp_results/intervention_results/$model/*/;


# Phi-4-mini
model="phi4-mini"
model_path="microsoft/Phi-4-mini-instruct"
seed=0
# only label
python hcp_finetune_model.py -model_type $model -model_path $model_path -device cuda \
    -use_label -max_epoch 100 -val_epoch 1 -val_all_order -train_weight 0.0 -test_batch_size 20 -use_all_stimuli \
    -log_stdout -lora -bf16 -lr 1e-4 -grad_accumulate_step 96 -seed $seed;

# NARF+Label
python hcp_finetune_model.py -model_type $model -model_path $model_path -device cuda \
    -max_epoch 100 -val_epoch 1 -val_all_order -train_weight 0.1 -test_batch_size 20 \
    -use_label -use_fmri -loss_type cosine -use_ridgecv -use_all_stimuli \
    -log_stdout -lora -bf16 -lr 1e-4 -grad_accumulate_step 96 -seed $seed -filter_subject \
    -use_intervention_filter -INTERVENTION_RESULTS_DIR ./hcp_results/intervention_results/$model/*/;


# Gemma-2-9B
model="gemma2_9b"
model_path="google/gemma-2-9b-it"
seed=0
# only label
python hcp_finetune_model.py -model_type $model -model_path $model_path -device cuda \
    -use_label -max_epoch 100 -val_epoch 1 -val_all_order -train_weight 0.0 -test_batch_size 20 -use_all_stimuli \
    -log_stdout -lora -bf16 -lr 1e-4 -grad_accumulate_step 96 -seed $seed;

# NARF+Label
python hcp_finetune_model.py -model_type $model -model_path $model_path -device cuda \
    -max_epoch 100 -val_epoch 1 -val_all_order -train_weight 0.01 -test_batch_size 20 \
    -use_label -use_fmri -loss_type cosine -use_ridgecv -use_all_stimuli \
    -log_stdout -lora -bf16 -lr 1e-4 -grad_accumulate_step 96 -seed $seed -filter_subject \
    -use_intervention_filter -INTERVENTION_RESULTS_DIR ./hcp_results/intervention_results/$model/*/;