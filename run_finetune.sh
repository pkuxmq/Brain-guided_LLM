# NARF, require intervention results
# can use -forward_partial for acceleration, but is depends on transformers version
CUDA_VISIBLE_DEVICES='0' python finetune_model_sep.py -model_type qwen1-5b -model_path /data3/huggingface/Qwen2-1.5B-Instruct/ -use_intervention_info -loss_type cosine -use_ridgecv -reg_weight 10. -max_epoch 100 -val_epoch 10 -lr 1e-6 -select_nearest -val_all_order -device cuda;
CUDA_VISIBLE_DEVICES='0,1' python finetune_model_sep.py -model_type qwen7b -model_path /data3/huggingface/Qwen2-7B-Instruct/ -use_intervention_info -loss_type cosine -use_ridgecv -reg_weight 1. -max_epoch 100 -val_epoch 10 -lr 1e-6 -select_nearest -val_all_order -device cuda;
CUDA_VISIBLE_DEVICES='0,1' python finetune_model_sep.py -model_type mistral -model_path /data3/huggingface/Mistral-7B-Instruct-v0.2/ -use_intervention_info -loss_type cosine -use_ridgecv -reg_weight 10. -max_epoch 100 -val_epoch 10 -lr 1e-6 -select_nearest -val_all_order -device cuda;
CUDA_VISIBLE_DEVICES='0,1' python finetune_model_sep.py -model_type llama2 -model_path /data3/huggingface/Llama-2-7b-chat-hf/ -use_intervention_info -loss_type cosine -use_ridgecv -reg_weight 1. -max_epoch 100 -val_epoch 10 -lr 1e-6 -select_nearest -val_all_order -device cuda;
CUDA_VISIBLE_DEVICES='0,1' python finetune_model_sep.py -model_type llama3 -model_path /data3/huggingface/Meta-Llama-3-8B-Instruct/ -use_intervention_info -loss_type cosine -use_ridgecv -reg_weight 1. -max_epoch 100 -val_epoch 10 -lr 1e-6 -select_nearest -val_all_order -device cuda;


# NARF+Label
# can use -forward_partial for acceleration, but is depends on transformers version
# can use -test_origin to first test the performance of the original model
seed=0
CUDA_VISIBLE_DEVICES='0' python finetune_model_sep.py -model_type qwen1-5b -model_path /data3/huggingface/Qwen2-1.5B-Instruct/ -loss_type cosine -use_ridgecv -train_weight 0.01 -reg_weight 0. -use_label -label_loss_weight 1. -max_epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0,1' python finetune_model_sep.py -model_type qwen7b -model_path /data3/huggingface/Qwen2-7B-Instruct/ -loss_type cosine -use_ridgecv -train_weight 0.1 -reg_weight 0. -use_label -label_loss_weight 1. -max_epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0,1' python finetune_model_sep.py -model_type mistral -model_path /data3/huggingface/Mistral-7B-Instruct-v0.2/ -loss_type cosine -use_ridgecv -train_weight 0.1 -reg_weight 0. -use_label -label_loss_weight 1. -max_epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -seed $seed;
# llama2 from NARF-trained model
CUDA_VISIBLE_DEVICES='0,1' python finetune_model_sep.py -model_type llama2 -model_path ./results/finetune_results/llama2/useintvinfo_*/model/ -loss_type cosine -use_ridgecv -train_weight 0.1 -reg_weight 0. -use_label -label_loss_weight 1. -max_epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -suffix 'resumemodel' -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0' python finetune_model_sep.py -model_type phi4-mini -model_path /data3/huggingface/Phi-4-mini-instruct/ -loss_type cosine -use_ridgecv -train_weight 0.1 -reg_weight 0. -use_label -label_loss_weight 1. -max_epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0,1,2,3' python finetune_model_sep.py -model_type gemma2_9b -model_path /data3/huggingface/gemma-2-9b-it/ -loss_type cosine -use_ridgecv -train_weight 0.1 -reg_weight 0. -use_label -label_loss_weight 1. -max_epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed -filter_sub_fmri;
CUDA_VISIBLE_DEVICES='0' python finetune_model_sep.py -model_type qwen3_4b -model_path /data3/huggingface/Qwen3-4B/ -loss_type cosine -use_ridgecv -train_weight 0.1 -reg_weight 0. -use_label -label_loss_weight 1. -max_epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed -filter_sub_fmri;
CUDA_VISIBLE_DEVICES='0,1' python finetune_model_sep.py -model_type llama3 -model_path /data3/huggingface/Meta-Llama-3-8B-Instruct/ -loss_type cosine -use_ridgecv -train_weight 0.1 -reg_weight 0. -use_label -label_loss_weight 1. -max_epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python finetune_model_sep.py -model_type qwen72b -model_path /data3/huggingface/Qwen2-72B-Instruct/ -loss_type cosine -use_ridgecv -train_weight 0.1 -reg_weight 0. -use_label -label_loss_weight 1. -max_epoch 50 -val_epoch 1 -lora -bf16 -lr 1e-4 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python finetune_model_sep.py -model_type llama3-3_70b -model_path /data3/huggingface/Llama-3.3-70B-Instruct/ -loss_type cosine -use_ridgecv -train_weight 0.1 -reg_weight 0. -use_label -label_loss_weight 1. -max_epoch 50 -val_epoch 1 -lora -bf16 -lr 1e-4 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed -filter_sub_fmri;


# only label
seed=0
CUDA_VISIBLE_DEVICES='0' python finetune_model_sep_onlylabel.py -model_type qwen1-5b -model_path /data3/huggingface/Qwen2-1.5B-Instruct/ -epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0,1' python finetune_model_sep_onlylabel.py -model_type qwen7b -model_path /data3/huggingface/Qwen2-7B-Instruct/ -epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0,1' python finetune_model_sep_onlylabel.py -model_type mistral -model_path /data3/huggingface/Mistral-7B-Instruct-v0.2/ -epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -seed $seed;
CUDA_VISIBLE_DEVICES='0,1' python finetune_model_sep_onlylabel.py -model_type llama2 -model_path /data3/huggingface/Llama-2-7b-chat-hf/ -epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0,1' python finetune_model_sep_onlylabel.py -model_type llama3 -model_path /data3/huggingface/Meta-Llama-3-8B-Instruct/ -epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0' python finetune_model_sep_onlylabel.py -model_type phi4-mini -model_path /data3/huggingface/Phi-4-mini-instruct/ -epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0,1,2,3' python finetune_model_sep_onlylabel.py -model_type gemma2_9b -model_path /data3/huggingface/gemma-2-9b-it/ -epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0' python finetune_model_sep_onlylabel.py -model_type qwen3_4b -model_path /data3/huggingface/Qwen3-4B/ -epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python finetune_model_sep_onlylabel.py -model_type qwen72b -model_path /data3/huggingface/Qwen2-72B-Instruct -epoch 50 -val_epoch 1 -lora -bf16 -lr 1e-4 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python finetune_model_sep_onlylabel.py -model_type llama3-3_70b -model_path /data3/huggingface/Llama-3.3-70B-Instruct/ -epoch 50 -val_epoch 1 -lora -bf16 -lr 1e-4 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -test_batch_size 20 -seed $seed;

# ablation experiments
# use -use_random_fmri to replace fmri by random signals, so that the method is only based on the structure of model representations
# use -use_label_as_fmri to replace fmri by one-hot labels

# synthetic data
seed=0
# 32 synthetic questions
scaled_train_weight=0.15 # 0.2, 0.25, 0.3, 0.35, 0.4 for 64, 96, 128, 160, 192
synthetic_data_file='./data/deductive_reasoning_data_train_32.json'
# NARF+Label
python finetune_model_sep_syntheticdata.py -model_type mistral -model_path /data3/huggingface/Mistral-7B-Instruct-v0.2/ -loss_type cosine -use_ridgecv -train_weight $scaled_train_weight -reg_weight 0. -use_label -label_loss_weight 1. -max_epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -seed $seed -synthetic_data_file $synthetic_data_file;
# only label
python finetune_model_sep_onlylabel_syntheticdata.py -model_type mistral -model_path /data3/huggingface/Mistral-7B-Instruct-v0.2/ -epoch 100 -val_epoch 1 -lora -lr 1e-4 -bf16 -grad_accumulate_step 70 -val_all_order -device cuda -delete_checkpoint -seed $seed -synthetic_data_file $synthetic_data_file;