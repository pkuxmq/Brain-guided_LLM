# example
CUDA_VISIBLE_DEVICES='0,1' python test_model.py -model_type llama3 -model_path /data3/huggingface/Meta-Llama-3-8B-Instruct/ -device cuda -test_all_order -suffix original; # -test_batch_size 10
CUDA_VISIBLE_DEVICES='0,1' python test_model.py -model_type llama3 -model_path ./results/finetune_results/llama3/useintvinfo_*/model/ -device cuda -test_all_order -suffix narf; # -test_batch_size 10

# example more premises
CUDA_VISIBLE_DEVICES='0,1' python test_model.py -model_type llama3 -model_path /data3/huggingface/Meta-Llama-3-8B-Instruct/ -device cuda -test_all_order -suffix original -DATA ./data/deductive_reasoning_data_test_4premises.json -num_premises 4; # -test_batch_size 10
CUDA_VISIBLE_DEVICES='0,1' python test_model.py -model_type llama3 -model_path /data3/huggingface/Meta-Llama-3-8B-Instruct/ -device cuda -test_all_order -suffix original -DATA ./data/deductive_reasoning_data_test_5premises.json -num_premises 5; # -test_batch_size 10
CUDA_VISIBLE_DEVICES='0,1' python test_model.py -model_type llama3 -model_path /data3/huggingface/Meta-Llama-3-8B-Instruct/ -device cuda -test_all_order -suffix original -DATA ./data/deductive_reasoning_data_test_6premises.json -num_premises 6; # -test_batch_size 10

# example propositional
CUDA_VISIBLE_DEVICES='0,1' python test_model_analysis_propositional.py -model_type llama3 -model_path /data3/huggingface/Meta-Llama-3-8B-Instruct/ -device cuda;