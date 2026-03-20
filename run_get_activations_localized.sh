CUDA_VISIBLE_DEVICES='0' python get_activations_localized.py -model_type qwen1-5b -model_path Qwen/Qwen2-1.5B-Instruct -device cuda;
CUDA_VISIBLE_DEVICES='0' python get_activations_localized.py -model_type qwen7b -model_path Qwen/Qwen2-7B-Instruct -device cuda;
CUDA_VISIBLE_DEVICES='0' python get_activations_localized.py -model_type mistral -model_path mistralai/Mistral-7B-Instruct-v0.2 -device cuda;
CUDA_VISIBLE_DEVICES='0' python get_activations_localized.py -model_type llama2 -model_path meta-llama/Llama-2-7b-chat-hf -device cuda; #-llama2_type
CUDA_VISIBLE_DEVICES='0' python get_activations_localized.py -model_type llama3 -model_path meta-llama/Meta-Llama-3-8B-Instruct -device cuda;
python get_activations_localized.py -model_type qwen72b -model_path /data3/huggingface/Qwen2-72B-Instruct/;
python get_activations_localized.py -model_type llama3-3_70b -model_path /data3/huggingface/Llama-3.3-70B-Instruct/;
CUDA_VISIBLE_DEVICES='0' python get_activations_localized.py -model_type phi4-mini -model_path microsoft/Phi-4-mini-instruct -device cuda;
CUDA_VISIBLE_DEVICES='0' python get_activations_localized.py -model_type gemma2_9b -model_path google/gemma-2-9b-it -device cuda;
CUDA_VISIBLE_DEVICES='0' python get_activations_localized.py -model_type qwen3_4b -model_path Qwen/Qwen3-4B -device cuda;

# untrained model example
CUDA_VISIBLE_DEVICES='0' python get_activations_localized.py -model_type qwen1-5b -model_path Qwen/Qwen2-1.5B-Instruct -device cuda -untrained;