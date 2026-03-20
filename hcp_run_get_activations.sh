CUDA_VISIBLE_DEVICES='0' python hcp_get_activations.py -model_type qwen7b -model_path Qwen/Qwen2-7B-Instruct -device cuda;
CUDA_VISIBLE_DEVICES='0' python hcp_get_activations.py -model_type mistral -model_path mistralai/Mistral-7B-Instruct-v0.2 -device cuda;
CUDA_VISIBLE_DEVICES='0' python hcp_get_activations.py -model_type llama3 -model_path meta-llama/Meta-Llama-3-8B-Instruct -device cuda;
CUDA_VISIBLE_DEVICES='0' python hcp_get_activations.py -model_type phi4-mini -model_path microsoft/Phi-4-mini-instruct -device cuda;
CUDA_VISIBLE_DEVICES='0' python hcp_get_activations.py -model_type gemma2_9b -model_path google/gemma-2-9b-it -device cuda;
