# example: Phi-4-mini
for x in 1 2 3 4 5 6; do
    CUDA_VISIBLE_DEVICES='0' python hcp_make_intervention.py -device cuda -iteration 50 -lr 0.1 -iter_interval 5 -model_type phi4-mini -model_path microsoft/Phi-4-mini-instruct -loss_type 'cosine' -proj_alpha "$x" -use_ridgecv -save_rep -save_index;
done
for x in 1 2 3 4 5 6; do
    CUDA_VISIBLE_DEVICES='0' python hcp_make_intervention.py -device cuda -iteration 50 -lr 0.1 -iter_interval 5 -model_type phi4-mini -model_path microsoft/Phi-4-mini-instruct -loss_type 'cosine' -proj_alpha "$x" -use_ridgecv -random;
done
for x in 1 2 3 4 5 6; do
    CUDA_VISIBLE_DEVICES='0' python hcp_make_intervention.py -device cuda -iteration 50 -lr 0.1 -iter_interval 5 -model_type phi4-mini -model_path microsoft/Phi-4-mini-instruct -loss_type 'cosine' -proj_alpha "$x" -use_ridgecv -random_fmri;
done

# other models
CUDA_VISIBLE_DEVICES='0' python hcp_make_intervention.py -device cuda -iteration 50 -lr 0.1 -iter_interval 5 -model_type qwen7b -model_path Qwen/Qwen2-7B-Instruct -loss_type 'cosine' -proj_alpha 3 -use_ridgecv -save_rep -save_index;
CUDA_VISIBLE_DEVICES='0' python hcp_make_intervention.py -device cuda -iteration 50 -lr 0.1 -iter_interval 5 -model_type mistral -model_path mistralai/Mistral-7B-Instruct-v0.2 -loss_type 'cosine' -proj_alpha 3 -use_ridgecv -save_rep -save_index;
CUDA_VISIBLE_DEVICES='0' python hcp_make_intervention.py -device cuda -iteration 50 -lr 0.1 -iter_interval 5 -model_type llama3 -model_path meta-llama/Meta-Llama-3-8B-Instruct -loss_type 'cosine' -proj_alpha 3 -use_ridgecv -save_rep -save_index;
CUDA_VISIBLE_DEVICES='0' python hcp_make_intervention.py -device cuda -iteration 50 -lr 0.1 -iter_interval 5 -model_type gemma2_9b -model_path google/gemma-2-9b-it -loss_type 'cosine' -proj_alpha 3 -use_ridgecv -save_rep -save_index;