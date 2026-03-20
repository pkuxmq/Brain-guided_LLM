# qwen2-1.5b
CUDA_VISIBLE_DEVICES='0' python make_intervention_sep.py -device cuda -iteration 200 -lr 0.1 -iter_interval 5 -model_type qwen1-5b -model_path /data3/huggingface/Qwen2-1.5B-Instruct/ -loss_type 'cosine' -analyze_type deductive_reasoning -proj_alpha 10. -use_ridgecv -save_rep;
# example: different alpha
for x in {1..10}; do
    CUDA_VISIBLE_DEVICES='0' python make_intervention_sep.py -device cuda -iteration 200 -lr 0.1 -iter_interval 5 -model_type qwen1-5b -model_path /data3/huggingface/Qwen2-1.5B-Instruct/ -loss_type 'cosine' -analyze_type deductive_reasoning -proj_alpha "$x" -use_ridgecv;
done
# use -random for random directions; use -random_fmri for only structure of model representations (replace fmri by random signals), can further use -with_fmri_mean to keep the mean value the same as fmri data
# can use -manual_seed to specify seed for randomness
# replace -loss_type (cosine/mse/pearsonr) or -analyze_type (deductive_reasoning, md, language) for analysis
# can use -add_intercept to include the intercept term during direction calculation

# other models
CUDA_VISIBLE_DEVICES='0,1' python make_intervention_sep.py -device cuda -iteration 200 -lr 0.1 -iter_interval 5 -model_type qwen7b -model_path /data3/huggingface/Qwen2-7B-Instruct/ -loss_type 'cosine' -analyze_type deductive_reasoning -proj_alpha 10. -use_ridgecv -save_rep;
CUDA_VISIBLE_DEVICES='0,1' python make_intervention_sep.py -device cuda -iteration 200 -lr 0.1 -iter_interval 5 -model_type mistral -model_path /data3/huggingface/Mistral-7B-Instruct-v0.2/ -loss_type 'cosine' -analyze_type deductive_reasoning -proj_alpha 10. -use_ridgecv -save_rep;
CUDA_VISIBLE_DEVICES='0,1' python make_intervention_sep.py -device cuda -iteration 200 -lr 0.1 -iter_interval 5 -model_type llama2 -model_path /data3/huggingface/Llama-2-7b-chat-hf/ -loss_type 'cosine' -analyze_type deductive_reasoning -proj_alpha 10. -use_ridgecv -save_rep;
CUDA_VISIBLE_DEVICES='0,1' python make_intervention_sep.py -device cuda -iteration 200 -lr 0.1 -iter_interval 5 -model_type llama3 -model_path /data3/huggingface/Meta-Llama-3-8B-Instruct/ -loss_type 'cosine' -analyze_type deductive_reasoning -proj_alpha 10. -use_ridgecv -save_rep;
CUDA_VISIBLE_DEVICES='0,1' python make_intervention_sep.py -device cuda -iteration 200 -lr 0.1 -iter_interval 5 -model_type phi4-mini -model_path /data3/huggingface/Phi-4-mini-instruct/ -loss_type 'cosine' -analyze_type deductive_reasoning -proj_alpha 10. -use_ridgecv -save_rep;

# deepseek-qwen-1.5b
CUDA_VISIBLE_DEVICES='0' python make_intervention_sep.py -device cuda -iteration 50 -lr 0.1 -iter_interval 5 -model_type deepseekqwen1-5b -model_path /data3/huggingface/DeepSeek-R1-Distill-Qwen-1.5B/ -loss_type 'cosine' -analyze_type deductive_reasoning -proj_alpha 5. -use_ridgecv -save_rep;
