model_types=("qwen1-5b" "qwen7b" "qwen72b" "qwen3_4b" "mistral" "llama2" "llama3" "llama3-3_70b" "phi4-mini" "gemma2_9b" "untrained/qwen1-5b" "untrained/qwen7b" "untrained/qwen72b" "untrained/qwen3_4b" "untrained/mistral" "untrained/llama2" "untrained/llama3" "untrained/llama3-3_70b" "untrained/phi4-mini" "untrained/gemma2_9b" "qwen7b_base" "llama2_base")
analyze_types=("deductive_reasoning" "language" "md")

# all questions
for model in "${model_types[@]}"; do
    for analyze in "${analyze_types[@]}"; do
        python make_analysis.py -model_type "$model" -analyze_type "$analyze" -use_ridgecv
    done
done

# syllogisms questions
for model in "${model_types[@]}"; do
    for analyze in "${analyze_types[@]}"; do
        python make_analysis.py -model_type "$model" -analyze_type "$analyze" -use_ridgecv -only_syllogisms
    done
done

# transitive questions
for model in "${model_types[@]}"; do
    for analyze in "${analyze_types[@]}"; do
        python make_analysis.py -model_type "$model" -analyze_type "$analyze" -use_ridgecv -only_transitive
    done
done

# max of random feature
for model in "${model_types[@]}"; do
    for analyze in "${analyze_types[@]}"; do
        python make_analysis.py -model_type "$model" -analyze_type "$analyze" -use_ridgecv -rand_feature
    done
done
