analyze_types=("deductive_reasoning" "language" "md")

# all questions
for analyze in "${analyze_types[@]}"; do
    echo "Calculating ceiling for analyze_type=$analyze, all questions"
    python calculate_ceiling.py -analyze_type "$analyze" -use_ridgecv
done

# syllogisms questions
for analyze in "${analyze_types[@]}"; do
    echo "Calculating ceiling for analyze_type=$analyze, syllogisms questions"
    python calculate_ceiling.py -analyze_type "$analyze" -use_ridgecv -only_syllogisms
done

# transitive questions
for analyze in "${analyze_types[@]}"; do
    echo "Calculating ceiling for analyze_type=$analyze, transitive questions"
    python calculate_ceiling.py -analyze_type "$analyze" -use_ridgecv -only_transitive
done
