import random
import json

random.seed(0)

# Syllogisms
# Step 1: List of adjectives
adjectives = [
    "strong", "weak", "bright", "dark", "warm", "cold", "soft", "hard", 
    "smooth", "rough", "quiet", "loud", "sharp", "dull", "heavy", "light", 
    "wide", "narrow", "deep", "shallow", "clean", "dirty", "fresh", "stale",
    "rich", "poor", "brave", "cowardly", "cheerful", "sad", "proud", "humble"
]

# Step 2: Function to generate monosyllabic pseudowords
def generate_pseudoword():
    consonants = "bcdfghjklmnpqrstvwxyz"
    consonants_ = "bcdfghjklmnpqrtvwyz"
    vowels = "aeiou"
    pseudoword = random.choice(consonants) + random.choice(vowels) + random.choice(consonants_)
    return pseudoword

# Step 3: Deductive reasoning problem generator
def generate_propositional_problem(category):
    pseudoword1 = generate_pseudoword()
    pseudoword2 = generate_pseudoword()
    adj1, adj2 = random.sample(adjectives, 2)

    # Generate premises

    # modus ponens
    if category == 'modus_ponens_true':
        premise1 = f"If {pseudoword1} is {adj1}, then {pseudoword2} is {adj2}."
        premise2 = f"{pseudoword1} is {adj1}."
        conclusion = f"{pseudoword2} is {adj2}."
    elif category == 'modus_ponens_false':
        premise1 = f"If {pseudoword1} is {adj1}, then {pseudoword2} is {adj2}."
        premise2 = f"{pseudoword1} is {adj1}."
        conclusion = f"{pseudoword2} is not {adj2}."
    # modus tollens
    elif category == 'modus_tollens_true':
        premise1 = f"If {pseudoword1} is {adj1}, then {pseudoword2} is {adj2}."
        premise2 = f"{pseudoword2} is not {adj2}."
        conclusion = f"{pseudoword1} is not {adj1}."
    elif category == 'modus_tollens_false':
        premise1 = f"If {pseudoword1} is {adj1}, then {pseudoword2} is {adj2}."
        premise2 = f"{pseudoword2} is not {adj2}."
        conclusion = f"{pseudoword1} is {adj1}."
    # disjunction elimination
    elif category == 'disjunction_elimination_true':
        premise1 = f"{pseudoword1} is {adj1} or {pseudoword2} is {adj2}."
        premise2 = f"{pseudoword1} is not {adj1}."
        conclusion = f"{pseudoword2} is {adj2}"
    elif category == 'disjunction_elimination_false':
        premise1 = f"{pseudoword1} is {adj1} or {pseudoword2} is {adj2}."
        premise2 = f"{pseudoword1} is not {adj1}."
        conclusion = f"{pseudoword2} is not {adj2}"
    else:
        raise(f"error for category {category}")
    

    return {
        "premise1": premise1,
        "premise2": premise2,
        "conclusion": conclusion,
        "trial_type": category
    }

# Step 4: Generate a dataset of deductive reasoning problems
def generate_propositional_dataset(num_problems_each=50):
    categories = [
        "modus_ponens_true", "modus_ponens_false",
        "modus_tollens_true", "modus_tollens_false",
        "disjunction_elimination_true", "disjunction_elimination_false",
    ]
    
    dataset = []

    for category in categories:
        for _ in range(num_problems_each):
            pro = generate_propositional_problem(category)
            dataset.append(pro)
    
    return dataset

# Step 5: Save the dataset to a JSON file
def save_to_json(dataset, filename="deductive_reasoning_data.json"):
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=4)

# Generate the dataset
propositional_data = generate_propositional_dataset(100)  # Generate 100 problems for each category
data = {'propositional': propositional_data}
save_to_json(data, "deductive_reasoning_data_test_propositional.json")
