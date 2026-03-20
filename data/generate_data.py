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

# Step 3: List of imaginary character names
characters = [
    "Zim", "Nod", "Mav", "Fen", "Luz", "Yor", "Rek", "Pal", "Vak", "Jiv", 
    "Sog", "Taz", "Wek", "Bim", "Lyn", "Jax", "Kor", "Dex", "Nim", "Tev",
    "Ryl", "Fex", "Bren", "Zol", "Mek", "Jor", "Cin", "Vek", "Lur", "Nix",
    "Xel", "Tor"
]


# Step 4: Deductive reasoning problem generator
def generate_syllogisms_problem(category):
    pseudoword = generate_pseudoword()
    adj1, adj2 = random.sample(adjectives, 2)
    character = random.choice(characters)

    # Generate premises based on category
    premise1 = f"All {pseudoword}s are {adj1}."
    premise2 = f"All {adj1} things are {adj2}."

    # true affirm
    if category == '2_true_affirm':
        if random.random() < 0.5:
            premise3 = f"{character} is a {pseudoword}."
            conclusion = f"{character} is {adj1}."
        else:
            # this condition is not in the neuroimaging dataset
            premise3 = f"{character} is {adj1}."
            conclusion = f"{character} is {adj2}."
    elif category == '3_true_affirm':
        premise3 = f"{character} is a {pseudoword}."
        conclusion = f"{character} is {adj2}."
    # false affirm
    elif category == '2_false_affirm':
        if random.random() < 0.5:
            premise3 = f"{character} is not {adj1}."
            conclusion = f"{character} is a {pseudoword}."
        else:
            # this condition is not in the neuroimaging dataset
            premise3 = f"{character} is not {adj2}."
            conclusion = f"{character} is {adj1}."
    elif category == '3_false_affirm':
        premise3 = f"{character} is not {adj2}."
        conclusion = f"{character} is a {pseudoword}."
    # true negate
    elif category == '2_true_negate':
        if random.random() < 0.5:
            premise3 = f"{character} is not {adj1}."
            conclusion = f"{character} is not a {pseudoword}."
        else:
            # this condition is not in the neuroimaging dataset
            premise3 = f"{character} is not {adj2}."
            conclusion = f"{character} is not {adj1}."
    elif category == '3_true_negate':
        premise3 = f"{character} is not {adj2}."
        conclusion = f"{character} is not a {pseudoword}."
    # false negate
    elif category == '2_false_negate':
        if random.random() < 0.5:
            conclusion = f"{character} is not {adj1}."
            if random.random() < 0.5:
                premise3 = f"{character} is a {pseudoword}."
            else:
                # this condition is not in the neuroimaging dataset
                premise3 = f"{character} is not a {pseudoword}."
        else:
            # this condition is not in the neuroimaging dataset
            conclusion = f"{character} is not {adj2}."
            if random.random() < 0.5:
                premise3 = f"{character} is {adj1}."
            else:
                premise3 = f"{character} is not {adj1}."
    elif category == '3_false_negate':
        conclusion = f"{character} is not {adj2}."
        if random.random() < 0.5:
            premise3 = f"{character} is a {pseudoword}."
        else:
            premise3 = f"{character} is not a {pseudoword}."
    

    return {
        "premise1": premise1,
        "premise2": premise2,
        "premise3": premise3,
        "conclusion": conclusion,
        "trial_type": category
    }

# Step 5: Generate a dataset of deductive reasoning problems
def generate_syllogisms_dataset(num_problems_each=50):
    categories = [
        "2_true_affirm", "2_false_affirm",
        "3_true_affirm", "3_false_affirm",
        "2_true_negate", "2_false_negate",
        "3_true_negate", "3_false_negate"
    ]
    
    dataset = []

    for category in categories:
        for _ in range(num_problems_each):
            pro = generate_syllogisms_problem(category)
            dataset.append(pro)
    
    return dataset

# Step 6: Save the dataset to a JSON file
def save_to_json(dataset, filename="deductive_reasoning_data.json"):
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=4)

# Transitive
comparative_adjectives = [
    "brighter", "darker", "richer", "poorer", "stronger", "weaker", 
    "quieter", "louder", "cleaner", "dirtier", "fresher", "staler", 
    "heavier", "lighter", "smoother", "rougher", "warmer", "colder", 
    "wider", "narrower", "deeper", "shallower", "sharper", "duller", 
    "softer", "harder", "prouder", "humbler", "braver", "calmer", 
    "sadder", "happier"
]

def generate_transitive_problem(category):
    adj = random.choice(comparative_adjectives)
    c1, c2, c3, c4 = random.sample(characters, 4)

    # Generate premises based on category
    premise1 = f"{c1} is {adj} than {c2}."
    premise2 = f"{c2} is {adj} than {c3}."
    premise3 = f"{c3} is {adj} than {c4}."

    # true affirm
    if category == '2_true_affirm':
        if random.random() < 0.5:
            conclusion = f"{c1} is {adj} than {c3}."
        else:
            # this condition is not in the neuroimaging dataset
            conclusion = f"{c2} is {adj} than {c4}."
    elif category == '3_true_affirm':
        conclusion = f"{c1} is {adj} than {c4}."
    # false affirm
    elif category == '2_false_affirm':
        if random.random() < 0.5:
            conclusion = f"{c3} is {adj} than {c1}."
        else:
            # this condition is not in the neuroimaging dataset
            conclusion = f"{c4} is {adj} than {c2}."
    elif category == '3_false_affirm':
        conclusion = f"{c4} is {adj} than {c1}."
    # true negate
    elif category == '2_true_negate':
        if random.random() < 0.5:
            conclusion = f"{c3} is not {adj} than {c1}."
        else:
            # this condition is not in the neuroimaging dataset
            conclusion = f"{c4} is not {adj} than {c2}."
    elif category == '3_true_negate':
        conclusion = f"{c4} is not {adj} than {c1}."
    # false negate
    elif category == '2_false_negate':
        if random.random() < 0.5:
            conclusion = f"{c1} is not {adj} than {c3}."
        else:
            # this condition is not in the neuroimaging dataset
            conclusion = f"{c2} is not {adj} than {c4}."
    elif category == '3_false_negate':
        conclusion = f"{c1} is not {adj} than {c4}."
    

    return {
        "premise1": premise1,
        "premise2": premise2,
        "premise3": premise3,
        "conclusion": conclusion,
        "trial_type": category
    }

def generate_transitive_dataset(num_problems_each=50):
    categories = [
        "2_true_affirm", "2_false_affirm",
        "3_true_affirm", "3_false_affirm",
        "2_true_negate", "2_false_negate",
        "3_true_negate", "3_false_negate"
    ]
    
    dataset = []

    for category in categories:
        for _ in range(num_problems_each):
            pro = generate_transitive_problem(category)
            dataset.append(pro)
    
    return dataset

# Generate the dataset
syllogisms_data = generate_syllogisms_dataset(100)  # Generate 100 problems for each category
transitive_data = generate_transitive_dataset(100)  # Generate 100 problems for each category
data = {'syllogisms': syllogisms_data, 'transitive': transitive_data}
save_to_json(data, "deductive_reasoning_data_test.json")

syllogisms_data = generate_syllogisms_dataset(10)  # Generate 10 problems for each category
transitive_data = generate_transitive_dataset(10)  # Generate 10 problems for each category
data = {'syllogisms': syllogisms_data, 'transitive': transitive_data}
save_to_json(data, "deductive_reasoning_data_val_new.json")

