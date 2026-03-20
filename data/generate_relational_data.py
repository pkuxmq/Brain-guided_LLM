import random
import json
import os

random.seed(42)

ATTRIBUTES = {
    'color': ["red", "green", "blue", "yellow", "black", "white"],
    'quality': ["normal", "noisy", "blurred", "underexposed", "overexposed", "compressed"],
    'geometry': ["rotated", "translated", "zoomed_in", "zoomed_out", "flipped", "sheared"],
    'style': ["sketch", "watercolor", "oil_painting", "digital_art", "cartoon", "natural_photo"]
}

ATTR_KEYS = list(ATTRIBUTES.keys())

def generate_relational_item(category, active_attrs):
    """
    Generates a single relational reasoning item.
    category: 'Dim1_Yes', 'Dim1_No', 'Dim2_Yes', 'Dim2_No'
              Dim1 corresponds to active_attrs[0]
              Dim2 corresponds to active_attrs[1]
    active_attrs: list of 2 attribute names, e.g. ['color', 'style']
    """
    attr1_name = active_attrs[0]
    attr2_name = active_attrs[1]
    
    vals1 = ATTRIBUTES[attr1_name]
    vals2 = ATTRIBUTES[attr2_name]
    
    # --- Generate Image A and Image B ---
    # They must differ in exactly one dimension based on the category's prefix (Dim1 or Dim2) implies 
    # checking logic, but actually the category defines the logical condition.
    # Standard Relational Task structure:
    # A and B differ in X. 
    # Question: Do C and D differ in X?
    
    # We interpret the categories as:
    # Dim1_Yes: A/B diff in Dim1. C/D diff in Dim1. (Ans: Yes)
    # Dim1_No:  A/B diff in Dim1. C/D diff in Dim2. (Ans: No)
    # Dim2_Yes: A/B diff in Dim2. C/D diff in Dim2. (Ans: Yes)
    # Dim2_No:  A/B diff in Dim2. C/D diff in Dim1. (Ans: No)
    
    # Select values for A
    a_val1 = random.choice(vals1)
    a_val2 = random.choice(vals2)
    
    # Setup B based on "Reference Relation" (A vs B)
    if 'Dim1' in category:
        # Diff in Dim1, Same in Dim2
        # Pick b_val1 != a_val1
        b_val1 = random.choice([x for x in vals1 if x != a_val1])
        b_val2 = a_val2
        diff_attr_ab = attr1_name
    else: # Dim2
        # Same in Dim1, Diff in Dim2
        b_val1 = a_val1
        b_val2 = random.choice([x for x in vals2 if x != a_val2])
        diff_attr_ab = attr2_name
        
    # Setup C and D based on Answer (Yes/No)
    # Yes -> Match the relation of A/B
    # No  -> Mismatch (Inverse the relation)
    
    # Select C values independently
    c_val1 = random.choice(vals1)
    c_val2 = random.choice(vals2)
    
    label = 1 if 'Yes' in category else 0
    
    def get_diff_val(curr, opts):
        return random.choice([x for x in opts if x != curr])

    # Determine D values
    # For property A/B differ in (diff_attr_ab):
    #   If Yes: C/D differ
    #   If No:  C/D same
    # For the other property:
    #   Randomly same (0.5) or different (0.5)
    
    if diff_attr_ab == attr1_name:
        # Target: Dim1 (attr1)
        # Other:  Dim2 (attr2)
        
        # Target Logic
        if label == 1:
             d_val1 = get_diff_val(c_val1, vals1)
        else:
             d_val1 = c_val1
             
        # Other Logic
        if random.random() < 0.5:
             d_val2 = c_val2
        else:
             d_val2 = get_diff_val(c_val2, vals2)
             
    else:
        # Target: Dim2 (attr2)
        # Other:  Dim1 (attr1)
        
        # Target Logic
        if label == 1:
             d_val2 = get_diff_val(c_val2, vals2)
        else:
             d_val2 = c_val2
             
        # Other Logic
        if random.random() < 0.5:
             d_val1 = c_val1
        else:
             d_val1 = get_diff_val(c_val1, vals1)

    # Construct positions dict
    # Using Generic 'attr1', 'attr2' keys might be confusing if we don't know which is which in the prompt generator
    # Better to use specific names 'color', 'style', etc.
    
    positions = {
        'TopLeft': {attr1_name: a_val1, attr2_name: a_val2},     # A
        'TopRight': {attr1_name: b_val1, attr2_name: b_val2},    # B
        'BottomLeft': {attr1_name: c_val1, attr2_name: c_val2},  # C
        'BottomRight': {attr1_name: d_val1, attr2_name: d_val2}  # D
    }
    
    return {
        'positions': positions,
        'label': label,
        'active_attributes': active_attrs,
        'category': category,
        'diff_attr_ab': diff_attr_ab
    }

def generate_dataset(num_per_category=100):
    dataset = []
    categories = ['Dim1_Yes', 'Dim1_No', 'Dim2_Yes', 'Dim2_No']
    
    for _ in range(num_per_category):
        # For each set of 4 categories, we can randomise the attributes pair?
        # Or should we balance attribute pairs?
        # Let's randomise pairs for each item to ensure diversity.
        
        # Pick 2 different attributes
        active_attrs = random.sample(ATTR_KEYS, 2)
        
        for cat in categories:
            item = generate_relational_item(cat, active_attrs)
            dataset.append(item)
            
    random.shuffle(dataset)
    return dataset

def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    output_dir = './data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Generating Test Set...")
    test_data = generate_dataset(num_per_category=100) # 400 total
    save_jsonl(test_data, os.path.join(output_dir, 'relational_test_set.jsonl'))
    print(f"Saved {len(test_data)} items to relational_test_set.jsonl")
    
    print("Generating Validation Set...")
    val_data = generate_dataset(num_per_category=20) # 80 total
    save_jsonl(val_data, os.path.join(output_dir, 'relational_val_set.jsonl'))
    print(f"Saved {len(val_data)} items to relational_val_set.jsonl")