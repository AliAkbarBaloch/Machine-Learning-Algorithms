# dataset_generator_multinomial_nb.py
import numpy as np
import pandas as pd
import random

def generate_customer_support_data(n_samples=2000, urgent_ratio=0.4, random_state=42):
    """
    Generates a dataset for customer support message classification.

    Args:
        n_samples (int): Total number of samples.
        urgent_ratio (float): Proportion of messages that are urgent.
        random_state (int): Seed for reproducibility.

    Returns:
        pandas.DataFrame: Generated dataset with columns ['message_text', 'category'].
                          Category: 'Urgent Technical Issue' or 'General Inquiry'.
    """
    np.random.seed(random_state)
    random.seed(random_state)

    data = []
    num_urgent = int(n_samples * urgent_ratio)
    num_general = n_samples - num_urgent

    # Keywords
    urgent_strong = ["broken", "crash", "urgent", "immediately", "cannot access", "down", "fail", "critical error"]
    general_strong = ["information", "question", "details", "how to", "request", "feature inquiry", "pricing", "docs"]
    
    # Ambiguous keywords (the "curveball")
    ambiguous = ["update", "account", "service", "login", "payment", "issue", "problem", "help"]
    
    # Neutral filler words
    filler = ["the", "my", "is", "a", "please", "i need", "can you", "about", "experiencing", "regarding", "on"]

    # Generate Urgent Messages
    for _ in range(num_urgent):
        text_parts = []
        # Usually 1-2 strong urgent words
        text_parts.extend(random.sample(urgent_strong, random.randint(1, 2)))
        # Sometimes an ambiguous word
        if random.random() < 0.6:
            text_parts.append(random.choice(ambiguous))
        # Add some filler
        text_parts.extend(random.sample(filler, random.randint(2, 4)))
        random.shuffle(text_parts)
        data.append({"message_text": " ".join(text_parts), "category": "Urgent Technical Issue"})

    # Generate General Inquiry Messages
    for _ in range(num_general):
        text_parts = []
        # Usually 1-2 strong general words
        text_parts.extend(random.sample(general_strong, random.randint(1, 2)))
        # Sometimes an ambiguous word
        if random.random() < 0.6:
            text_parts.append(random.choice(ambiguous))
        # Add some filler
        text_parts.extend(random.sample(filler, random.randint(2, 4)))
        random.shuffle(text_parts)
        data.append({"message_text": " ".join(text_parts), "category": "General Inquiry"})
        
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True) # Shuffle

    print("Customer Support Message Dataset Generation Complete.")
    print(f"Total samples: {len(df)}")
    print(f"Urgent messages: {num_urgent}, General messages: {num_general}")
    print("\nDataset Head:")
    print(df.head())
    print("\nCategory Distribution:")
    print(df['category'].value_counts(normalize=True))
    
    return df

if __name__ == "__main__":
    N_SAMPLES_MNB = 2000
    URGENT_RATIO = 0.4 # Creates a slight imbalance
    RANDOM_SEED_MNB = 42
    OUTPUT_FILENAME_MNB = "customer_support_messages.csv"

    df_multinomial = generate_customer_support_data(
        n_samples=N_SAMPLES_MNB,
        urgent_ratio=URGENT_RATIO,
        random_state=RANDOM_SEED_MNB
    )
    df_multinomial.to_csv(OUTPUT_FILENAME_MNB, index=False)
    print(f"\nMultinomial Naive Bayes dataset saved to {OUTPUT_FILENAME_MNB}")

    # Example of how ambiguous words might appear:
    print("\nExamples containing 'account':")
    print(df_multinomial[df_multinomial['message_text'].str.contains("account")].head())
    print("\nExamples containing 'problem':")
    print(df_multinomial[df_multinomial['message_text'].str.contains("problem")].head())