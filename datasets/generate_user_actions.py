# data_generator_logistic.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_action_data(n_samples=10000,
                         center_coords=(5, 5), # Center of the 'action=1' cluster
                         cluster_radius_mean=2.0, # Average radius of the 'action=1' cluster
                         cluster_radius_std=0.5, # Spread of the 'action=1' cluster radius
                         surround_radius_min=3.0, # Min radius for 'action=0' points
                         surround_radius_max=8.0, # Max radius for 'action=0' points
                         noise_level=0.4, # General noise/overlap level
                         action_ratio=0.4, # Proportion of samples with action=1
                         random_state=42):
    """
    Generates a dataset for user action prediction with a non-linear boundary.
    Points for action=1 form a cluster around center_coords, surrounded by action=0 points.

    Args:
        n_samples (int): Total number of samples.
        center_coords (tuple): (x, y) coordinates for the center of the action=1 cluster.
        cluster_radius_mean (float): Mean radius for action=1 points.
        cluster_radius_std (float): Std dev of radius for action=1 points.
        surround_radius_min (float): Minimum radius for action=0 points.
        surround_radius_max (float): Maximum radius for action=0 points.
        noise_level (float): Standard deviation of Gaussian noise added to point positions.
        action_ratio (float): Approximate proportion of samples where action_taken=1.
        random_state (int): Seed for reproducibility.

    Returns:
        pandas.DataFrame: Generated dataset with columns
                          ['metric_A', 'metric_B', 'action_taken'].
    """
    np.random.seed(random_state)
    center_x, center_y = center_coords
    n_action_1 = int(n_samples * action_ratio)
    n_action_0 = n_samples - n_action_1

    # 1. Generate points for action_taken = 1 (Inner Cluster)
    angles_1 = np.random.uniform(0, 2 * np.pi, n_action_1)
    radii_1 = np.random.normal(cluster_radius_mean, cluster_radius_std, n_action_1)
    radii_1 = np.clip(radii_1, 0, None) # Ensure radius is non-negative

    x_1 = center_x + radii_1 * np.cos(angles_1) + np.random.normal(0, noise_level, n_action_1)
    y_1 = center_y + radii_1 * np.sin(angles_1) + np.random.normal(0, noise_level, n_action_1)
    label_1 = np.ones(n_action_1, dtype=int)

    # 2. Generate points for action_taken = 0 (Surrounding Ring/Cloud)
    angles_0 = np.random.uniform(0, 2 * np.pi, n_action_0)
    # Sample radii ensuring they are mostly outside the cluster radius
    radii_0 = np.random.uniform(surround_radius_min, surround_radius_max, n_action_0)

    x_0 = center_x + radii_0 * np.cos(angles_0) + np.random.normal(0, noise_level, n_action_0)
    y_0 = center_y + radii_0 * np.sin(angles_0) + np.random.normal(0, noise_level, n_action_0)
    label_0 = np.zeros(n_action_0, dtype=int)

    # 3. Combine and Create DataFrame
    metrics_A = np.concatenate([x_1, x_0])
    metrics_B = np.concatenate([y_1, y_0])
    actions = np.concatenate([label_1, label_0])

    df = pd.DataFrame({
        'metric_A': metrics_A,
        'metric_B': metrics_B,
        'action_taken': actions
    })

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("Dataset Generation Complete.")
    print(f"Total samples: {len(df)}")
    print(f"Action=1 samples: {n_action_1} ({action_ratio*100:.1f}%)")
    print(f"Action=0 samples: {n_action_0}")
    print("\nDataset Head:")
    print(df.head())
    print("\nBasic Statistics:")
    print(df.describe())

    return df

# --- Parameters ---
N_SAMPLES_LOG = 10000
CENTER = (5, 5)
CLUSTER_RAD_MEAN = 2.0
CLUSTER_RAD_STD = 0.6 # Increased std dev slightly for more overlap
SURROUND_MIN = 3.0  # Start ring just outside the main cluster mean
SURROUND_MAX = 8.0
NOISE_LOG = 0.5      # Noise level for overlap
ACTION_1_RATIO = 0.35 # ~35% take action
RANDOM_SEED_LOG = 42
OUTPUT_FILENAME_LOG = "user_actions.csv"
PLOT_DATA_LOG = False # Set to True to visualize the generated data

# --- Generate Data ---
df_generated_log = generate_action_data(
    n_samples=N_SAMPLES_LOG,
    center_coords=CENTER,
    cluster_radius_mean=CLUSTER_RAD_MEAN,
    cluster_radius_std=CLUSTER_RAD_STD,
    surround_radius_min=SURROUND_MIN,
    surround_radius_max=SURROUND_MAX,
    noise_level=NOISE_LOG,
    action_ratio=ACTION_1_RATIO,
    random_state=RANDOM_SEED_LOG
)

# --- Save Data ---
df_generated_log.to_csv(OUTPUT_FILENAME_LOG, index=False)
print(f"\nDataset saved to {OUTPUT_FILENAME_LOG}")

# --- Optional Visualization ---
if PLOT_DATA_LOG:
    plt.figure(figsize=(8, 7))
    sns.scatterplot(data=df_generated_log, x='metric_A', y='metric_B',
                    hue='action_taken', palette='viridis', alpha=0.6)
    plt.title('Generated User Action Data (Encirclement)')
    plt.xlabel('Engagement Metric A')
    plt.ylabel('Engagement Metric B')
    plt.axhline(CENTER[1], color='grey', linestyle='--', alpha=0.5)
    plt.axvline(CENTER[0], color='grey', linestyle='--', alpha=0.5)
    plt.legend(title='Action Taken (1=Yes, 0=No)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()