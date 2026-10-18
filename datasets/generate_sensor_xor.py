# dataset_generator_gaussian_nb.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sensor_classification_data(n_samples_per_quadrant=250, gap=0.5, scale=2, random_state=42):
    """
    Generates a dataset for sensor classification with an XOR-like pattern.
    Class 0: clusters in bottom-left and top-right quadrants.
    Class 1: clusters in top-left and bottom-right quadrants.

    Args:
        n_samples_per_quadrant (int): Number of samples per sub-cluster.
        gap (float): Gap from the origin for cluster centers.
        scale (float): Standard deviation of clusters (controls overlap).
        random_state (int): Seed for reproducibility.

    Returns:
        pandas.DataFrame: Generated dataset with columns ['SensorX', 'SensorY', 'DeviceType'].
                          DeviceType: 0 or 1.
    """
    np.random.seed(random_state)
    
    # Class 0 centers
    center0_q1 = [-gap, -gap]  # Bottom-left
    center0_q2 = [gap, gap]    # Top-right
    
    # Class 1 centers
    center1_q1 = [-gap, gap]   # Top-left
    center1_q2 = [gap, -gap]   # Bottom-right

    # Generate data for Class 0
    class0_q1 = np.random.normal(loc=center0_q1, scale=scale, size=(n_samples_per_quadrant, 2))
    class0_q2 = np.random.normal(loc=center0_q2, scale=scale, size=(n_samples_per_quadrant, 2))
    X_class0 = np.vstack((class0_q1, class0_q2))
    y_class0 = np.zeros(len(X_class0))

    # Generate data for Class 1
    class1_q1 = np.random.normal(loc=center1_q1, scale=scale, size=(n_samples_per_quadrant, 2))
    class1_q2 = np.random.normal(loc=center1_q2, scale=scale, size=(n_samples_per_quadrant, 2))
    X_class1 = np.vstack((class1_q1, class1_q2))
    y_class1 = np.ones(len(X_class1))

    # Combine data
    X = np.vstack((X_class0, X_class1))
    y = np.concatenate((y_class0, y_class1))

    df = pd.DataFrame(X, columns=['SensorX', 'SensorY'])
    df['DeviceType'] = y.astype(int)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("Sensor Classification Dataset Generation Complete.")
    print(f"Total samples: {len(df)}")
    print("\nDataset Head:")
    print(df.head())
    print("\nDeviceType Distribution:")
    print(df['DeviceType'].value_counts(normalize=True))
    
    return df

if __name__ == "__main__":
    N_SAMPLES_PER_QUADRANT_GNB = 250 
    GAP_GNB = 2.0  # Increase gap to make separation clearer for visualization
    SCALE_GNB = 0.8 # Smaller scale for tighter clusters
    RANDOM_SEED_GNB = 42
    OUTPUT_FILENAME_GNB = "sensor_device_types_xor.csv"
    PLOT_DATA_GNB = True

    df_gaussian = generate_sensor_classification_data(
        n_samples_per_quadrant=N_SAMPLES_PER_QUADRANT_GNB,
        gap=GAP_GNB,
        scale=SCALE_GNB,
        random_state=RANDOM_SEED_GNB
    )
    df_gaussian.to_csv(OUTPUT_FILENAME_GNB, index=False)
    print(f"\nGaussian Naive Bayes dataset saved to {OUTPUT_FILENAME_GNB}")

    if PLOT_DATA_GNB:
        plt.figure(figsize=(8, 7))
        sns.scatterplot(data=df_gaussian, x='SensorX', y='SensorY', hue='DeviceType', palette='viridis', style='DeviceType', s=50)
        plt.title('Sensor Readings (XOR-like Distribution)')
        plt.xlabel('Sensor X Reading')
        plt.ylabel('Sensor Y Reading')
        plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
        plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
        plt.legend(title='Device Type')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        sns.histplot(data=df_gaussian, x='SensorX', hue='DeviceType', kde=True, ax=axes[0], palette='viridis')
        axes[0].set_title('Distribution of SensorX by DeviceType')
        sns.histplot(data=df_gaussian, x='SensorY', hue='DeviceType', kde=True, ax=axes[1], palette='viridis')
        axes[1].set_title('Distribution of SensorY by DeviceType')
        plt.tight_layout()
        plt.show()