import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_yield_data(n_samples=10000, rainfall_mean=600, rainfall_std=150,
                        max_fertilizer=300, saturation_point=150,
                        base_yield=2, rain_effect=0.01,
                        fertilizer_effect_initial=0.05,
                        noise_level=0.5, random_state=42):
    """
    Generates a dataset for crop yield prediction with a non-linear
    saturation effect for fertilizer.

    Args:
        n_samples (int): Total number of samples.
        rainfall_mean (float): Mean rainfall.
        rainfall_std (float): Standard deviation of rainfall.
        max_fertilizer (float): Maximum fertilizer amount.
        saturation_point (float): Fertilizer amount where effect starts to diminish significantly.
        base_yield (float): Base yield with zero inputs (intercept hint).
        rain_effect (float): Linear effect of rainfall on yield.
        fertilizer_effect_initial (float): Initial linear effect of fertilizer.
        noise_level (float): Standard deviation of Gaussian noise added to yield.
        random_state (int): Seed for reproducibility.

    Returns:
        pandas.DataFrame: Generated dataset with columns
                          ['Rainfall_mm', 'Fertilizer_kg_ha', 'Yield_tons_ha'].
    """
    np.random.seed(random_state)

    # 1. Generate Rainfall (Normally distributed around mean)
    rainfall = np.random.normal(rainfall_mean, rainfall_std, n_samples)
    rainfall = np.clip(rainfall, 0, None) # Rainfall cannot be negative

    # 2. Generate Fertilizer (Uniformly distributed up to max)
    fertilizer = np.random.uniform(0, max_fertilizer, n_samples)

    # 3. Calculate Yield Components
    # Base yield
    yield_base = base_yield

    # Linear effect from rainfall
    yield_from_rain = rain_effect * rainfall

    # Non-linear effect from fertilizer (using a saturating function like tanh or logistic,
    # or simply capping the linear effect / using quadratic term conceptually)
    # Let's use a quadratic term conceptually: Yield = a*F - b*F^2
    # Or more simply, make the *increase* diminish after the saturation point.
    # We can model this with a modified linear term that slows down.
    # Example: effect = initial_slope * (saturation_point * tanh(F / saturation_point))
    # This ensures initial slope is fertilizer_effect_initial but it saturates.

    # Simpler approach for demonstration: Piecewise-like effect for generation
    yield_from_fertilizer = np.zeros(n_samples)
    # Initial linear increase
    mask_low_fert = fertilizer <= saturation_point
    yield_from_fertilizer[mask_low_fert] = fertilizer_effect_initial * fertilizer[mask_low_fert]

    # Diminished increase after saturation point (slope drastically reduced)
    mask_high_fert = fertilizer > saturation_point
    # Calculate the yield reached at the saturation point
    yield_at_saturation = fertilizer_effect_initial * saturation_point
    # Calculate the reduced slope effect after saturation (e.g., 1/10th of initial)
    fertilizer_effect_saturated = fertilizer_effect_initial / 10
    yield_from_fertilizer[mask_high_fert] = yield_at_saturation + \
                                           fertilizer_effect_saturated * (fertilizer[mask_high_fert] - saturation_point)


    # 4. Calculate Total Yield + Noise
    yield_total = yield_base + yield_from_rain + yield_from_fertilizer + \
                  np.random.normal(0, noise_level, n_samples)
    yield_total = np.clip(yield_total, 0, None) # Yield cannot be negative

    # 5. Create DataFrame
    df = pd.DataFrame({
        'Rainfall_mm': rainfall,
        'Fertilizer_kg_ha': fertilizer,
        'Yield_tons_ha': yield_total
    })

    print("Dataset Generation Complete.")
    print(f"Total samples: {len(df)}")
    print("\nDataset Head:")
    print(df.head())
    print("\nBasic Statistics:")
    print(df.describe())

    return df

# --- Parameters ---
N_SAMPLES = 10000
SATURATION_FERTILIZER = 150 # kg/ha where effect diminishes
MAX_FERTILIZER = 300       # Max fertilizer applied
RAIN_MEAN = 700            # mm
RAIN_STD = 100             # mm
BASE_YIELD = 1.5           # tons/ha
RAIN_EFFECT = 0.005        # tons per mm of rain
FERT_EFFECT_INIT = 0.02    # Initial tons per kg of fertilizer
NOISE = 0.75               # tons/ha noise std dev
RANDOM_SEED = 42
OUTPUT_FILENAME = "crop_yield_saturation_data.csv"
PLOT_DATA = False # Set to True to visualize the generated data

# --- Generate Data ---
df_generated = generate_yield_data(
    n_samples=N_SAMPLES,
    rainfall_mean=RAIN_MEAN,
    rainfall_std=RAIN_STD,
    max_fertilizer=MAX_FERTILIZER,
    saturation_point=SATURATION_FERTILIZER,
    base_yield=BASE_YIELD,
    rain_effect=RAIN_EFFECT,
    fertilizer_effect_initial=FERT_EFFECT_INIT,
    noise_level=NOISE,
    random_state=RANDOM_SEED
)

# --- Save Data ---
df_generated.to_csv(OUTPUT_FILENAME, index=False)
print(f"\nDataset saved to {OUTPUT_FILENAME}")

# --- Optional Visualization ---
if PLOT_DATA:
    plt.figure(figsize=(14, 6))

    # Plot 1: Yield vs Fertilizer (Shows the curveball)
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_generated, x='Fertilizer_kg_ha', y='Yield_tons_ha', alpha=0.2)
    plt.axvline(SATURATION_FERTILIZER, color='r', linestyle='--', label=f'Saturation Point ({SATURATION_FERTILIZER} kg/ha)')
    plt.title('Yield vs. Fertilizer (Saturation Effect)')
    plt.xlabel('Fertilizer Applied (kg/ha)')
    plt.ylabel('Crop Yield (tons/ha)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Yield vs Rainfall (Shows more linear trend)
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df_generated, x='Rainfall_mm', y='Yield_tons_ha', alpha=0.2)
    sns.regplot(data=df_generated, x='Rainfall_mm', y='Yield_tons_ha', scatter=False, color='g', line_kws={'label':'Approx. Linear Trend'})
    plt.title('Yield vs. Rainfall')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Crop Yield (tons/ha)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
