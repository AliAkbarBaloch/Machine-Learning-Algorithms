import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_customer_subscription_data(n_samples=5000, noise_factor=0.1, random_state=42):
    """
    Generates a dataset for predicting customer subscription likelihood (binary classification).
    The dataset includes a mix of numerical and categorical features with some
    underlying rules and noise, suitable for Decision Tree and Random Forest models.

    Args:
        n_samples (int): Total number of samples.
        noise_factor (float): Proportion of labels to flip for borderline cases, adds noise.
        random_state (int): Seed for reproducibility.

    Returns:
        pandas.DataFrame: Generated dataset.
    """
    np.random.seed(random_state)

    # --- Feature Generation ---

    # 1. Age (Numerical): 18 - 70 years
    age = np.random.randint(18, 71, n_samples)

    # 2. Income Level (Categorical): Low, Medium, High
    income_level_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    income_raw = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]) # More low/medium income
    income_level = np.array([income_level_map[i] for i in income_raw])

    # 3. Time on Platform (Numerical): 1 - 500 hours
    time_on_platform = np.random.randint(1, 501, n_samples)

    # 4. Pages Viewed (Numerical): 5 - 1000 pages
    pages_viewed = np.random.randint(5, 1001, n_samples)

    # 5. Subscription Plan Viewed (Categorical): None, Basic, Premium, Enterprise
    # 'None' means they haven't explicitly looked at plans yet.
    plan_viewed_map = {0: 'None', 1: 'Basic', 2: 'Premium', 3: 'Enterprise'}
    plan_raw = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.2, 0.2, 0.1])
    plan_viewed = np.array([plan_viewed_map[p] for p in plan_raw])

    # 6. Support Queries (Numerical): 0 - 10 queries
    # Introduce non-monotonicity: 0-1 queries might be normal/good, high queries might be bad.
    support_queries = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        n_samples,
        p=[0.3, 0.2, 0.15, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03, 0.025, 0.025]
    )

    # 7. Marketing Campaign Interaction (Categorical): NoInteraction, Clicked, ViewedOffer
    # This might be a less predictive or somewhat noisy feature.
    campaign_map = {0: 'NoInteraction', 1: 'Clicked', 2: 'ViewedOffer'}
    campaign_raw = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15])
    campaign_interaction = np.array([campaign_map[c] for c in campaign_raw])
    
    # 8. Random Noise Feature (Numerical) - To test feature selection/importance
    random_feature = np.random.randn(n_samples) * 10


    # --- Target Variable Generation (Subscribed: 0 or 1) ---
    # Define some rules for subscription, then add noise.
    # Base likelihood, then apply boosts or penalties.
    
    likelihood_score = np.zeros(n_samples)

    # Rule 1: Time on platform and pages viewed are strong positive indicators
    likelihood_score += (time_on_platform / 100)  # Max +5
    likelihood_score += (pages_viewed / 200)      # Max +5

    # Rule 2: Age (sweet spot, e.g., 28-50)
    likelihood_score[ (age >= 28) & (age <= 50) ] += 1.5
    likelihood_score[ age < 28 ] -= 0.5
    likelihood_score[ age > 50 ] -= 0.25


    # Rule 3: Income Level
    likelihood_score[ income_level == 'High' ] += 2.0
    likelihood_score[ income_level == 'Medium' ] += 1.0
    # 'Low' income no direct boost, relies on other factors

    # Rule 4: Plan Viewed
    likelihood_score[ plan_viewed == 'Premium' ] += 1.5
    likelihood_score[ plan_viewed == 'Enterprise' ] += 2.0
    likelihood_score[ plan_viewed == 'Basic' ] += 0.5
    # 'None' viewed no direct boost

    # Rule 5: Support Queries (non-monotonic)
    likelihood_score[ support_queries == 0 ] += 0.5 # No issues, good sign
    likelihood_score[ support_queries == 1 ] += 0.2 # Minor engagement
    likelihood_score[ (support_queries >= 2) & (support_queries <= 4) ] -= 1.0 # Some trouble
    likelihood_score[ support_queries > 4 ] -= 2.5 # Significant trouble, unlikely to subscribe

    # Rule 6: Campaign Interaction (minor positive effect)
    likelihood_score[ campaign_interaction == 'ViewedOffer' ] += 0.75
    likelihood_score[ campaign_interaction == 'Clicked' ] += 0.25

    # Convert likelihood score to probability (sigmoid-like scaling)
    # We can scale score to be roughly in [-5, 5] for sigmoid input
    # Min possible score roughly: 0 (time/pages min) - 0.5 (age) + 0 (income) + 0 (plan) - 2.5 (queries) = -3
    # Max possible score roughly: 5 (time) + 5 (pages) + 1.5 (age) + 2 (income) + 2 (plan) + 0.5 (queries) + 0.75 (campaign) = 16.75
    # Let's normalize the score to a reasonable range for a probability conversion
    # A simple thresholding based on a scaled score or quantile can also work.
    
    # For simplicity, let's use a threshold on the likelihood_score after some scaling.
    # The goal is to get a reasonable number of 0s and 1s.
    # Let's find a threshold that gives roughly 30-40% subscriptions.
    
    # Normalize scores (optional, but can help in setting a threshold)
    # score_normalized = (likelihood_score - np.mean(likelihood_score)) / np.std(likelihood_score)
    
    # Determine subscription based on a threshold (e.g., if score > median or a fixed value)
    # Let's try a threshold that's somewhat dynamic or picked to give a decent class balance.
    # For this example, let's pick a threshold.
    # Scores could range from approx -3 to +17. Let's say a score of 6+ means subscribe.
    threshold = np.percentile(likelihood_score, 65) # Aim for ~35% subscription rate initially
    subscribed = (likelihood_score > threshold).astype(int)

    # Introduce noise: Flip some labels, especially for samples near the threshold
    # Identify "borderline" cases (e.g., scores within +/- X% of the threshold)
    borderline_range = 0.20 * threshold 
    borderline_mask = (likelihood_score > threshold - borderline_range) & \
                      (likelihood_score < threshold + borderline_range)
    
    num_borderline = np.sum(borderline_mask)
    num_to_flip = int(noise_factor * num_borderline)
    
    if num_to_flip > 0:
        flip_indices = np.random.choice(np.where(borderline_mask)[0], size=num_to_flip, replace=False)
        subscribed[flip_indices] = 1 - subscribed[flip_indices] # Flip 0 to 1 and 1 to 0

    # --- Create DataFrame ---
    df = pd.DataFrame({
        'Age': age,
        'Income_Level': income_level,
        'Time_On_Platform_Hours': time_on_platform,
        'Pages_Viewed_Count': pages_viewed,
        'Plan_Viewed': plan_viewed,
        'Support_Queries': support_queries,
        'Campaign_Interaction': campaign_interaction,
        'Random_Noise_Feature': random_feature,
        'Subscribed': subscribed
    })

    print("Subscription Dataset Generation Complete.")
    print(f"Total samples: {len(df)}")
    print(f"Subscription Rate: {df['Subscribed'].mean():.2%}")
    print("\nDataset Head:")
    print(df.head())
    # print("\nBasic Statistics:")
    # print(df.describe(include='all')) # Include categorical for counts/unique

    return df

# --- Example Usage of the Generator ---
if __name__ == '__main__':
    N_SAMPLES_SUBSCRIPTION = 5000
    NOISE_SUBSCRIPTION = 0.15 # 15% of borderline cases might be flipped
    RANDOM_SEED_SUBSCRIPTION = 42
    OUTPUT_FILENAME_SUBSCRIPTION = "customer_subscription_data.csv"
    PLOT_SUBSCRIPTION_DATA = False

    df_subscription = generate_customer_subscription_data(
        n_samples=N_SAMPLES_SUBSCRIPTION,
        noise_factor=NOISE_SUBSCRIPTION,
        random_state=RANDOM_SEED_SUBSCRIPTION
    )

    df_subscription.to_csv(OUTPUT_FILENAME_SUBSCRIPTION, index=False)
    print(f"\nSubscription dataset saved to {OUTPUT_FILENAME_SUBSCRIPTION}")

    if PLOT_SUBSCRIPTION_DATA:
        print("\nVisualizing some feature distributions and relationships with target:")
        
        plt.figure(figsize=(18, 12))
        
        plt.subplot(2, 3, 1)
        sns.histplot(data=df_subscription, x='Time_On_Platform_Hours', hue='Subscribed', kde=True, multiple="stack")
        plt.title('Time on Platform vs. Subscription')

        plt.subplot(2, 3, 2)
        sns.histplot(data=df_subscription, x='Pages_Viewed_Count', hue='Subscribed', kde=True, multiple="stack")
        plt.title('Pages Viewed vs. Subscription')

        plt.subplot(2, 3, 3)
        sns.countplot(data=df_subscription, x='Income_Level', hue='Subscribed', order=['Low', 'Medium', 'High'])
        plt.title('Income Level vs. Subscription')
        
        plt.subplot(2, 3, 4)
        sns.countplot(data=df_subscription, x='Plan_Viewed', hue='Subscribed', order=['None', 'Basic', 'Premium', 'Enterprise'])
        plt.title('Plan Viewed vs. Subscription')

        plt.subplot(2, 3, 5)
        sns.boxplot(data=df_subscription, x='Subscribed', y='Age')
        plt.title('Age Distribution by Subscription')

        plt.subplot(2, 3, 6)
        sns.countplot(data=df_subscription, x='Support_Queries', hue='Subscribed')
        plt.title('Support Queries vs. Subscription')
        
        plt.tight_layout()
        plt.show()

        # Scatter plot for two strong continuous features
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_subscription, x='Time_On_Platform_Hours', y='Pages_Viewed_Count', hue='Subscribed', alpha=0.5)
        plt.title('Time on Platform vs. Pages Viewed (Colored by Subscription)')
        plt.xlabel('Time on Platform (Hours)')
        plt.ylabel('Pages Viewed Count')
        plt.grid(True)
        plt.show()