# Datasets

The five synthetic datasets and their generator scripts were provided by the course staff with the weekly exercises. They are committed so the examples and tests run without extra steps. Each generator uses a fixed seed and reproduces its CSV exactly (`user_actions.csv` matches in value; only the float text formatting differs).

| File | Rows | Task | Target | Generator | Used by |
|---|---:|---|---|---|---|
| `crop_yield_saturation_data.csv` | 10,000 | Regression: yield from rainfall and fertilizer, with fertilizer saturating past a threshold | `Yield_tons_ha` | `generate_crop_yield.py` | `LinearRegressor`, `SGDRegression`, `SVR`, `MLPRegressor` |
| `user_actions.csv` | 10,000 | Binary classification, 65/35 split; the positive class sits inside a ring of negatives, so it is **not linearly separable** | `action_taken` | `generate_user_actions.py` | `LogisticRegression`, `SGDClassifier`, `SVM`, `SVC` |
| `customer_support_messages.csv` | 2,000 | Text classification from template-generated messages (60/40 split) | `category` | `generate_support_messages.py` | `MultinomialNB` |
| `sensor_device_types_xor.csv` | 1,000 | Binary classification with an XOR layout: each class occupies two opposite quadrants | `DeviceType` | `generate_sensor_xor.py` | `GaussianNB`, `SVC` |
| `customer_subscription_data.csv` | 5,000 | Binary classification with mixed numeric/categorical features and one pure-noise feature (64/36 split) | `Subscribed` | `generate_customer_subscription.py` | `DecisionTreeClassifier`, `RandomForestClassifier` |

## Regenerating

The scripts write to the current directory, so run them from inside `datasets/`. They need the `examples` extra (`pandas`, `matplotlib`, `seaborn`).

```bash
cd datasets
python generate_crop_yield.py
python generate_user_actions.py
python generate_support_messages.py
python generate_sensor_xor.py
python generate_customer_subscription.py
```

## MNIST digits (not committed)

The neural-network exercises used the Kaggle *Digit Recognizer* version of MNIST: `train.csv` (42,000 labelled rows: `label` + 784 pixel columns) and `test.csv` (28,000 unlabelled rows). The two files total about 125 MB, so they're excluded from Git.

To use them, download both files from <https://www.kaggle.com/competitions/digit-recognizer/data> into this folder. Without them, `examples/compare_with_sklearn.py` falls back to scikit-learn's bundled 8×8 digits, and the tests never need them.
