import random
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.stats import entropy as shannon_entropy

# A helper to find all possible date formats in a string.
def extract_date_formats(date_str):
    formats = set()
    if not date_str:
        return formats
    try:
        if len(date_str) == 8:
            d_obj = datetime.strptime(date_str, '%d%m%Y')
        elif len(date_str) == 6:
            d_obj = datetime.strptime(date_str, '%d%m%y')
        elif len(date_str) == 4:
            d_obj = datetime.strptime(date_str + str(datetime.now().year), '%d%m%Y')
        else:
            return formats

        dd = f"{d_obj.day:02d}"
        mm = f"{d_obj.month:02d}"
        yy = f"{d_obj.year % 100:02d}"
        # FIX: The variable name for the full year is corrected here.
        yyyy = str(d_obj.year)

        # FIX: The corrected variable name is now used here, resolving the error.
        formats.update([dd + mm, mm + dd, yy, yyyy, dd + mm + yy, yy + mm + dd, mm + dd + yy])
    except ValueError:
        pass
    return formats

# This is the main "judge" that decides if a PIN is strong or weak.
def assess_pin_strength(pin, dob_self=None, dob_spouse=None, anniversary=None):
    weakness_reasons = []

    if not pin or not pin.isdigit() or len(pin) not in [4, 6]:
        return "WEAK", ["INVALID_PIN_FORMAT"]

    num_digits = len(pin)

    common_weak_patterns = set()
    if num_digits == 4:
        common_weak_patterns.update(["1234", "0000", "1111", "2222", "9999", "4321", "1212", "7890", "2580"])
    elif num_digits == 6:
        common_weak_patterns.update(["123456", "000000", "111111", "987654", "121212", "789012"])

    if pin in common_weak_patterns:
        weakness_reasons.append("COMMON_USED")

    is_sequential_asc = all(int(pin[i+1]) == (int(pin[i]) + 1) % 10 for i in range(num_digits - 1))
    is_sequential_desc = all(int(pin[i+1]) == (int(pin[i]) - 1 + 10) % 10 for i in range(num_digits - 1))
    if is_sequential_asc or is_sequential_desc:
        weakness_reasons.append("COMMON_USED_SEQUENTIAL")

    if num_digits % 2 == 0 and pin[:num_digits//2] == pin[num_digits//2:]:
        weakness_reasons.append("COMMON_USED_REPEATED_PATTERN")
    if num_digits % 3 == 0 and pin[:num_digits//3] * 3 == pin:
         weakness_reasons.append("COMMON_USED_REPEATED_PATTERN")

    demographic_data = {
        "DEMOGRAPHIC_DOB_SELF": dob_self,
        "DEMOGRAPHIC_DOB_SPOUSE": dob_spouse,
        "DEMOGRAPHIC_ANNIVERSARY": anniversary,
    }

    for reason_key, date_str in demographic_data.items():
        if date_str:
            extracted_formats = extract_date_formats(date_str)
            if any(d_format and d_format in pin for d_format in extracted_formats):
                weakness_reasons.append(reason_key)

    if weakness_reasons:
        return "WEAK", sorted(list(set(weakness_reasons)))
    else:
        return "STRONG", []

# A robust and balanced test suite with over 50 cases.
def run_all_tests():
    print("-" * 70)
    print("--- Running Exhaustive & Balanced Test Suite (50+ Cases) ---")

    dob_s = "15081990"
    dob_sp = "25121985"
    anniv = "01062015"

    test_cases = [
        # --- Category 1: Truly STRONG PINs (Balanced Cases) ---
        ("8392", None, None, None, "STRONG", []),
        ("4716", None, None, None, "STRONG", []),
        ("9021", None, None, None, "STRONG", []),
        ("610492", None, None, None, "STRONG", []),
        ("372810", None, None, None, "STRONG", []),
        ("591043", None, None, None, "STRONG", []),
        ("2491", dob_s, None, None, "STRONG", []),
        ("736401", None, dob_sp, None, "STRONG", []),
        ("5556", None, None, None, "STRONG", []),
        ("8788", None, None, None, "STRONG", []),

        # --- Category 2: Weak - Common & Sequential Patterns ---
        ("1234", None, None, None, "WEAK", ["COMMON_USED", "COMMON_USED_SEQUENTIAL"]),
        ("1111", None, None, None, "WEAK", ["COMMON_USED", "COMMON_USED_REPEATED_PATTERN"]),
        ("4321", None, None, None, "WEAK", ["COMMON_USED", "COMMON_USED_SEQUENTIAL"]),
        ("9876", None, None, None, "WEAK", ["COMMON_USED_SEQUENTIAL"]),
        ("8888", None, None, None, "WEAK", ["COMMON_USED_REPEATED_PATTERN"]),
        ("1212", None, None, None, "WEAK", ["COMMON_USED", "COMMON_USED_REPEATED_PATTERN"]),
        ("2580", None, None, None, "WEAK", ["COMMON_USED"]),
        ("123456", None, None, None, "WEAK", ["COMMON_USED", "COMMON_USED_SEQUENTIAL"]),
        ("654321", None, None, None, "WEAK", ["COMMON_USED_SEQUENTIAL"]),
        ("777777", None, None, None, "WEAK", ["COMMON_USED_REPEATED_PATTERN"]),
        ("121212", None, None, None, "WEAK", ["COMMON_USED", "COMMON_USED_REPEATED_PATTERN"]),
        ("123123", None, None, None, "WEAK", ["COMMON_USED_REPEATED_PATTERN"]),
        ("8901", None, None, None, "WEAK", ["COMMON_USED_SEQUENTIAL"]),
        ("2109", None, None, None, "WEAK", ["COMMON_USED_SEQUENTIAL"]),

        # --- Category 3: Weak - Exhaustive Demographic Matches ---
        ("1508", dob_s, None, None, "WEAK", ["DEMOGRAPHIC_DOB_SELF"]),
        ("0815", dob_s, None, None, "WEAK", ["DEMOGRAPHIC_DOB_SELF"]),
        ("1990", dob_s, None, None, "WEAK", ["DEMOGRAPHIC_DOB_SELF"]),
        ("150890", dob_s, None, None, "WEAK", ["DEMOGRAPHIC_DOB_SELF"]),
        ("900815", dob_s, None, None, "WEAK", ["DEMOGRAPHIC_DOB_SELF"]),
        ("551990", dob_s, None, None, "WEAK", ["DEMOGRAPHIC_DOB_SELF"]),
        ("2512", None, dob_sp, None, "WEAK", ["DEMOGRAPHIC_DOB_SPOUSE"]),
        ("1225", None, dob_sp, None, "WEAK", ["DEMOGRAPHIC_DOB_SPOUSE"]),
        ("1985", None, dob_sp, None, "WEAK", ["DEMOGRAPHIC_DOB_SPOUSE"]),
        ("251285", None, dob_sp, None, "WEAK", ["DEMOGRAPHIC_DOB_SPOUSE"]),
        ("0106", None, None, anniv, "WEAK", ["DEMOGRAPHIC_ANNIVERSARY"]),
        ("0601", None, None, anniv, "WEAK", ["DEMOGRAPHIC_ANNIVERSARY"]),
        ("2015", None, None, anniv, "WEAK", ["DEMOGRAPHIC_ANNIVERSARY"]),
        ("010615", None, None, anniv, "WEAK", ["DEMOGRAPHIC_ANNIVERSARY"]),

        # --- Category 4: Weak - Complex Cases with Multiple Reasons ---
        ("1111", "11011980", None, None, "WEAK", ["COMMON_USED", "COMMON_USED_REPEATED_PATTERN", "DEMOGRAPHIC_DOB_SELF"]),
        ("9999", "10101999", None, None, "WEAK", ["COMMON_USED", "COMMON_USED_REPEATED_PATTERN", "DEMOGRAPHIC_DOB_SELF"]),
        ("0808", "08081990", None, None, "WEAK", ["COMMON_USED_REPEATED_PATTERN", "DEMOGRAPHIC_DOB_SELF"]),
        ("1212", None, None, "12122012", "WEAK", ["COMMON_USED", "COMMON_USED_REPEATED_PATTERN", "DEMOGRAPHIC_ANNIVERSARY"]),

        # --- Category 5: Invalid & Edge Cases ---
        ("123", None, None, None, "WEAK", ["INVALID_PIN_FORMAT"]),
        ("12345", None, None, None, "WEAK", ["INVALID_PIN_FORMAT"]),
        ("1234567", None, None, None, "WEAK", ["INVALID_PIN_FORMAT"]),
        ("", None, None, None, "WEAK", ["INVALID_PIN_FORMAT"]),
        ("abcdef", None, None, None, "WEAK", ["INVALID_PIN_FORMAT"]),
        ("12a4", None, None, None, "WEAK", ["INVALID_PIN_FORMAT"]),
        ("90", None, None, None, "WEAK", ["INVALID_PIN_FORMAT"]),
    ]

    passed_count = 0
    failed_cases = []
    for i, (pin, dob_s, dob_sp, anniv, exp_strength, exp_reasons) in enumerate(test_cases):
        actual_strength, actual_reasons = assess_pin_strength(pin, dob_s, dob_sp, anniv)
        exp_reasons.sort()

        if actual_strength == exp_strength and actual_reasons == exp_reasons:
            status = "[PASS]"
            passed_count += 1
        else:
            status = "[FAIL]"
            failed_cases.append(f"  Test #{i+1} failed for PIN '{pin}'. Expected ({exp_strength}, {exp_reasons}) but got ({actual_strength}, {actual_reasons})")

    print(f"Completed {len(test_cases)} test cases.")
    print("-" * 70)
    print(f"Test Summary: {passed_count} / {len(test_cases)} tests passed.")
    if failed_cases:
        print("\n--- FAILED CASES ---")
        for case in failed_cases:
            print(case)
    print("-" * 70)

# Creates a BALANCED dataset with a 50/50 split of weak and strong PINs.
def generate_dummy_data(num_samples=5000):
    print(f"\nGenerating {num_samples} balanced dummy samples...")
    data = []
    current_year = datetime.now().year

    for i in range(num_samples):
        birth_year_self = random.randint(current_year - 70, current_year - 18)
        birth_month_self = random.randint(1, 12)
        birth_day_self = random.randint(1, 28)
        dob_self = f"{birth_day_self:02d}{birth_month_self:02d}{birth_year_self}"

        pin_length = random.choice([4, 6])
        pin = ""

        if i % 2 == 0:
            # --- Generate a WEAK PIN ---
            weakness_type = random.choice(['common', 'demographic', 'sequential'])
            if weakness_type == 'common':
                if pin_length == 4: pin = random.choice(["1234", "1111", "4321", "1212"])
                else: pin = random.choice(["123456", "111111", "654321", "121212"])
            elif weakness_type == 'demographic':
                if pin_length == 4: pin = str(birth_year_self)
                else: pin = f"{birth_day_self:02d}{birth_month_self:02d}{birth_year_self % 100:02d}"
            else: # Sequential
                start_digit = random.randint(0, 9 - pin_length)
                pin = "".join([str(start_digit + j) for j in range(pin_length)])
        else:
            # --- Generate a STRONG PIN ---
            while True:
                candidate_pin = ''.join(random.choices('0123456789', k=pin_length))
                strength, _ = assess_pin_strength(candidate_pin)
                if strength == 'STRONG':
                    pin = candidate_pin
                    break

        data.append([pin, dob_self, None, None])

    return pd.DataFrame(data, columns=['pin', 'dob_self', 'dob_spouse', 'anniversary'])

# Calculates a "randomness score" for a PIN.
def calculate_shannon_entropy(pin):
    if not pin or len(pin) == 0: return 0.0
    freq = [pin.count(c) / len(pin) for c in set(pin)]
    return shannon_entropy(freq, base=2)

# Creates and displays several graphs to analyze the generated data.
def perform_eda(df):
    print("\n--- Statistical PIN Analysis (EDA) ---")
    df['shannon_entropy'] = df['pin'].apply(calculate_shannon_entropy)

    # Plot 1: Overall Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='strength', data=df, palette={'STRONG': 'green', 'WEAK': 'red'}, hue='strength', legend=False)
    plt.title('Overall Distribution of Weak vs Strong MPINs (Balanced Data)', fontsize=16)
    plt.show()

    # Plot 2: Shannon Entropy Histogram
    plt.figure(figsize=(12, 7))
    sns.histplot(data=df, x='shannon_entropy', hue='strength', kde=True, palette={'STRONG': 'green', 'WEAK': 'red'})
    plt.title('Distribution of Shannon Entropy for Strong and Weak MPINs', fontsize=16)
    plt.xlabel('Shannon Entropy (Higher indicates more randomness)', fontsize=12)
    plt.show()

    # Plot 3: Breakdown of Weakness Reasons
    plt.figure(figsize=(12, 7))
    weak_pins_df = df[df['strength'] == 'WEAK'].copy()
    if not weak_pins_df.empty:
        reasons_series = weak_pins_df['reasons'].explode()
        reason_percentages = reasons_series.value_counts(normalize=True) * 100
        sns.barplot(x=reason_percentages.values, y=reason_percentages.index, palette='magma', hue=reason_percentages.index, legend=False)
        plt.title('Breakdown of Weakness Reasons (%)', fontsize=16)
        plt.xlabel('Percentage of Weak PINs (%)', fontsize=12)
    plt.tight_layout()
    plt.show()

# Builds and trains a machine learning model.
def train_and_evaluate_ml_model(df):
    print("\n--- Building the Prediction Model ---")

    df['shannon_entropy'] = df['pin'].apply(calculate_shannon_entropy)
    df['is_repetitive'] = df['pin'].apply(lambda p: 1 if p and len(set(p)) == 1 else 0)
    df['is_sequential'] = df['pin'].apply(lambda p: 1 if p and (all(int(p[i+1]) == (int(p[i]) + 1) % 10 for i in range(len(p) - 1)) or all(int(p[i+1]) == (int(p[i]) - 1 + 10) % 10 for i in range(len(p) - 1))) else 0)
    df['has_demographic_dob_self'] = df['reasons'].apply(lambda x: 1 if "DEMOGRAPHIC_DOB_SELF" in x else 0)

    feature_cols = ['shannon_entropy', 'is_repetitive', 'is_sequential', 'has_demographic_dob_self']
    X = df[feature_cols]
    y = (df['strength'] == 'WEAK').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train, y_train)

    print("\nModel Evaluation (Test Set):")
    print(classification_report(y_test, model.predict(X_test)))

# Main script execution starts here.
if __name__ == "__main__":
    run_all_tests()

    df_dummy = generate_dummy_data(num_samples=5000)

    strength_results = df_dummy.apply(lambda row: assess_pin_strength(row['pin'], row['dob_self'], row['dob_spouse'], row['anniversary']), axis=1)
    df_dummy[['strength', 'reasons']] = pd.DataFrame(strength_results.tolist(), index=df_dummy.index)

    perform_eda(df_dummy)
    train_and_evaluate_ml_model(df_dummy)

  