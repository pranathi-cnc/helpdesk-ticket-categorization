import csv
import json
import os
from pathlib import Path

import ollama
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")

DATA_FILE = Path("data/tickets.csv")
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

TRAIN_SIZE = 20
VALIDATION_SIZE = 5
TEST_SIZE = 5

ALLOWED_LABELS = {"Billing", "Login", "Delivery"}

client = ollama.Client(host=OLLAMA_URL)


def load_tickets():
    df = pd.read_csv(DATA_FILE)

    required_columns = {
        "ticket_id",
        "ticket",
        "expected_label",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError("tickets.csv has missing required columns.")

    if len(df) != 30:
        raise ValueError("Expected exactly 30 tickets.")

    if not set(df["expected_label"]).issubset(ALLOWED_LABELS):
        raise ValueError("Invalid label found in tickets.csv.")

    return df
def split_tickets(df):
    # Keep all three categories represented in validation and test.
    billing = df[df["expected_label"] == "Billing"].sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    login = df[df["expected_label"] == "Login"].sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    delivery = df[df["expected_label"] == "Delivery"].sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    # Validation: 5 tickets
    validation = pd.concat([
        billing.iloc[:1],
        login.iloc[:2],
        delivery.iloc[:2],
    ])

    # Test: 5 tickets
    test = pd.concat([
        billing.iloc[1:2],
        login.iloc[2:4],
        delivery.iloc[2:4],
    ])

    # Remaining 20 tickets for training
    used_ids = set(validation["ticket_id"]) | set(test["ticket_id"])

    train = df[~df["ticket_id"].isin(used_ids)].copy()

    # Shuffle each final dataset
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    validation = validation.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)

    return train, validation, test



def save_jsonl(df, filename):
    output_file = DATA_DIR / filename

    with open(output_file, "w", encoding="utf-8") as file:
        for _, row in df.iterrows():
            record = {
                "input": row["ticket"],
                "output": row["expected_label"],
            }

            file.write(json.dumps(record) + "\n")

    return output_file


def verify_split(train, validation, test):
    train_ids = set(train["ticket_id"])
    validation_ids = set(validation["ticket_id"])
    test_ids = set(test["ticket_id"])

    if train_ids & validation_ids:
        raise ValueError("Training and validation data overlap.")

    if train_ids & test_ids:
        raise ValueError("Training and test data overlap.")

    if validation_ids & test_ids:
        raise ValueError("Validation and test data overlap.")

    print("\nDataset split verified: no overlap.")


def create_prompt(ticket):
    return f"""
You are a helpdesk ticket classification system.

Classify the following support ticket into exactly ONE of these
three categories:

Billing
Login
Delivery

Rules:
- Return ONLY one category name.
- Do not provide an explanation.
- Do not return any other text.
- Do not create new categories.

Ticket:
{ticket}
"""


def predict_ticket(ticket):
    prompt = create_prompt(ticket)

    response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
    )

    raw_response = response["response"].strip()

    # Exact match
    if raw_response in ALLOWED_LABELS:
        return raw_response, ""

    # Try to find a valid label inside the response.
    # This is only used for recording a prediction;
    # unexpected responses are still marked as invalid.
    for label in ALLOWED_LABELS:
        if raw_response.lower() == label.lower():
            return label, ""

    return None, raw_response


def run_baseline(test):
    results = []

    print("\nBASELINE PREDICTIONS")
    print("=" * 70)

    for _, row in test.iterrows():
        prediction, invalid_output = predict_ticket(row["ticket"])

        actual = row["expected_label"]

        if prediction is None:
            status = "INVALID"
        elif prediction == actual:
            status = "CORRECT"
        else:
            status = "INCORRECT"

        print(f"\nTicket: {row['ticket_id']}")
        print(f"Text: {row['ticket']}")
        print(f"Actual: {actual}")

        if prediction:
            print(f"Gemma: {prediction}")
        else:
            print("Gemma: INVALID")

        print(f"Status: {status}")

        results.append({
            "ticket_id": row["ticket_id"],
            "ticket": row["ticket"],
            "actual_label": actual,
            "baseline_prediction": prediction or "",
            "invalid_output": invalid_output,
            "status": status,
        })

    return results


def save_baseline_results(results):
    OUTPUT_DIR.mkdir(exist_ok=True)

    output_file = OUTPUT_DIR / "baseline_predictions.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "ticket_id",
            "ticket",
            "actual_label",
            "baseline_prediction",
            "invalid_output",
            "status",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    return output_file


def calculate_accuracy(results):
    valid_results = [
        result
        for result in results
        if result["baseline_prediction"]
    ]

    if not valid_results:
        return 0.0

    correct = sum(
        result["baseline_prediction"] == result["actual_label"]
        for result in valid_results
    )

    return correct / len(valid_results)


def main():
    print("Loading tickets...")

    df = load_tickets()

    print(f"Total tickets: {len(df)}")

    train, validation, test = split_tickets(df)

    print(f"Training tickets:   {len(train)}")
    print(f"Validation tickets: {len(validation)}")
    print(f"Test tickets:       {len(test)}")

    verify_split(train, validation, test)

    # Save datasets
    save_jsonl(train, "train.jsonl")
    save_jsonl(validation, "validation.jsonl")

    test_file = DATA_DIR / "test.csv"
    test.to_csv(test_file, index=False)

    print("\nDataset files created.")

    # Run Gemma baseline
    results = run_baseline(test)

    output_file = save_baseline_results(results)

    accuracy = calculate_accuracy(results)

    invalid_count = sum(
        1
        for result in results
        if not result["baseline_prediction"]
    )

    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)

    print(f"Model: {OLLAMA_MODEL}")
    print(f"Test tickets: {len(test)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Invalid predictions: {invalid_count}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()