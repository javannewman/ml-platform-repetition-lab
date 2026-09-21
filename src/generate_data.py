import csv
import random

random.seed(42)

OUTPUT_FILE = "data/equipment_data.csv"
NUMBER_OF_ROWS = 500


with open(OUTPUT_FILE, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(
        [
            "temperature",
            "vibration",
            "machine_age",
            "error_count",
            "failure",
        ]
    )

    for _ in range(NUMBER_OF_ROWS):

        temperature = round(random.uniform(50, 100), 2)
        vibration = round(random.uniform(0.5, 5.0), 2)
        machine_age = random.randint(1, 15)
        error_count = random.randint(0, 10)

        failure_probability = 0.03

        if temperature > 80:
            failure_probability += 0.25

        if vibration > 3.0:
            failure_probability += 0.30

        if machine_age > 8:
            failure_probability += 0.20

        if error_count >= 4:
            failure_probability += 0.30

        failure_probability = min(failure_probability, 0.95)

        failure = 1 if random.random() < failure_probability else 0

        writer.writerow(
            [
                temperature,
                vibration,
                machine_age,
                error_count,
                failure,
            ]
        )


print(f"Generated {NUMBER_OF_ROWS} rows in {OUTPUT_FILE}")