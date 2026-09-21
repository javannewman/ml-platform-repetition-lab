import csv
import sys

DATA_FILE = "data/equipment_data.csv"

EXPECTED_COLUMNS = [
    "temperature",
    "vibration",
    "machine_age",
    "error_count",
    "failure",
]


def validate_data():

    errors = []
    row_count = 0

    with open(DATA_FILE, "r") as file:

        reader = csv.DictReader(file)

        # Check that the expected columns exist
        if reader.fieldnames != EXPECTED_COLUMNS:
            errors.append(
                f"Invalid columns. Expected {EXPECTED_COLUMNS}, "
                f"but found {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):

            row_count += 1

            try:
                temperature = float(row["temperature"])
                vibration = float(row["vibration"])
                machine_age = int(row["machine_age"])
                error_count = int(row["error_count"])
                failure = int(row["failure"])

            except (ValueError, TypeError):
                errors.append(
                    f"Row {row_number}: contains invalid data types"
                )
                continue

            if not 50 <= temperature <= 100:
                errors.append(
                    f"Row {row_number}: invalid temperature {temperature}"
                )

            if not 0.5 <= vibration <= 5.0:
                errors.append(
                    f"Row {row_number}: invalid vibration {vibration}"
                )

            if not 1 <= machine_age <= 15:
                errors.append(
                    f"Row {row_number}: invalid machine_age {machine_age}"
                )

            if not 0 <= error_count <= 10:
                errors.append(
                    f"Row {row_number}: invalid error_count {error_count}"
                )

            if failure not in [0, 1]:
                errors.append(
                    f"Row {row_number}: failure must be 0 or 1"
                )

    if errors:

        print("DATA VALIDATION FAILED")

        for error in errors:
            print(error)

        sys.exit(1)

    print("DATA VALIDATION PASSED")
    print(f"Validated {row_count} rows")


validate_data()