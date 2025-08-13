import os
import csv
import re
from collections import defaultdict

def create_freshness_csv(root_folder, output_csv):
    data_rows = []

    day_pattern = re.compile(r'Day (\d+)', re.IGNORECASE)

    deterioration_rates = {
        'Apple': 1/45,
        'Banana': 1/7,
        'Plum': 1/6,
        'Tomato': 1/6,
        'Potato': 1/45,
        'Cauliflower': 1/8.5,
        'Drumsticks': 1/6,
        'Sapodilla': 1/6,
        'Carrot': 1/45,
    }

    for fruit_name in os.listdir(root_folder):
        fruit_path = os.path.join(root_folder, fruit_name)
        if not os.path.isdir(fruit_path):
            continue

        rate = deterioration_rates.get(fruit_name, 0.1)

        for day_folder_name in os.listdir(fruit_path):
            day_match = day_pattern.match(day_folder_name)
            if not day_match:
                continue

            day_num = int(day_match.group(1))
            day_folder = os.path.join(fruit_path, day_folder_name)
            if not os.path.isdir(day_folder):
                continue

            freshness_score = max(0, 1.0 - rate * (day_num - 1))

            for file_name in os.listdir(day_folder):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    image_path = os.path.join(day_folder, file_name)
                    data_rows.append({
                        'fruit': fruit_name,
                        'day': day_num,
                        'image_path': image_path,
                        'freshness': round(freshness_score, 3)
                    })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['fruit', 'day', 'image_path', 'freshness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_rows:
            writer.writerow(row)

    print(f"CSV file '{output_csv}' created with {len(data_rows)} records.")

def summarize_dataset(csv_file):
    fruit_counts = defaultdict(int)
    day_counts = defaultdict(set)
    freshness_values = []
    fruit_freshness_sum = defaultdict(float)

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows_found = False
        for row in reader:
            rows_found = True
            fruit = row['fruit']
            day = int(row['day'])
            freshness = float(row['freshness'])

            fruit_counts[fruit] += 1
            day_counts[fruit].add(day)
            freshness_values.append(freshness)
            fruit_freshness_sum[fruit] += freshness

    if not rows_found:
        print("No data found in CSV.")
        return

    print("Summary of dataset:")
    for fruit, count in fruit_counts.items():
        days = sorted(day_counts[fruit])
        avg_freshness = fruit_freshness_sum[fruit] / count
        print(f"- {fruit}: {count} images, days: {days}, avg freshness: {avg_freshness:.3f}")

    print(f"\nTotal images: {sum(fruit_counts.values())}")
    print(f"Freshness range: {min(freshness_values):.2f} to {max(freshness_values):.2f}")

if __name__ == "__main__":
    root_data_folder = r"D:\Projects\FreshGuard\data"
    output_file = r"D:\Projects\FreshGuard\analysis\freshness_dataset.csv"
    create_freshness_csv(root_data_folder, output_file)
    summarize_dataset(output_file)
