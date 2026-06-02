import pandas as pd
import re
import json
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import time

# ============ CONFIG ============
USE_LLM = False   # set to True if you want to use Ollama (slower)
INPUT_FILE = "synthetic_lung_cancer.csv"
OUTPUT_FILE = "patient_symptoms_output.json"

# ============ HELPERS ============

def parse_duration_to_days(duration: str) -> int:
    """Convert things like '2 weeks', '5 months', '3 years' → days"""
    if not duration:
        return 0
    duration = duration.lower().strip()
    num = int(re.search(r"\d+", duration).group()) if re.search(r"\d+", duration) else 0
    if "day" in duration:
        return num
    if "week" in duration:
        return num * 7
    if "month" in duration:
        return num * 30
    if "year" in duration:
        return num * 365
    return 0

def calculate_start_date(visit_date: str, duration: str) -> str:
    """Back-calculate start date from visit date and duration"""
    try:
        visit_dt = datetime.strptime(visit_date, "%H:%M %d:%m:%Y")
    except Exception:
        return ""
    days = parse_duration_to_days(duration)
    start_dt = visit_dt - timedelta(days=days)
    return start_dt.strftime("%Y-%m-%d")

# Rule-based extraction
def rule_based_extract(note: str):
    symptoms = []
    patterns = [
        r"([A-Za-z\s]+?)\s*\(since\s*([\d]+\s*(?:days?|weeks?|months?|years?))\)",
    ]
    for pat in patterns:
        for match in re.findall(pat, note):
            symptom = match[0].strip()
            duration = match[1].strip()
            if symptom:
                symptoms.append({"name": symptom, "duration": duration})
    return symptoms

# ============ MAIN PROCESSOR ============
def process_csv():
    df = pd.read_csv(INPUT_FILE, sep=",")
    df.columns = df.columns.str.strip()  # clean any extra spaces in headers
    print("CSV columns detected:", df.columns.tolist())

    grouped = defaultdict(lambda: {"visits": [], "consolidated_symptoms": {}})

    start_time = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing patients"):
        subject_id = str(row.get("subject_id")).strip()
        visit_date = str(row.get("visit_date")).strip()
        note = str(row.get("clinical_notes")).strip()

        # Extract symptoms (regex only for now)
        symptoms = rule_based_extract(note)

        visit_entry = {
            "visit_date": visit_date,
            "symptoms": []
        }

        for s in symptoms:
            s["start_date"] = calculate_start_date(visit_date, s["duration"])
            visit_entry["symptoms"].append(s)

            # Update consolidated symptoms
            grouped[subject_id]["consolidated_symptoms"][s["name"]] = {
                "name": s["name"],
                "duration": s["duration"],
                "start_date": s["start_date"],
                "last_seen": visit_date
            }

        grouped[subject_id]["visits"].append(visit_entry)

    # Convert defaultdict → normal dict
    patients = []
    for sid, data in grouped.items():
        patients.append({
            "subject_id": sid,
            "visits": data["visits"],
            "consolidated_symptoms": list(data["consolidated_symptoms"].values())
        })

    # Save JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n✅ Done. Processed {len(patients)} patients in {elapsed:.2f} seconds → {OUTPUT_FILE}")

# ============ RUN ============
if __name__ == "__main__":
    process_csv()
