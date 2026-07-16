import json
import re
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import time

# Ollama import
try:
    from ollama import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[WARN] Ollama not installed. Falling back to rule-based extraction only.")

# Symptom normalization map
NORMALIZATION_MAP = {
    "wheezing": "Wheezing",
    "coughing": "Cough",
    "shortness of breath": "Dyspnea",
    "chest pain": "Chest Pain",
    "shoulder pain": "Shoulder Pain",
    "night sweats": "Night Sweats",
    "peripheral edema": "Peripheral Edema",
    "svc syndrome": "Superior Vena Cava Syndrome",
    "digital clubbing": "Digital Clubbing",
    "hypercalcemia": "Hypercalcemia",
    "pleuritic pain": "Pleuritic Chest Pain",
    "persistent pneumonia": "Recurrent Pneumonia",
    "yellow fingers": "Yellow Fingers",
    "anxiety": "Anxiety",
    "fatigue": "Fatigue",
    "swallowing difficulty": "Dysphagia",
    "paraneoplastic syndrome": "Paraneoplastic Syndrome",
    "dysphonia": "Dysphonia",
    "unintentional weight loss": "Unintentional Weight Loss",
    "bone pain": "Bone Pain",
    "hemoptysis": "Hemoptysis",
    "recurrent respiratory infections": "Recurrent Respiratory Infections",
    "persistent pneumonia": "Persistent Pneumonia",
    "dizziness": "Dizziness",
    "facial swelling": "Facial Swelling",
    "headache": "Headache",
    "hyponatremia": "Hyponatremia"
}

def clean_duration(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.lower().strip()
    if "present" in raw or "recent" in raw:
        return "present"
    return raw

def normalize_symptom(name: str, duration: str) -> dict:
    key = name.lower().strip()
    normalized_name = NORMALIZATION_MAP.get(key, name.title())
    cleaned_duration = clean_duration(duration)
    return {"name": normalized_name, "duration": cleaned_duration}

def calculate_start_date(visit_date_str: str, duration: str) -> str:
    try:
        # Parse visit_date
        visit_dt = datetime.strptime(visit_date_str.split()[1], "%d:%m:%Y")
        if not duration or duration.lower() == "present":
            return visit_dt.strftime("%Y-%m-%d")
        # Simple regex to extract number + unit
        m = re.match(r"(\d+)\s*(day|week|month|year)s?", duration.lower())
        if m:
            num, unit = int(m.group(1)), m.group(2)
            if unit == "day":
                start_dt = visit_dt - timedelta(days=num)
            elif unit == "week":
                start_dt = visit_dt - timedelta(weeks=num)
            elif unit == "month":
                start_dt = visit_dt - timedelta(days=30*num)
            elif unit == "year":
                start_dt = visit_dt - timedelta(days=365*num)
            return start_dt.strftime("%Y-%m-%d")
    except Exception as e:
        return visit_date_str
    return visit_date_str

# ----------------- Regex-based fallback -----------------
def extract_symptoms_regex(note: str):
    symptoms = []
    pattern = re.findall(r"([\w\s\-]+)\s*\(since\s*([\w\s]+)\)", note.lower())
    for name, duration in pattern:
        symptoms.append(normalize_symptom(name, duration))
    return symptoms

# ----------------- Ollama LLM extraction -----------------
def extract_symptoms_ollama(note: str):
    if not OLLAMA_AVAILABLE:
        return extract_symptoms_regex(note)
    try:
        client = Ollama()
        prompt = f"""
        Extract all symptoms and their durations from this clinical note.
        Return ONLY a JSON list with 'name' and 'duration'.

        Note: {note}
        """
        response = client.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        text = response['choices'][0]['message']['content']
        # Extract JSON
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return [normalize_symptom(s.get("name", ""), s.get("duration", "")) for s in parsed]
    except Exception as e:
        return extract_symptoms_regex(note)
    return extract_symptoms_regex(note)

# ----------------- Process CSV -----------------
def process_csv(input_csv="synthetic_lung_cancer.csv", output_json="patient_symptoms_output.json"):
    start_time = time.time()
    df = pd.read_csv(input_csv, sep="\t")
    # Fix column parsing if entire row is a single string
    if len(df.columns) == 1:
        df = pd.read_csv(input_csv, sep=",")
    print(f"CSV columns detected: {df.columns.tolist()}")

    patients = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing patients"):
        subject_id = str(row["subject_id"]).strip()
        visit_date = str(row["visit_date"]).strip()
        note = str(row.get("clinical_notes", ""))

        # Extract symptoms
        symptoms = extract_symptoms_ollama(note)
        # Calculate start_dates
        for s in symptoms:
            s['start_date'] = calculate_start_date(visit_date, s['duration'])

        # Multi-visit
        if subject_id not in patients:
            patients[subject_id] = {"subject_id": subject_id, "visits": [], "consolidated_symptoms": {}}

        patients[subject_id]["visits"].append({"visit_date": visit_date, "symptoms": symptoms})

        # Update consolidated_symptoms
        for s in symptoms:
            name = s['name']
            existing = patients[subject_id]["consolidated_symptoms"].get(name)
            if not existing or existing['start_date'] > s['start_date']:
                patients[subject_id]["consolidated_symptoms"][name] = {
                    "name": name,
                    "duration": s['duration'],
                    "start_date": s['start_date'],
                    "last_seen": visit_date
                }

    # Convert consolidated_symptoms dict to list
    for pid in patients:
        patients[pid]["consolidated_symptoms"] = list(patients[pid]["consolidated_symptoms"].values())

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(list(patients.values()), f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n✅ Done. Processed {len(patients)} patients in {elapsed:.2f} seconds → {output_json}")

# ----------------- ENTRY POINT -----------------
if __name__ == "__main__":
    process_csv()
