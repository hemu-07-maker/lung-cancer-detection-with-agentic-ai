"""
patient_journey_extractor.py

- Reads CSV that must contain at least: clinical_notes
- Treats each row as a separate "patient" and uses the row index as subject_id
- Does NOT use visit_date or assign any dates
- Attempts to call an LLM (Ollama via LangChain) for extraction; falls back to a rule-based keyword extractor if LLM isn't available or errors
- Outputs JSON mapping row-index -> { subject_id, symptoms }
- Shows tqdm progress and prints total elapsed time
"""

import os
import json
import re
import time
from typing import Dict, Any, List
from datetime import datetime
from tqdm import tqdm

import pandas as pd
import dateparser

# Try to import Ollama / LLMChain from the newer locations first, then fall back.
chain = None
try:
    # preferred community import (if installed)
    from langchain_community.llms import Ollama  # type: ignore
    from langchain.chains import LLMChain  # type: ignore
    from langchain.prompts import PromptTemplate  # type: ignore
except Exception:
    try:
        # fallback older imports (may produce deprecation warnings)
        from langchain.llms import Ollama  # type: ignore
        from langchain import LLMChain  # type: ignore
        from langchain.prompts import PromptTemplate  # type: ignore
    except Exception:
        # LLM unavailable; we'll fallback to rule-based extractor
        Ollama = None
        LLMChain = None
        PromptTemplate = None

# ----------------- CONFIG -----------------
INPUT_CSV = r"D:\\User Prajwala\\OneDrive\\Desktop\\Sem Project\\synthetic_lung_cancer.csv"
OUTPUT_JSON = "patient_symptoms_output.json"

# LLM / Ollama config (only used if Ollama import succeeded)
OLLAMA_MODEL = "llama2"
OLLAMA_BASE_URL = "http://localhost:11434"

# Symptom vocabulary (fallback)
SYMPTOM_VOCAB = [
    "yellow fingers", "anxiety", "fatigue", "wheezing", "coughing",
    "shortness of breath", "swallowing difficulty", "chest pain",
    "unexplained weight loss", "fever", "hemoptysis", "dizziness"
]

# ----------------- PROMPT -----------------
FEW_SHOT_EXAMPLES = [
    # keep small representative few-shot examples to help LLM extract structure
    {
        "clinical_note": "Patient reports fatigue and unexplained weight loss starting two weeks ago, now worse.",
        "visit_date": "2023-05-10",
        "extracted": {
            "fatigue": {
                "symptom_start_date": "2023-04-26",
                "symptom_presented_date": "2023-05-10",
                "reflection_duration": "14 days"
            }
        }
    }
]

def build_few_shot_string(examples: List[Dict[str, Any]]) -> str:
    out = []
    for ex in examples:
        out.append(f"Clinical Note: {ex['clinical_note']}")
        out.append(f"Visit Date: {ex['visit_date']}")
        out.append("Extracted Symptoms JSON:")
        out.append(json.dumps(ex['extracted'], indent=2))
        out.append("---")
    return "\n".join(out)

FEW_SHOT_STR = build_few_shot_string(FEW_SHOT_EXAMPLES)

PROMPT_TEMPLATE_STR = """
You are a clinical extraction assistant.

Task:
Extract all symptoms mentioned in the clinical note. For each symptom output:
- "symptom_start_date": earliest date the patient reports symptom (YYYY-MM-DD) if present, otherwise empty string.
- "symptom_presented_date": the visit date (YYYY-MM-DD) if present, otherwise empty string.
- "reflection_duration": number of days between symptom_start_date and symptom_presented_date as "<N> days"; use "0 days" if unknown.

Rules:
- If a start/presented date cannot be determined, set that field to an empty string and reflection_duration to "0 days".
- Output valid JSON object mapping symptom name -> {{ "symptom_start_date": ..., "symptom_presented_date": ..., "reflection_duration": ... }}.

Examples (few-shot):
{few_shot}

Now extract for this note:

Clinical Note: {clinical_note}
Visit Date: {visit_date}

Output ONLY the JSON object.
"""


# If PromptTemplate is available, create the prompt template
if PromptTemplate is not None:
    PROMPT = PromptTemplate(input_variables=["few_shot", "clinical_note", "visit_date"],
                            template=PROMPT_TEMPLATE_STR)
else:
    PROMPT = None

# ----------------- LLM INIT -----------------
def init_llm():
    """
    Try to initialize an Ollama-based LLM chain. If unable, return None.
    """
    global chain
    if Ollama is None or LLMChain is None or PROMPT is None:
        print("[LLM] LangChain/Ollama import not available — using rule-based fallback.")
        chain = None
        return None
    try:
        llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        chain_local = LLMChain(llm=llm, prompt=PROMPT)
        chain = chain_local
        print("[LLM] Ollama initialized (LangChain).")
        return chain_local
    except Exception as e:
        print(f"[LLM] Failed to initialize Ollama/LLMChain: {e}\nUsing rule-based fallback.")
        chain = None
        return None

init_llm()

# ----------------- Helpers -----------------
def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """Extract first JSON object found in text; robust to surrounding text."""
    if not raw_text or not isinstance(raw_text, str):
        return {}
    text = raw_text.strip()
    # try direct load
    try:
        return json.loads(text)
    except Exception:
        pass
    # find first '{' and last '}' and try that slice
    if '{' in text and '}' in text:
        try:
            start = text.index('{')
            end = text.rfind('}')
            candidate = text[start:end+1]
            return json.loads(candidate)
        except Exception:
            pass
    # fallback empty
    return {}

def safe_parse_date(text: str, ref_date: str = "") -> str:
    """
    Try absolute parse (YYYY-MM-DD). If not parseable or no ref_date provided for relative parsing,
    return empty string.
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.strip()
    # try direct YYYY-MM-DD
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1)
    # if ref_date provided, try parse relative phrases with dateparser
    if ref_date and isinstance(ref_date, str) and ref_date.strip():
        try:
            base = datetime.strptime(ref_date, "%Y-%m-%d")
            dt = dateparser.parse(text, settings={'RELATIVE_BASE': base, 'PREFER_DATES_FROM': 'past'})
            if dt:
                return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    # otherwise return empty
    return ""

def parse_duration_to_days(duration_text: str, default_days: int = 0) -> int:
    if duration_text is None:
        return default_days
    t = str(duration_text).lower().strip()
    m = re.search(r"(\d+)", t)
    if m:
        n = int(m.group(1))
        if 'week' in t:
            return n * 7
        if 'month' in t:
            return n * 30
        return n
    return default_days

def rule_based_extract(note: str) -> Dict[str, Any]:
    """
    Simple keyword-based extractor returning symptom -> {start:'', presented:'', reflection:'0 days'}
    """
    found = {}
    note_lower = (note or "").lower()
    for s in SYMPTOM_VOCAB:
        if s in note_lower:
            found[s] = {
                "symptom_start_date": "",
                "symptom_presented_date": "",
                "reflection_duration": "0 days"
            }
    return found

def normalize_extracted(extracted: Dict[str, Any], visit_date: str = "") -> Dict[str, Any]:
    """
    Convert LLM output into normalized dict with empty strings for unknown dates and '0 days' default.
    If visit_date is given, symptom_start_date can be calculated as visit_date - duration.
    """
    normalized = {}
    if not extracted or not isinstance(extracted, dict):
        return normalized

    for sym, vals in extracted.items():
        if not isinstance(vals, dict):
            normalized[sym] = {
                "symptom_start_date": "",
                "symptom_presented_date": visit_date if visit_date else "",
                "reflection_duration": "0 days"
            }
            continue

        raw_start = vals.get("symptom_start_date") or vals.get("start_date") or vals.get("onset") or ""
        raw_presented = vals.get("symptom_presented_date") or vals.get("presented_date") or visit_date or ""
        raw_duration = vals.get("reflection_duration") or vals.get("duration") or vals.get("days") or ""

        ### 🔥 Parse dates properly
        presented_date = safe_parse_date(str(raw_presented), ref_date="") or visit_date

        # If explicit start_date present, parse it
        start_date = safe_parse_date(str(raw_start), ref_date=presented_date)

        # If no start_date but duration & visit_date present → back-calc
        if not start_date and raw_duration and presented_date:
            days = parse_duration_to_days(str(raw_duration), default_days=0)
            try:
                base = datetime.strptime(presented_date, "%Y-%m-%d")
                calc = base - pd.to_timedelta(days, unit="d")
                start_date = calc.strftime("%Y-%m-%d")
            except Exception:
                start_date = ""
        else:
            days = parse_duration_to_days(str(raw_duration), default_days=0)

        # If both start & presented exist, recalc reflection_duration
        if start_date and presented_date:
            try:
                d1 = datetime.strptime(start_date, "%Y-%m-%d")
                d2 = datetime.strptime(presented_date, "%Y-%m-%d")
                days = (d2 - d1).days
            except Exception:
                pass

        normalized[sym] = {
            "symptom_start_date": start_date,
            "symptom_presented_date": presented_date,
            "reflection_duration": f"{days} days"
        }

    return normalized

# ----------------- LLM CALL WRAPPER -----------------
def extract_with_llm(note: str, visit_date: str = "") -> Dict[str, Any]:
    note_safe = note or ""
    if chain is None:
        return rule_based_extract(note_safe)

    try:
        prompt_input = {
            "few_shot": FEW_SHOT_STR,
            "clinical_note": note_safe,
            "visit_date": visit_date
        }
        try:
            raw = chain.run(
                few_shot=FEW_SHOT_STR,
                clinical_note=note_safe,
                visit_date=visit_date
            )
        except Exception as e:
            print(f"[LLM error] {e} — using rule-based fallback for this note.")
            return rule_based_extract(note_safe)
        parsed = parse_llm_json(raw)
        if not parsed:
            return rule_based_extract(note_safe)

        return normalize_extracted(parsed, visit_date=visit_date)   # ✅ pass visit_date
    except Exception as e:
        print(f"[LLM error] {e} — using rule-based fallback for this note.")
        return rule_based_extract(note_safe)


# ----------------- PROCESS DATASET (row-index as id) -----------------
def process_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    if 'clinical_notes' not in df.columns:
        raise ValueError("Input CSV must contain the 'clinical_notes' column.")

    output: Dict[str, Any] = {}
    total = len(df)

    for idx, row in tqdm(df.iterrows(), total=total, desc="Processing rows"):
        note = str(row.get('clinical_notes') or "").strip()
        visit_date = str(row.get('visit_date') or "").strip()  # ✅ use visit_date
        if not note:
            continue

        patient_id = str(idx)
        extracted = extract_with_llm(note, visit_date=visit_date)

        if not extracted:
            extracted = {}

        output[patient_id] = {
            "subject_id": patient_id,
            "symptoms": extracted
        }

    return output


# ----------------- MAIN -----------------
def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found at {INPUT_CSV}. Please set INPUT_CSV path.")

    # read with latin1 to avoid common decode errors
    df = pd.read_csv(INPUT_CSV, dtype=str, encoding='latin1')

    # ✅ normalize column names (lowercase + underscores)
    rename_map = {
        "subject_id": "subject_id",
        "visit_date": "visit_date",
        "GENDER": "gender",
        "AGE": "age",
        "SMOKING": "smoking",
        "YELLOW_FINGERS": "yellow_fingers",
        "ANXIETY": "anxiety",
        "PEER_PRESSURE": "peer_pressure",
        "CHRONIC DISEASE": "chronic_disease",
        "FATIGUE": "fatigue",
        "ALLERGY": "allergy",
        "WHEEZING": "wheezing",
        "ALCOHOL CONSUMING": "alcohol_consuming",
        "COUGHING": "coughing",
        "SHORTNESS OF BREATH": "shortness_of_breath",
        "SWALLOWING DIFFICULTY": "swallowing_difficulty",
        "CHEST PAIN": "chest_pain",
        "LUNG_CANCER": "lung_cancer",
        "clinical_notes": "clinical_notes"
    }
    df.rename(columns=rename_map, inplace=True)

    result = process_dataframe(df)

    # write JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Done. Output written to {OUTPUT_JSON}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
