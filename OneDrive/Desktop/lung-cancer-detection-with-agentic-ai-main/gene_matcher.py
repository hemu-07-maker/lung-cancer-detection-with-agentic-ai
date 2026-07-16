import json
import networkx as nx
from tqdm import tqdm
import time

"""
Agent 2: Gene Matcher (Knowledge Graph-Based)

- Loads patient symptoms from Agent 1 (patient_symptoms_output.json).
- Builds a lung cancer knowledge graph (symptom → disease → gene).
- Performs multi-hop traversal to link symptoms → diseases → genes.
- Saves enriched patient data with matched diseases & genes.
"""

# =========================================================
# STEP 1: Knowledge Graph Construction
# =========================================================
def build_lung_cancer_kg():
    """
    Build a simple Lung Cancer Knowledge Graph.
    Replace/extend this with DisGeNET, UMLS, or custom biomedical KG.
    """
    G = nx.Graph()

    # --- Symptoms ---
    symptoms = [
        "Cough", "Dyspnea", "Chest Pain", "Fatigue", "Hemoptysis",
        "Weight Loss", "Wheezing", "Anxiety", "Night Sweats",
        "Peripheral Edema", "Dysphagia", "Paraneoplastic Syndrome"
    ]
    for s in symptoms:
        G.add_node(s, type="symptom")

    # --- Diseases ---
    diseases = [
        "Lung Cancer", "Small Cell Lung Cancer", "Non-Small Cell Lung Cancer"
    ]
    for d in diseases:
        G.add_node(d, type="disease")

    # --- Genes (example set of lung cancer drivers) ---
    genes = ["TP53", "EGFR", "KRAS", "ALK", "ROS1"]
    for g in genes:
        G.add_node(g, type="gene")

    # --- Symptom → Disease ---
    G.add_edge("Cough", "Lung Cancer")
    G.add_edge("Dyspnea", "Lung Cancer")
    G.add_edge("Chest Pain", "Lung Cancer")
    G.add_edge("Fatigue", "Lung Cancer")
    G.add_edge("Hemoptysis", "Lung Cancer")
    G.add_edge("Weight Loss", "Lung Cancer")
    G.add_edge("Wheezing", "Lung Cancer")
    G.add_edge("Anxiety", "Lung Cancer")
    G.add_edge("Night Sweats", "Lung Cancer")
    G.add_edge("Peripheral Edema", "Lung Cancer")
    G.add_edge("Dysphagia", "Lung Cancer")
    G.add_edge("Paraneoplastic Syndrome", "Lung Cancer")

    # --- Disease → Subtypes ---
    G.add_edge("Lung Cancer", "Small Cell Lung Cancer")
    G.add_edge("Lung Cancer", "Non-Small Cell Lung Cancer")

    # --- Disease → Gene associations ---
    G.add_edge("Small Cell Lung Cancer", "TP53")
    G.add_edge("Small Cell Lung Cancer", "KRAS")
    G.add_edge("Non-Small Cell Lung Cancer", "EGFR")
    G.add_edge("Non-Small Cell Lung Cancer", "ALK")
    G.add_edge("Non-Small Cell Lung Cancer", "ROS1")

    return G


# =========================================================
# STEP 2: Map Symptoms → Genes
# =========================================================
def map_symptoms_to_genes(symptoms, kg: nx.Graph):
    """
    Traverse the KG:
      Symptom → Disease → Gene
    Returns matched diseases & genes for the given symptoms.
    """
    matched_genes = set()
    matched_diseases = set()

    for symptom in symptoms:
        sname = symptom.get("name", "").strip()
        if sname in kg.nodes and kg.nodes[sname]["type"] == "symptom":
            # Hop 1: Symptom → Disease
            for disease in nx.neighbors(kg, sname):
                if kg.nodes[disease]["type"] == "disease":
                    matched_diseases.add(disease)
                    # Hop 2: Disease → Gene
                    for gene in nx.neighbors(kg, disease):
                        if kg.nodes[gene]["type"] == "gene":
                            matched_genes.add(gene)

    return {
        "diseases": sorted(list(matched_diseases)),
        "genes": sorted(list(matched_genes))
    }


# =========================================================
# STEP 3: Process Patients
# =========================================================
def process_patient_symptoms(
    input_json="patient_symptoms_output.json",
    output_json="patient_gene_matches.json"
):
    start_time = time.time()

    # Load patient data
    with open(input_json, "r", encoding="utf-8") as f:
        patients = json.load(f)

    # Build KG
    kg = build_lung_cancer_kg()

    results = []
    for patient in tqdm(patients, desc="Mapping patients"):
        consolidated = patient.get("consolidated_symptoms", [])
        matches = map_symptoms_to_genes(consolidated, kg)

        results.append({
            "subject_id": patient.get("subject_id", ""),
            "symptoms": consolidated,
            "matched_diseases": matches["diseases"],
            "matched_genes": matches["genes"]
        })

    # Save results
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n✅ Done. Processed {len(results)} patients in {elapsed:.2f} seconds → {output_json}")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    process_patient_symptoms()
