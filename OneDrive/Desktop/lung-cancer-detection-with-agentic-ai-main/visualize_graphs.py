import json
import os
import networkx as nx
import matplotlib.pyplot as plt

def build_patient_graph(patient, kg: nx.DiGraph):
    """
    Build a subgraph for a single patient from the global knowledge graph.
    """
    G = nx.DiGraph()
    symptoms = [s["name"] for s in patient.get("consolidated_symptoms", [])]

    for s in symptoms:
        if s in kg.nodes and kg.nodes[s]["type"] == "symptom":
            G.add_node(s, type="symptom")

            for disease in kg.neighbors(s):
                if kg.nodes[disease]["type"] == "disease":
                    G.add_node(disease, type="disease")
                    G.add_edge(s, disease)

                    for gene in kg.neighbors(disease):
                        if kg.nodes[gene]["type"] == "gene":
                            G.add_node(gene, type="gene")
                            G.add_edge(disease, gene)
    return G

def visualize_patient_graph(patient, kg, save_path):
    """
    Save a PNG visualization of a patient’s Symptom → Disease → Gene subgraph.
    """
    G = build_patient_graph(patient, kg)

    if G.number_of_nodes() == 0:
        print(f"⚠️ No graph generated for patient {patient['subject_id']}")
        return

    # Color & shape mapping
    color_map = []
    shape_map = {"symptom": "o", "disease": "s", "gene": "^"}
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes by type
    for ntype, shape in shape_map.items():
        nodes = [n for n in G.nodes if G.nodes[n]["type"] == ntype]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_shape=shape,
                               node_size=800,
                               node_color=("skyblue" if ntype == "symptom"
                                           else "orange" if ntype == "disease"
                                           else "lightgreen"))
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True)
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"Patient {patient['subject_id']} – Symptom → Disease → Gene", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, format="png", dpi=300)
    plt.close()
    print(f"✅ Saved graph: {save_path}")

if __name__ == "__main__":
    # Load knowledge graph
    with open("knowledge_graph.json", "r") as f:
        kg_data = json.load(f)
    kg = nx.node_link_graph(kg_data)

    # Load patient-gene matches
    with open("patient_gene_matches.json", "r") as f:
        patients = json.load(f)

    os.makedirs("patient_graphs", exist_ok=True)

    # Process first 30 patients
    for i, patient in enumerate(patients[:30], start=1):
        save_path = os.path.join("patient_graphs", f"patient_{patient['subject_id']}_graph.png")
        visualize_patient_graph(patient, kg, save_path)
