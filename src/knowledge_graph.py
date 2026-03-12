"""
Knowledge Graph Module
======================

Builds a simple biomedical knowledge graph using NetworkX that links:
    Patient -> Disease -> Gene -> Treatment

The graph can be visualized and saved as a PNG image for inspection.
"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PNG generation
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# Color palette for node types
# ---------------------------------------------------------------------------
NODE_COLORS: dict[str, str] = {
    "patient": "#4FC3F7",     # light blue
    "disease": "#EF5350",     # red
    "gene": "#66BB6A",        # green
    "treatment": "#FFA726",   # orange
    "symptom": "#AB47BC",     # purple
}

NODE_SHAPES: dict[str, str] = {
    "patient": "s",      # square
    "disease": "o",      # circle
    "gene": "D",         # diamond
    "treatment": "^",    # triangle up
    "symptom": "v",      # triangle down
}


class BiomedicalKnowledgeGraph:
    """Builds and visualizes a knowledge graph from entity extraction results.

    Nodes represent patients, diseases, genes, treatments, and symptoms.
    Edges represent relationships such as 'diagnosed_with', 'associated_gene',
    'treated_with', and 'has_symptom'.
    """

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()

    # ------------------------------------------------------------------
    # Node / edge helpers
    # ------------------------------------------------------------------

    def add_patient(self, patient_id: str) -> None:
        """Add a patient node to the graph."""
        self.graph.add_node(patient_id, node_type="patient", label=patient_id)

    def add_disease(self, disease_name: str) -> None:
        """Add a disease node to the graph."""
        self.graph.add_node(disease_name, node_type="disease", label=disease_name)

    def add_gene(self, gene_name: str) -> None:
        """Add a gene node to the graph."""
        self.graph.add_node(gene_name, node_type="gene", label=gene_name)

    def add_treatment(self, treatment_name: str) -> None:
        """Add a treatment node to the graph."""
        self.graph.add_node(treatment_name, node_type="treatment", label=treatment_name)

    def add_symptom(self, symptom_name: str) -> None:
        """Add a symptom node to the graph."""
        self.graph.add_node(symptom_name, node_type="symptom", label=symptom_name)

    def link_patient_disease(self, patient_id: str, disease_name: str) -> None:
        """Link a patient to a disease with a 'diagnosed_with' edge."""
        self.add_patient(patient_id)
        self.add_disease(disease_name)
        self.graph.add_edge(patient_id, disease_name, relation="diagnosed_with")

    def link_disease_gene(self, disease_name: str, gene_name: str) -> None:
        """Link a disease to a gene with an 'associated_gene' edge."""
        self.add_disease(disease_name)
        self.add_gene(gene_name)
        self.graph.add_edge(disease_name, gene_name, relation="associated_gene")

    def link_disease_treatment(self, disease_name: str, treatment_name: str) -> None:
        """Link a disease to a treatment with a 'treated_with' edge."""
        self.add_disease(disease_name)
        self.add_treatment(treatment_name)
        self.graph.add_edge(disease_name, treatment_name, relation="treated_with")

    def link_patient_symptom(self, patient_id: str, symptom_name: str) -> None:
        """Link a patient to a symptom with a 'has_symptom' edge."""
        self.add_patient(patient_id)
        self.add_symptom(symptom_name)
        self.graph.add_edge(patient_id, symptom_name, relation="has_symptom")

    # ------------------------------------------------------------------
    # Build from DataFrame
    # ------------------------------------------------------------------

    def build_from_dataframe(self, df: pd.DataFrame) -> None:
        """Populate the graph from the pipeline's entity extraction DataFrame.

        Expected DataFrame columns:
            - patient_id: str
            - entity: str
            - category: str  (one of DISEASE, GENE, TREATMENT, SYMPTOM)

        Args:
            df: Entity extraction results as a pandas DataFrame.
        """
        for _, row in df.iterrows():
            patient_id: str = str(row["patient_id"])
            entity: str = str(row["entity"])
            category: str = str(row["category"]).upper()

            if category == "DISEASE":
                self.link_patient_disease(patient_id, entity)
            elif category == "GENE":
                self.add_gene(entity)
                # Try to link gene to any diseases the patient has
                for neighbor in list(self.graph.neighbors(patient_id)) if self.graph.has_node(patient_id) else []:
                    if self.graph.nodes[neighbor].get("node_type") == "disease":
                        self.link_disease_gene(neighbor, entity)
            elif category == "TREATMENT":
                self.add_treatment(entity)
                # Link treatment to diseases the patient has
                for neighbor in list(self.graph.neighbors(patient_id)) if self.graph.has_node(patient_id) else []:
                    if self.graph.nodes[neighbor].get("node_type") == "disease":
                        self.link_disease_treatment(neighbor, entity)
            elif category == "SYMPTOM":
                self.link_patient_symptom(patient_id, entity)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_nodes_by_type(self, node_type: str) -> list[str]:
        """Return all node names of a given type."""
        return [
            node for node, data in self.graph.nodes(data=True)
            if data.get("node_type") == node_type
        ]

    def get_node_types(self) -> set[str]:
        """Return the set of distinct node types in the graph."""
        return {
            data.get("node_type", "unknown")
            for _, data in self.graph.nodes(data=True)
        }

    def summary(self) -> dict[str, int]:
        """Return a summary of node counts by type."""
        type_counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            ntype = data.get("node_type", "unknown")
            type_counts[ntype] = type_counts.get(ntype, 0) + 1
        return type_counts

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, output_path: Optional[str] = None) -> str:
        """Draw the knowledge graph and save as a PNG image.

        Args:
            output_path: File path for the saved image. Defaults to
                         'graph.png' in the project root.

        Returns:
            The absolute path of the saved image file.
        """
        if output_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_path = os.path.join(project_root, "graph.png")

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_title(
            "Biomedical Knowledge Graph: Patient - Disease - Gene - Treatment",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        pos = nx.spring_layout(self.graph, k=2.5, iterations=50, seed=42)

        # Draw edges first
        edge_labels = nx.get_edge_attributes(self.graph, "relation")
        nx.draw_networkx_edges(
            self.graph, pos, ax=ax,
            edge_color="#BDBDBD", width=1.5, alpha=0.7,
        )
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, ax=ax,
            font_size=7, font_color="#616161",
        )

        # Draw nodes by type so each type gets its own color
        for node_type, color in NODE_COLORS.items():
            nodes = self.get_nodes_by_type(node_type)
            if nodes:
                nx.draw_networkx_nodes(
                    self.graph, pos, nodelist=nodes, ax=ax,
                    node_color=color, node_size=800, alpha=0.9,
                    edgecolors="#333333", linewidths=1.2,
                )

        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos, ax=ax,
            font_size=8, font_weight="bold",
        )

        # Build legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, edgecolor="#333", label=ntype.capitalize())
            for ntype, color in NODE_COLORS.items()
            if self.get_nodes_by_type(ntype)
        ]
        ax.legend(
            handles=legend_elements, loc="upper left",
            fontsize=10, framealpha=0.9,
        )

        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    kg = BiomedicalKnowledgeGraph()

    # Manually add some relationships
    kg.link_patient_disease("Patient_001", "Ehlers-Danlos Syndrome")
    kg.link_disease_gene("Ehlers-Danlos Syndrome", "COL5A1")
    kg.link_disease_treatment("Ehlers-Danlos Syndrome", "Physical Therapy")
    kg.link_patient_symptom("Patient_001", "Hypermobile Joints")

    kg.link_patient_disease("Patient_002", "Wilson's Disease")
    kg.link_disease_gene("Wilson's Disease", "ATP7B")
    kg.link_disease_treatment("Wilson's Disease", "Penicillamine")

    print("Knowledge Graph Summary:")
    for ntype, count in kg.summary().items():
        print(f"  {ntype}: {count} nodes")

    path = kg.visualize()
    print(f"\nGraph saved to: {path}")
