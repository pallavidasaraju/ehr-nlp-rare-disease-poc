"""
EHR NLP Pipeline for Rare Disease Entity Extraction
====================================================

Main pipeline script that:
  1. Loads simulated clinical notes (EHR text snippets)
  2. Processes them with spaCy NLP + custom PhraseMatcher
  3. Extracts rare disease entities, gene names, and treatments
  4. Maps entities to ICD-10 / SNOMED-CT ontology codes
  5. Builds a knowledge graph linking patients -> diseases -> genes -> treatments
  6. Outputs a structured DataFrame and a human-readable summary report

Run with:
    python src/ehr_nlp_pipeline.py
"""

import os
import sys
from typing import Optional

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

# ---------------------------------------------------------------------------
# Ensure the src package is importable when running as a script
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ontology_mapper import OntologyMapper
from knowledge_graph import BiomedicalKnowledgeGraph


# ===================================================================
# 1. SIMULATED CLINICAL NOTES
# ===================================================================

CLINICAL_NOTES: list[dict[str, str]] = [
    {
        "patient_id": "Patient_001",
        "note": (
            "42-year-old female presents with hypermobile joints, frequent "
            "joint dislocations, and skin hyperextensibility. Patient reports "
            "chronic fatigue and widespread musculoskeletal pain. Family "
            "history significant for Ehlers-Danlos Syndrome in mother and "
            "maternal aunt. Genetic testing revealed COL5A1 mutation. "
            "Current management includes physical therapy and NSAIDs for "
            "pain control. Referral to genetics clinic recommended."
        ),
    },
    {
        "patient_id": "Patient_002",
        "note": (
            "28-year-old male, tall stature (6'5\"), with arachnodactyly and "
            "lens subluxation detected on ophthalmology exam. Echocardiogram "
            "shows aortic root dilation (4.2 cm). Clinical presentation "
            "consistent with Marfan Syndrome. FBN1 gene mutation confirmed "
            "via whole exome sequencing. Started on losartan 50mg daily for "
            "aortic protection. Advised to avoid contact sports and heavy "
            "lifting. Cardiac follow-up in 6 months."
        ),
    },
    {
        "patient_id": "Patient_003",
        "note": (
            "35-year-old male presenting with tremor, dystonia, and "
            "progressive hepatic dysfunction. Liver biopsy shows elevated "
            "copper levels. Slit-lamp examination reveals Kayser-Fleischer "
            "rings. Serum ceruloplasmin markedly reduced at 8 mg/dL (normal "
            "20-35). Findings are consistent with Wilson's Disease. ATP7B "
            "gene testing ordered. Initiated penicillamine therapy with zinc "
            "supplementation. Low-copper diet counseling provided. "
            "Neuropsychiatric evaluation scheduled."
        ),
    },
    {
        "patient_id": "Patient_004",
        "note": (
            "5-year-old male with progressive muscle weakness, difficulty "
            "climbing stairs, and Gowers' sign positive. Creatine kinase "
            "level markedly elevated at 15,000 U/L. EMG shows myopathic "
            "pattern. Clinical suspicion for Duchenne Muscular Dystrophy. "
            "DMD gene analysis confirms out-of-frame deletion in exons 45-50. "
            "Started on deflazacort. Physical therapy and occupational "
            "therapy initiated. Cardiac monitoring with annual echocardiogram "
            "recommended. Family counseling for X-linked inheritance pattern."
        ),
    },
]


# ===================================================================
# 2. ENTITY TERM DICTIONARIES
# ===================================================================

RARE_DISEASE_TERMS: list[str] = [
    "Ehlers-Danlos Syndrome",
    "Marfan Syndrome",
    "Wilson's Disease",
    "Wilson Disease",
    "Huntington's Disease",
    "Huntington Disease",
    "Cystic Fibrosis",
    "Phenylketonuria",
    "Gaucher Disease",
    "Fabry Disease",
    "Pompe Disease",
    "Sickle Cell Disease",
    "Tay-Sachs Disease",
    "Amyotrophic Lateral Sclerosis",
    "Duchenne Muscular Dystrophy",
    "Tuberous Sclerosis",
    "Neurofibromatosis",
]

GENE_TERMS: list[str] = [
    "COL5A1", "COL5A2", "COL3A1",
    "FBN1",
    "ATP7B",
    "HTT",
    "CFTR",
    "PAH",
    "GBA",
    "GLA",
    "GAA",
    "HBB",
    "HEXA",
    "SOD1",
    "DMD",
    "TSC1", "TSC2",
    "NF1",
]

TREATMENT_TERMS: list[str] = [
    "physical therapy",
    "occupational therapy",
    "NSAIDs",
    "losartan",
    "penicillamine",
    "zinc supplementation",
    "deflazacort",
    "gene therapy",
    "enzyme replacement therapy",
    "miglustat",
    "corticosteroids",
]

SYMPTOM_TERMS: list[str] = [
    "hypermobile joints",
    "joint dislocations",
    "skin hyperextensibility",
    "arachnodactyly",
    "lens subluxation",
    "tremor",
    "dystonia",
    "Kayser-Fleischer rings",
    "muscle weakness",
    "chronic fatigue",
    "hepatomegaly",
    "chorea",
    "seizures",
    "cardiomyopathy",
    "tall stature",
]


# ===================================================================
# 3. PIPELINE CLASS
# ===================================================================

class EHRNLPPipeline:
    """NLP pipeline for extracting rare disease entities from clinical notes.

    Uses spaCy as the base NLP engine with custom PhraseMatchers for
    domain-specific entity recognition covering diseases, genes,
    treatments, and symptoms.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Initialize the pipeline with a spaCy model and custom matchers.

        Args:
            model_name: Name of the spaCy language model to load.
        """
        print(f"Loading spaCy model '{model_name}'...")
        self.nlp = spacy.load(model_name)

        # Create PhraseMatchers for each entity category
        self.disease_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.gene_matcher = PhraseMatcher(self.nlp.vocab)
        self.treatment_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.symptom_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        # Add patterns
        self._add_patterns(self.disease_matcher, "DISEASE", RARE_DISEASE_TERMS)
        self._add_patterns(self.gene_matcher, "GENE", GENE_TERMS)
        self._add_patterns(self.treatment_matcher, "TREATMENT", TREATMENT_TERMS)
        self._add_patterns(self.symptom_matcher, "SYMPTOM", SYMPTOM_TERMS)

        # Ontology mapper
        self.mapper = OntologyMapper()

        print("Pipeline initialized successfully.\n")

    def _add_patterns(
        self,
        matcher: PhraseMatcher,
        label: str,
        terms: list[str],
    ) -> None:
        """Add phrase patterns to a PhraseMatcher.

        Args:
            matcher: The PhraseMatcher instance.
            label: Category label for matched entities.
            terms: List of phrase strings to match.
        """
        patterns = [self.nlp.make_doc(term) for term in terms]
        matcher.add(label, patterns)

    def extract_entities(
        self,
        text: str,
        patient_id: str = "unknown",
    ) -> list[dict[str, Optional[str]]]:
        """Extract rare disease entities from a clinical text note.

        Args:
            text: Raw clinical note text.
            patient_id: Identifier for the patient.

        Returns:
            A list of dicts, each containing:
                - patient_id, entity, category, icd10, snomed_ct, start, end
        """
        doc = self.nlp(text)
        results: list[dict[str, Optional[str]]] = []
        seen: set[str] = set()  # Avoid duplicate entities per note

        # Run all matchers
        for matcher, category in [
            (self.disease_matcher, "DISEASE"),
            (self.gene_matcher, "GENE"),
            (self.treatment_matcher, "TREATMENT"),
            (self.symptom_matcher, "SYMPTOM"),
        ]:
            matches = matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                entity_text = span.text
                entity_key = f"{category}:{entity_text.lower()}"

                if entity_key in seen:
                    continue
                seen.add(entity_key)

                # Ontology mapping (only for diseases)
                icd10: Optional[str] = None
                snomed_ct: Optional[str] = None
                if category == "DISEASE":
                    icd10 = self.mapper.get_icd10(entity_text)
                    snomed_ct = self.mapper.get_snomed(entity_text)

                results.append({
                    "patient_id": patient_id,
                    "entity": entity_text,
                    "category": category,
                    "icd10": icd10,
                    "snomed_ct": snomed_ct,
                    "start_char": str(span.start_char),
                    "end_char": str(span.end_char),
                })

        return results

    def process_notes(
        self,
        notes: list[dict[str, str]],
    ) -> pd.DataFrame:
        """Process a batch of clinical notes and return a DataFrame of findings.

        Args:
            notes: List of dicts with keys 'patient_id' and 'note'.

        Returns:
            A pandas DataFrame with all extracted entities.
        """
        all_entities: list[dict[str, Optional[str]]] = []

        for note in notes:
            patient_id = note["patient_id"]
            text = note["note"]
            entities = self.extract_entities(text, patient_id)
            all_entities.extend(entities)

        df = pd.DataFrame(all_entities)
        return df

    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate a human-readable summary report from extraction results.

        Args:
            df: DataFrame of extracted entities.

        Returns:
            A formatted string report.
        """
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("  EHR NLP PIPELINE - RARE DISEASE ENTITY EXTRACTION REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Overall statistics
        lines.append(f"Total entities extracted: {len(df)}")
        if not df.empty:
            lines.append(f"Patients analyzed: {df['patient_id'].nunique()}")
            lines.append("")

            # Category breakdown
            lines.append("--- Entity Counts by Category ---")
            category_counts = df["category"].value_counts()
            for cat, count in category_counts.items():
                lines.append(f"  {cat:12s}: {count}")
            lines.append("")

            # Per-patient breakdown
            lines.append("--- Findings by Patient ---")
            for patient_id in df["patient_id"].unique():
                patient_df = df[df["patient_id"] == patient_id]
                lines.append(f"\n  [{patient_id}]")

                for category in ["DISEASE", "GENE", "TREATMENT", "SYMPTOM"]:
                    cat_df = patient_df[patient_df["category"] == category]
                    if not cat_df.empty:
                        entities = cat_df["entity"].tolist()
                        lines.append(f"    {category:12s}: {', '.join(entities)}")

                # Show ontology mappings for diseases
                disease_df = patient_df[patient_df["category"] == "DISEASE"]
                if not disease_df.empty:
                    lines.append(f"    {'ONTOLOGY':12s}:")
                    for _, row in disease_df.iterrows():
                        icd = row.get("icd10") or "N/A"
                        snomed = row.get("snomed_ct") or "N/A"
                        lines.append(
                            f"      {row['entity']} -> "
                            f"ICD-10: {icd}, SNOMED-CT: {snomed}"
                        )

        lines.append("")
        lines.append("=" * 70)
        lines.append("  End of Report")
        lines.append("=" * 70)

        return "\n".join(lines)


# ===================================================================
# 4. MAIN EXECUTION
# ===================================================================

def main() -> None:
    """Run the full EHR NLP pipeline on simulated clinical notes."""

    # Initialize pipeline
    pipeline = EHRNLPPipeline()

    # Process clinical notes
    print("Processing clinical notes...\n")
    df = pipeline.process_notes(CLINICAL_NOTES)

    # Display the DataFrame
    print("--- Extracted Entities (DataFrame) ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_colwidth", 40)
    print(df.to_string(index=False))
    print()

    # Generate and print the report
    report = pipeline.generate_report(df)
    print(report)

    # Build and visualize the knowledge graph
    print("\nBuilding knowledge graph...")
    kg = BiomedicalKnowledgeGraph()
    kg.build_from_dataframe(df)

    graph_path = kg.visualize()
    print(f"Knowledge graph saved to: {graph_path}")

    # Print graph summary
    print("\n--- Knowledge Graph Summary ---")
    for node_type, count in kg.summary().items():
        print(f"  {node_type:12s}: {count} nodes")
    print(f"  {'edges':12s}: {kg.graph.number_of_edges()} edges")
    print(f"  Total nodes: {kg.graph.number_of_nodes()}")


if __name__ == "__main__":
    main()
