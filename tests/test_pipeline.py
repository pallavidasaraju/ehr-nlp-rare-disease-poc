"""
Unit Tests for the EHR NLP Pipeline
====================================

Tests cover:
  1. Entity extraction from clinical text
  2. Ontology mapping (ICD-10 and SNOMED-CT codes)
  3. Knowledge graph construction and node types

Run with:
    pytest tests/test_pipeline.py -v
"""

import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Ensure src is on the path for imports
# ---------------------------------------------------------------------------
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from ehr_nlp_pipeline import EHRNLPPipeline, CLINICAL_NOTES
from ontology_mapper import OntologyMapper
from knowledge_graph import BiomedicalKnowledgeGraph


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def pipeline() -> EHRNLPPipeline:
    """Create a shared pipeline instance for all tests in this module."""
    return EHRNLPPipeline()


@pytest.fixture(scope="module")
def mapper() -> OntologyMapper:
    """Create a shared ontology mapper instance."""
    return OntologyMapper()


@pytest.fixture(scope="module")
def knowledge_graph() -> BiomedicalKnowledgeGraph:
    """Create a knowledge graph with some test data."""
    kg = BiomedicalKnowledgeGraph()
    kg.link_patient_disease("Patient_001", "Ehlers-Danlos Syndrome")
    kg.link_disease_gene("Ehlers-Danlos Syndrome", "COL5A1")
    kg.link_disease_treatment("Ehlers-Danlos Syndrome", "Physical Therapy")
    kg.link_patient_symptom("Patient_001", "Hypermobile Joints")
    return kg


# ===================================================================
# Entity Extraction Tests
# ===================================================================

class TestEntityExtraction:
    """Tests for entity extraction from clinical notes."""

    def test_extracts_ehlers_danlos(self, pipeline: EHRNLPPipeline) -> None:
        """Pipeline should detect Ehlers-Danlos Syndrome in clinical text."""
        text = (
            "Patient presents with hypermobile joints and skin "
            "hyperextensibility. Family history of Ehlers-Danlos Syndrome."
        )
        entities = pipeline.extract_entities(text, "test_patient")
        entity_names = [e["entity"].lower() for e in entities]
        assert any("ehlers-danlos" in name for name in entity_names), (
            "Expected 'Ehlers-Danlos Syndrome' to be extracted"
        )

    def test_extracts_marfan_syndrome(self, pipeline: EHRNLPPipeline) -> None:
        """Pipeline should detect Marfan Syndrome in clinical text."""
        text = "Clinical presentation consistent with Marfan Syndrome."
        entities = pipeline.extract_entities(text, "test_patient")
        entity_names = [e["entity"].lower() for e in entities]
        assert any("marfan" in name for name in entity_names), (
            "Expected 'Marfan Syndrome' to be extracted"
        )

    def test_extracts_wilsons_disease(self, pipeline: EHRNLPPipeline) -> None:
        """Pipeline should detect Wilson's Disease in clinical text."""
        text = "Findings are consistent with Wilson's Disease. ATP7B gene testing ordered."
        entities = pipeline.extract_entities(text, "test_patient")
        entity_names = [e["entity"].lower() for e in entities]
        assert any("wilson" in name for name in entity_names), (
            "Expected 'Wilson's Disease' to be extracted"
        )

    def test_extracts_gene_names(self, pipeline: EHRNLPPipeline) -> None:
        """Pipeline should extract gene identifiers like COL5A1 and FBN1."""
        text = "Genetic testing revealed COL5A1 mutation. FBN1 also tested."
        entities = pipeline.extract_entities(text, "test_patient")
        gene_entities = [e for e in entities if e["category"] == "GENE"]
        gene_names = [e["entity"] for e in gene_entities]
        assert "COL5A1" in gene_names, "Expected COL5A1 to be extracted"
        assert "FBN1" in gene_names, "Expected FBN1 to be extracted"

    def test_extracts_treatments(self, pipeline: EHRNLPPipeline) -> None:
        """Pipeline should extract treatment terms."""
        text = "Started on penicillamine therapy with zinc supplementation."
        entities = pipeline.extract_entities(text, "test_patient")
        treatment_entities = [e for e in entities if e["category"] == "TREATMENT"]
        treatment_names = [e["entity"].lower() for e in treatment_entities]
        assert any("penicillamine" in name for name in treatment_names), (
            "Expected 'penicillamine' to be extracted"
        )

    def test_extracts_symptoms(self, pipeline: EHRNLPPipeline) -> None:
        """Pipeline should extract symptom terms."""
        text = "Patient presents with tremor and dystonia."
        entities = pipeline.extract_entities(text, "test_patient")
        symptom_entities = [e for e in entities if e["category"] == "SYMPTOM"]
        symptom_names = [e["entity"].lower() for e in symptom_entities]
        assert "tremor" in symptom_names, "Expected 'tremor' to be extracted"
        assert "dystonia" in symptom_names, "Expected 'dystonia' to be extracted"

    def test_no_duplicates_per_note(self, pipeline: EHRNLPPipeline) -> None:
        """Each entity should appear at most once per note."""
        text = (
            "Ehlers-Danlos Syndrome confirmed. History of Ehlers-Danlos "
            "Syndrome in family."
        )
        entities = pipeline.extract_entities(text, "test_patient")
        disease_entities = [e for e in entities if e["category"] == "DISEASE"]
        entity_names = [e["entity"].lower() for e in disease_entities]
        # Check no exact duplicates
        assert len(entity_names) == len(set(entity_names)), (
            "Duplicate entities found in extraction"
        )

    def test_process_notes_returns_dataframe(self, pipeline: EHRNLPPipeline) -> None:
        """process_notes should return a DataFrame with expected columns."""
        import pandas as pd

        df = pipeline.process_notes(CLINICAL_NOTES)
        assert isinstance(df, pd.DataFrame)
        expected_columns = {"patient_id", "entity", "category", "icd10", "snomed_ct"}
        assert expected_columns.issubset(set(df.columns)), (
            f"Missing columns: {expected_columns - set(df.columns)}"
        )

    def test_process_notes_finds_all_patients(self, pipeline: EHRNLPPipeline) -> None:
        """All patients from the sample notes should appear in results."""
        df = pipeline.process_notes(CLINICAL_NOTES)
        patient_ids = set(df["patient_id"].unique())
        expected_ids = {note["patient_id"] for note in CLINICAL_NOTES}
        assert expected_ids == patient_ids, (
            f"Missing patients: {expected_ids - patient_ids}"
        )


# ===================================================================
# Ontology Mapping Tests
# ===================================================================

class TestOntologyMapping:
    """Tests for disease-to-ontology-code mapping."""

    def test_icd10_ehlers_danlos(self, mapper: OntologyMapper) -> None:
        """Ehlers-Danlos Syndrome should map to ICD-10 Q79.6."""
        code = mapper.get_icd10("Ehlers-Danlos Syndrome")
        assert code == "Q79.6"

    def test_icd10_marfan(self, mapper: OntologyMapper) -> None:
        """Marfan Syndrome should map to ICD-10 Q87.40."""
        code = mapper.get_icd10("Marfan Syndrome")
        assert code == "Q87.40"

    def test_icd10_wilsons(self, mapper: OntologyMapper) -> None:
        """Wilson's Disease should map to ICD-10 E83.01."""
        code = mapper.get_icd10("Wilson's Disease")
        assert code == "E83.01"

    def test_icd10_duchenne(self, mapper: OntologyMapper) -> None:
        """Duchenne Muscular Dystrophy should map to ICD-10 G71.01."""
        code = mapper.get_icd10("Duchenne Muscular Dystrophy")
        assert code == "G71.01"

    def test_snomed_ehlers_danlos(self, mapper: OntologyMapper) -> None:
        """Ehlers-Danlos Syndrome should map to SNOMED-CT 398114001."""
        code = mapper.get_snomed("Ehlers-Danlos Syndrome")
        assert code == "398114001"

    def test_snomed_wilsons(self, mapper: OntologyMapper) -> None:
        """Wilson's Disease should map to SNOMED-CT 88518009."""
        code = mapper.get_snomed("Wilson's Disease")
        assert code == "88518009"

    def test_unknown_disease_returns_none(self, mapper: OntologyMapper) -> None:
        """An unknown disease should return None for all codes."""
        assert mapper.get_icd10("Totally Made Up Disease") is None
        assert mapper.get_snomed("Totally Made Up Disease") is None

    def test_case_insensitive_lookup(self, mapper: OntologyMapper) -> None:
        """Lookups should be case-insensitive."""
        assert mapper.get_icd10("ehlers-danlos syndrome") == "Q79.6"
        assert mapper.get_icd10("EHLERS-DANLOS SYNDROME") == "Q79.6"

    def test_full_mapping_structure(self, mapper: OntologyMapper) -> None:
        """get_full_mapping should return a dict with expected keys."""
        result = mapper.get_full_mapping("Marfan Syndrome")
        assert "disease" in result
        assert "icd10" in result
        assert "snomed_ct" in result
        assert result["icd10"] == "Q87.40"

    def test_gene_disease_association(self, mapper: OntologyMapper) -> None:
        """Gene-to-disease mapping should return correct associations."""
        diseases = mapper.get_diseases_for_gene("COL5A1")
        assert "ehlers-danlos syndrome" in diseases

    def test_hpo_mapping(self, mapper: OntologyMapper) -> None:
        """HPO mapping should return correct phenotype codes."""
        code = mapper.get_hpo("hypermobile joints")
        assert code == "HP:0001382"


# ===================================================================
# Knowledge Graph Tests
# ===================================================================

class TestKnowledgeGraph:
    """Tests for knowledge graph construction."""

    def test_graph_has_patient_nodes(
        self, knowledge_graph: BiomedicalKnowledgeGraph
    ) -> None:
        """Graph should contain patient nodes."""
        patients = knowledge_graph.get_nodes_by_type("patient")
        assert len(patients) > 0, "Expected at least one patient node"
        assert "Patient_001" in patients

    def test_graph_has_disease_nodes(
        self, knowledge_graph: BiomedicalKnowledgeGraph
    ) -> None:
        """Graph should contain disease nodes."""
        diseases = knowledge_graph.get_nodes_by_type("disease")
        assert len(diseases) > 0, "Expected at least one disease node"
        assert "Ehlers-Danlos Syndrome" in diseases

    def test_graph_has_gene_nodes(
        self, knowledge_graph: BiomedicalKnowledgeGraph
    ) -> None:
        """Graph should contain gene nodes."""
        genes = knowledge_graph.get_nodes_by_type("gene")
        assert len(genes) > 0, "Expected at least one gene node"
        assert "COL5A1" in genes

    def test_graph_has_treatment_nodes(
        self, knowledge_graph: BiomedicalKnowledgeGraph
    ) -> None:
        """Graph should contain treatment nodes."""
        treatments = knowledge_graph.get_nodes_by_type("treatment")
        assert len(treatments) > 0, "Expected at least one treatment node"

    def test_graph_has_symptom_nodes(
        self, knowledge_graph: BiomedicalKnowledgeGraph
    ) -> None:
        """Graph should contain symptom nodes."""
        symptoms = knowledge_graph.get_nodes_by_type("symptom")
        assert len(symptoms) > 0, "Expected at least one symptom node"

    def test_expected_node_types(
        self, knowledge_graph: BiomedicalKnowledgeGraph
    ) -> None:
        """Graph should have all expected node types."""
        node_types = knowledge_graph.get_node_types()
        expected = {"patient", "disease", "gene", "treatment", "symptom"}
        assert expected == node_types, (
            f"Missing node types: {expected - node_types}"
        )

    def test_graph_has_edges(
        self, knowledge_graph: BiomedicalKnowledgeGraph
    ) -> None:
        """Graph should have edges connecting nodes."""
        assert knowledge_graph.graph.number_of_edges() > 0, (
            "Expected at least one edge in the graph"
        )

    def test_patient_disease_edge(
        self, knowledge_graph: BiomedicalKnowledgeGraph
    ) -> None:
        """Patient_001 should be connected to Ehlers-Danlos Syndrome."""
        assert knowledge_graph.graph.has_edge(
            "Patient_001", "Ehlers-Danlos Syndrome"
        )

    def test_summary_counts(
        self, knowledge_graph: BiomedicalKnowledgeGraph
    ) -> None:
        """Summary should return correct node counts."""
        summary = knowledge_graph.summary()
        assert summary["patient"] == 1
        assert summary["disease"] == 1
        assert summary["gene"] == 1
        assert summary["treatment"] == 1
        assert summary["symptom"] == 1

    def test_build_from_dataframe(self, pipeline: EHRNLPPipeline) -> None:
        """Knowledge graph should build correctly from pipeline output."""
        import pandas as pd

        df = pipeline.process_notes(CLINICAL_NOTES)
        kg = BiomedicalKnowledgeGraph()
        kg.build_from_dataframe(df)

        assert kg.graph.number_of_nodes() > 0
        assert kg.graph.number_of_edges() > 0
        assert len(kg.get_nodes_by_type("patient")) > 0
        assert len(kg.get_nodes_by_type("disease")) > 0
