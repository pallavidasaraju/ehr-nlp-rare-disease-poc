"""
Ontology Mapper Module
======================

Maps rare disease names, gene names, and treatment names to standard
medical ontology codes including ICD-10, SNOMED-CT, and HPO.

This module uses a built-in dictionary for demonstration purposes.
In a production system, this would connect to external ontology APIs
such as UMLS, OMIM, or Orphanet.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# ICD-10 mappings for rare diseases
# ---------------------------------------------------------------------------
ICD10_MAPPING: dict[str, str] = {
    "ehlers-danlos syndrome": "Q79.6",
    "marfan syndrome": "Q87.40",
    "wilson's disease": "E83.01",
    "wilson disease": "E83.01",
    "huntington's disease": "G10",
    "huntington disease": "G10",
    "cystic fibrosis": "E84.9",
    "phenylketonuria": "E70.1",
    "gaucher disease": "E75.22",
    "fabry disease": "E75.21",
    "pompe disease": "E74.02",
    "sickle cell disease": "D57.1",
    "tay-sachs disease": "E75.02",
    "amyotrophic lateral sclerosis": "G12.21",
    "duchenne muscular dystrophy": "G71.01",
    "tuberous sclerosis": "Q85.1",
    "neurofibromatosis": "Q85.00",
}

# ---------------------------------------------------------------------------
# SNOMED-CT mappings for rare diseases
# ---------------------------------------------------------------------------
SNOMEDCT_MAPPING: dict[str, str] = {
    "ehlers-danlos syndrome": "398114001",
    "marfan syndrome": "19346006",
    "wilson's disease": "88518009",
    "wilson disease": "88518009",
    "huntington's disease": "58756001",
    "huntington disease": "58756001",
    "cystic fibrosis": "190905008",
    "phenylketonuria": "7573000",
    "gaucher disease": "190794006",
    "fabry disease": "16652001",
    "pompe disease": "124361000119108",
    "sickle cell disease": "417357006",
    "tay-sachs disease": "111385000",
    "amyotrophic lateral sclerosis": "86044005",
    "duchenne muscular dystrophy": "76670001",
    "tuberous sclerosis": "7199000",
    "neurofibromatosis": "19133005",
}

# ---------------------------------------------------------------------------
# HPO (Human Phenotype Ontology) mappings for clinical features
# ---------------------------------------------------------------------------
HPO_MAPPING: dict[str, str] = {
    "hypermobile joints": "HP:0001382",
    "joint hypermobility": "HP:0001382",
    "skin hyperextensibility": "HP:0001030",
    "arachnodactyly": "HP:0001166",
    "hepatomegaly": "HP:0002240",
    "kayser-fleischer rings": "HP:0002172",
    "tremor": "HP:0001337",
    "dystonia": "HP:0001332",
    "chorea": "HP:0002072",
    "pancreatic insufficiency": "HP:0001738",
    "intellectual disability": "HP:0001249",
    "muscle weakness": "HP:0001324",
    "cardiomyopathy": "HP:0001638",
    "lens subluxation": "HP:0001083",
    "tall stature": "HP:0000098",
    "seizures": "HP:0001250",
}

# ---------------------------------------------------------------------------
# Gene-to-disease associations
# ---------------------------------------------------------------------------
GENE_DISEASE_MAPPING: dict[str, list[str]] = {
    "COL5A1": ["ehlers-danlos syndrome"],
    "COL5A2": ["ehlers-danlos syndrome"],
    "COL3A1": ["ehlers-danlos syndrome"],
    "FBN1": ["marfan syndrome"],
    "ATP7B": ["wilson's disease"],
    "HTT": ["huntington's disease"],
    "CFTR": ["cystic fibrosis"],
    "PAH": ["phenylketonuria"],
    "GBA": ["gaucher disease"],
    "GLA": ["fabry disease"],
    "GAA": ["pompe disease"],
    "HBB": ["sickle cell disease"],
    "HEXA": ["tay-sachs disease"],
    "SOD1": ["amyotrophic lateral sclerosis"],
    "DMD": ["duchenne muscular dystrophy"],
    "TSC1": ["tuberous sclerosis"],
    "TSC2": ["tuberous sclerosis"],
    "NF1": ["neurofibromatosis"],
}


class OntologyMapper:
    """Maps disease names, symptoms, and genes to standard ontology codes.

    This mapper provides lookups into ICD-10, SNOMED-CT, and HPO code
    dictionaries. All lookups are case-insensitive.
    """

    def get_icd10(self, disease_name: str) -> Optional[str]:
        """Return the ICD-10 code for a given disease name, or None."""
        return ICD10_MAPPING.get(disease_name.lower().strip())

    def get_snomed(self, disease_name: str) -> Optional[str]:
        """Return the SNOMED-CT code for a given disease name, or None."""
        return SNOMEDCT_MAPPING.get(disease_name.lower().strip())

    def get_hpo(self, phenotype: str) -> Optional[str]:
        """Return the HPO code for a given clinical phenotype, or None."""
        return HPO_MAPPING.get(phenotype.lower().strip())

    def get_diseases_for_gene(self, gene_name: str) -> list[str]:
        """Return the list of diseases associated with a gene."""
        return GENE_DISEASE_MAPPING.get(gene_name.upper().strip(), [])

    def get_full_mapping(self, disease_name: str) -> dict[str, Optional[str]]:
        """Return a dictionary with ICD-10 and SNOMED-CT codes for a disease.

        Returns:
            A dict with keys 'disease', 'icd10', and 'snomed_ct'.
        """
        return {
            "disease": disease_name,
            "icd10": self.get_icd10(disease_name),
            "snomed_ct": self.get_snomed(disease_name),
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mapper = OntologyMapper()
    test_diseases = [
        "Ehlers-Danlos Syndrome",
        "Marfan Syndrome",
        "Wilson's Disease",
        "Cystic Fibrosis",
        "Unknown Disease",
    ]
    for disease in test_diseases:
        mapping = mapper.get_full_mapping(disease)
        print(f"{disease:30s} -> ICD-10: {mapping['icd10'] or 'N/A':10s} | "
              f"SNOMED-CT: {mapping['snomed_ct'] or 'N/A'}")
