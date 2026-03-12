# EHR NLP Pipeline for Rare Disease Entity Extraction

A proof-of-concept NLP pipeline that extracts rare disease entities from simulated Electronic Health Record (EHR) clinical text. This project demonstrates techniques relevant to identifying undiagnosed or miscoded rare disease patients in real-world EHR data, aligned with the mission of Pangaea Data.

## Overview

Rare diseases affect approximately 300 million people worldwide, yet the average time to diagnosis is 5-7 years. Many patients are hidden in EHR systems under generic or incorrect ICD codes. This pipeline demonstrates how NLP can be used to surface potential rare disease patients by analyzing unstructured clinical notes.

## Architecture

```
Clinical Notes (EHR Text)
        |
        v
+------------------+
|  spaCy NLP Engine |  (tokenization, POS tagging, dependency parsing)
+------------------+
        |
        v
+------------------------+
|  Custom PhraseMatcher  |  (rare disease terms, gene names, treatments)
+------------------------+
        |
        v
+--------------------+
|  Ontology Mapper   |  (ICD-10, SNOMED-CT, HPO code mapping)
+--------------------+
        |
        v
+--------------------+
|  Knowledge Graph   |  (Patient -> Disease -> Gene -> Treatment links)
+--------------------+
        |
        v
  Structured Output (DataFrame + Summary Report + Graph Visualization)
```

## Features

- **Entity Extraction**: Uses spaCy with custom PhraseMatcher to identify rare disease names, associated genes, and treatments from clinical text.
- **Ontology Mapping**: Maps extracted entities to standard medical ontology codes (ICD-10, SNOMED-CT) using a built-in dictionary.
- **Knowledge Graph**: Builds a NetworkX graph linking patients to diseases, genes, and treatments, with PNG visualization output.
- **Structured Reporting**: Outputs findings as a pandas DataFrame and prints a human-readable summary.

## Project Structure

```
ehr-nlp-rare-disease-poc/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── src/
│   ├── ehr_nlp_pipeline.py      # Main pipeline script
│   ├── ontology_mapper.py       # Disease-to-ontology code mapper
│   └── knowledge_graph.py       # Knowledge graph builder and visualizer
├── tests/
│   └── test_pipeline.py         # Unit tests
└── graph.png                    # Generated knowledge graph visualization
```

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone or navigate to the project directory
cd ehr-nlp-rare-disease-poc

# Install dependencies
pip install -r requirements.txt

# Download the spaCy English model
python -m spacy download en_core_web_sm
```

## Usage

### Run the main pipeline

```bash
python src/ehr_nlp_pipeline.py
```

This will:
1. Process simulated clinical notes through the NLP pipeline
2. Extract rare disease entities, gene names, and treatments
3. Map entities to ICD-10 and SNOMED-CT codes
4. Build and visualize a knowledge graph (saved as `graph.png`)
5. Print a structured summary report

### Run tests

```bash
pytest tests/test_pipeline.py -v
```

## Sample Output

The pipeline processes clinical notes like:

> "Patient presents with hypermobile joints, frequent dislocations, and skin hyperextensibility. Family history of Ehlers-Danlos Syndrome. Genetic testing revealed COL5A1 mutation..."

And produces structured output including:
- Extracted entities with categories (DISEASE, GENE, TREATMENT)
- Ontology codes (e.g., ICD-10: Q79.6, SNOMED-CT: 398114001)
- A knowledge graph visualization linking patients, diseases, genes, and treatments

## Technologies

- **spaCy** - Industrial-strength NLP library
- **pandas** - Data manipulation and structured output
- **NetworkX** - Knowledge graph construction
- **matplotlib** - Graph visualization

## Relevance to Pangaea Data

This POC demonstrates core capabilities aligned with Pangaea Data's mission:
- Extracting clinical insights from unstructured EHR text
- Identifying potential rare disease patients who may be undiagnosed or miscoded
- Mapping clinical findings to standardized ontologies for downstream analysis
- Building knowledge representations that link patients, diseases, and genomic data

## License

This is a proof-of-concept project for demonstration purposes.
