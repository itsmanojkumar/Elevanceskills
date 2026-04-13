"""
Medical Named Entity Recognition (NER).

Hybrid approach:
  1. spaCy base model (en_core_web_sm) for general NER.
  2. Comprehensive medical term dictionaries (diseases, symptoms,
     treatments, medications, anatomy, procedures).
  3. spaCy EntityRuler for pattern-based medical entity detection.
"""

import re
from typing import Optional
import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler

# ---------------------------------------------------------------------------
# Medical Terminology Dictionaries
# ---------------------------------------------------------------------------

DISEASES = [
    # Cardiovascular
    "hypertension", "coronary artery disease", "heart failure", "atrial fibrillation",
    "myocardial infarction", "heart attack", "stroke", "angina", "arrhythmia",
    "cardiomyopathy", "endocarditis", "pericarditis", "aortic aneurysm",
    # Respiratory
    "asthma", "chronic obstructive pulmonary disease", "copd", "pneumonia",
    "tuberculosis", "bronchitis", "emphysema", "pulmonary fibrosis", "lung cancer",
    "sleep apnea", "pleurisy", "pulmonary embolism",
    # Diabetes & Metabolic
    "diabetes", "diabetes mellitus", "type 1 diabetes", "type 2 diabetes",
    "hypoglycemia", "hyperglycemia", "obesity", "metabolic syndrome",
    "hypothyroidism", "hyperthyroidism", "gout",
    # Cancer
    "cancer", "leukemia", "lymphoma", "melanoma", "breast cancer", "prostate cancer",
    "colon cancer", "cervical cancer", "ovarian cancer", "pancreatic cancer",
    "liver cancer", "bladder cancer", "kidney cancer", "thyroid cancer",
    "brain tumor", "glioblastoma",
    # Neurological
    "alzheimer's disease", "parkinson's disease", "multiple sclerosis",
    "epilepsy", "migraine", "dementia", "huntington's disease", "als",
    "amyotrophic lateral sclerosis", "meningitis", "encephalitis",
    "cerebral palsy", "peripheral neuropathy",
    # GI
    "crohn's disease", "ulcerative colitis", "irritable bowel syndrome", "ibs",
    "gastroesophageal reflux disease", "gerd", "peptic ulcer", "celiac disease",
    "cirrhosis", "hepatitis", "hepatitis a", "hepatitis b", "hepatitis c",
    "pancreatitis", "appendicitis", "diverticulitis",
    # Musculoskeletal
    "arthritis", "rheumatoid arthritis", "osteoarthritis", "osteoporosis",
    "fibromyalgia", "lupus", "scoliosis", "carpal tunnel syndrome",
    # Infectious
    "influenza", "flu", "covid-19", "hiv", "aids", "malaria", "dengue",
    "lyme disease", "shingles", "chickenpox", "measles", "mumps", "rubella",
    # Mental Health
    "depression", "anxiety", "schizophrenia", "bipolar disorder", "ptsd",
    "adhd", "autism spectrum disorder", "ocd", "anorexia", "bulimia",
    # Renal
    "chronic kidney disease", "kidney failure", "nephrotic syndrome",
    "urinary tract infection", "uti", "kidney stones",
    # Other
    "anemia", "sickle cell disease", "hemophilia", "psoriasis", "eczema",
    "glaucoma", "cataracts", "macular degeneration",
]

SYMPTOMS = [
    "fever", "cough", "headache", "fatigue", "nausea", "vomiting", "diarrhea",
    "constipation", "abdominal pain", "chest pain", "shortness of breath",
    "dyspnea", "wheezing", "palpitations", "dizziness", "vertigo", "syncope",
    "fainting", "weight loss", "weight gain", "loss of appetite", "anorexia",
    "excessive thirst", "frequent urination", "polyuria", "polydipsia",
    "blurred vision", "double vision", "hearing loss", "tinnitus", "rash",
    "hives", "itching", "swelling", "edema", "joint pain", "muscle pain",
    "myalgia", "back pain", "neck pain", "stiff neck", "tremor", "seizure",
    "confusion", "memory loss", "numbness", "tingling", "weakness",
    "paralysis", "difficulty swallowing", "dysphagia", "hoarseness",
    "runny nose", "nasal congestion", "sore throat", "ear pain", "eye pain",
    "jaundice", "pale skin", "bruising", "bleeding", "blood in urine",
    "blood in stool", "dark urine", "night sweats", "chills", "insomnia",
    "sleep disturbances", "anxiety", "depression", "mood changes",
    "irritability", "hallucinations",
]

TREATMENTS = [
    "chemotherapy", "radiation therapy", "radiotherapy", "immunotherapy",
    "surgery", "surgical resection", "transplant", "dialysis",
    "physical therapy", "occupational therapy", "psychotherapy",
    "cognitive behavioral therapy", "cbt", "electroconvulsive therapy", "ect",
    "blood transfusion", "bone marrow transplant", "stem cell transplant",
    "coronary artery bypass graft", "cabg", "angioplasty", "stenting",
    "pacemaker", "defibrillator", "insulin therapy", "oxygen therapy",
    "ventilation", "intubation", "tracheotomy", "amputation",
    "hysterectomy", "mastectomy", "appendectomy", "cholecystectomy",
    "colonoscopy", "endoscopy", "laparoscopy", "biopsy",
]

MEDICATIONS = [
    # Antibiotics
    "amoxicillin", "penicillin", "azithromycin", "ciprofloxacin", "doxycycline",
    "metronidazole", "vancomycin", "cephalexin", "trimethoprim",
    # Pain / Anti-inflammatory
    "aspirin", "ibuprofen", "acetaminophen", "naproxen", "celecoxib",
    "morphine", "codeine", "oxycodone", "tramadol", "fentanyl",
    "prednisone", "dexamethasone", "methylprednisolone",
    # Cardiovascular
    "metoprolol", "atenolol", "lisinopril", "amlodipine", "losartan",
    "atorvastatin", "simvastatin", "warfarin", "heparin", "clopidogrel",
    "digoxin", "furosemide", "spironolactone", "nitroglycerin",
    # Diabetes
    "metformin", "insulin", "glipizide", "glimepiride", "sitagliptin",
    "liraglutide", "empagliflozin",
    # CNS / Psychiatric
    "sertraline", "fluoxetine", "escitalopram", "paroxetine", "venlafaxine",
    "bupropion", "lithium", "risperidone", "olanzapine", "quetiapine",
    "alprazolam", "diazepam", "lorazepam", "zolpidem", "melatonin",
    # Respiratory
    "albuterol", "salbutamol", "ipratropium", "tiotropium", "fluticasone",
    "budesonide", "montelukast",
    # Other
    "methotrexate", "hydroxychloroquine", "omeprazole", "pantoprazole",
    "ranitidine", "ondansetron", "levothyroxine", "tamoxifen", "amlodipine",
]

ANATOMY = [
    "heart", "lung", "liver", "kidney", "brain", "spine", "stomach",
    "intestine", "colon", "rectum", "esophagus", "pancreas", "spleen",
    "gallbladder", "bladder", "uterus", "ovary", "prostate", "thyroid",
    "adrenal gland", "pituitary", "bone", "muscle", "tendon", "ligament",
    "joint", "artery", "vein", "lymph node", "skin", "nerve",
    "eye", "ear", "nose", "throat", "mouth", "tongue", "teeth",
]

PROCEDURES = [
    "mri", "ct scan", "x-ray", "ultrasound", "ecg", "ekg", "echocardiogram",
    "blood test", "urinalysis", "biopsy", "colonoscopy", "endoscopy",
    "mammography", "pap smear", "bone density scan", "angiography",
    "lumbar puncture", "spinal tap", "electroencephalogram", "eeg",
    "electromyography", "emg", "spirometry", "pulmonary function test",
]

# ---------------------------------------------------------------------------
# Entity label mapping
# ---------------------------------------------------------------------------
ENTITY_LABEL_MAP = {
    "DISEASE": ("DISEASE", DISEASES),
    "SYMPTOM": ("SYMPTOM", SYMPTOMS),
    "TREATMENT": ("TREATMENT", TREATMENTS),
    "MEDICATION": ("MEDICATION", MEDICATIONS),
    "ANATOMY": ("ANATOMY", ANATOMY),
    "PROCEDURE": ("PROCEDURE", PROCEDURES),
}

ENTITY_COLORS = {
    "DISEASE": "#FF6B6B",
    "SYMPTOM": "#FFA94D",
    "TREATMENT": "#69DB7C",
    "MEDICATION": "#74C0FC",
    "ANATOMY": "#DA77F2",
    "PROCEDURE": "#FFD43B",
    "PERSON": "#A9E34B",
    "ORG": "#63E6BE",
    "GPE": "#E599F7",
}


# ---------------------------------------------------------------------------
# NER Pipeline
# ---------------------------------------------------------------------------

class MedicalNER:
    """Hybrid medical entity recognition using spaCy + dictionary patterns."""

    def __init__(self):
        self.nlp: Optional[Language] = None
        self._initialized = False

    def initialize(self) -> None:
        """Load spaCy model and add EntityRuler with medical patterns."""
        if self._initialized:
            return
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback: blank English model
            self.nlp = spacy.blank("en")

        # Add EntityRuler before the NER component (or at end if no NER)
        ruler_name = "medical_ruler"
        if ruler_name not in self.nlp.pipe_names:
            if "ner" in self.nlp.pipe_names:
                ruler = self.nlp.add_pipe("entity_ruler", name=ruler_name, before="ner")
            else:
                ruler = self.nlp.add_pipe("entity_ruler", name=ruler_name)
        else:
            ruler = self.nlp.get_pipe(ruler_name)

        patterns = []
        for label, (_, terms) in ENTITY_LABEL_MAP.items():
            for term in terms:
                # Exact lowercase match
                patterns.append({"label": label, "pattern": term.lower()})
                # Title-case variant
                patterns.append({"label": label, "pattern": term.title()})
                # Handle multi-word with individual token patterns
                if " " in term:
                    tokens = [{"LOWER": t} for t in term.split()]
                    patterns.append({"label": label, "pattern": tokens})

        ruler.add_patterns(patterns)
        self._initialized = True

    def extract_entities(self, text: str) -> list[dict]:
        """
        Extract medical entities from text.

        Returns list of dicts:
            {text, label, start_char, end_char, label_color}
        """
        if not self._initialized:
            self.initialize()

        doc = self.nlp(text)
        entities = []
        seen_spans = set()

        for ent in doc.ents:
            span_key = (ent.start_char, ent.end_char)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "color": ENTITY_COLORS.get(ent.label_, "#ADB5BD"),
                }
            )

        return entities

    def highlight_text(self, text: str) -> str:
        """
        Return HTML string with medical entities highlighted
        using colored <mark> spans.
        """
        if not self._initialized:
            self.initialize()

        entities = self.extract_entities(text)
        # Sort by start position descending so replacement doesn't shift offsets
        entities_sorted = sorted(entities, key=lambda e: e["start_char"], reverse=True)

        result = text
        for ent in entities_sorted:
            s, e = ent["start_char"], ent["end_char"]
            color = ent["color"]
            label = ent["label"]
            original = result[s:e]
            replacement = (
                f'<mark class="ner-mark" style="background-color:{color};border-radius:4px;'
                f'padding:2px 4px;font-weight:600;" '
                f'title="{label}">{original}'
                f'<sup style="font-size:0.65em;margin-left:2px;">{label}</sup></mark>'
            )
            result = result[:s] + replacement + result[e:]

        return result

    def get_entity_summary(self, text: str) -> dict[str, list[str]]:
        """Return a dict grouping entity texts by label."""
        entities = self.extract_entities(text)
        summary: dict[str, list[str]] = {}
        for ent in entities:
            lbl = ent["label"]
            summary.setdefault(lbl, [])
            if ent["text"] not in summary[lbl]:
                summary[lbl].append(ent["text"])
        return summary


# Singleton instance
_ner_instance: Optional[MedicalNER] = None


def get_ner() -> MedicalNER:
    global _ner_instance
    if _ner_instance is None:
        _ner_instance = MedicalNER()
        _ner_instance.initialize()
    return _ner_instance
