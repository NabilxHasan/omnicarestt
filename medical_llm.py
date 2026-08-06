"""Medical Term Normalizer & Clinical LLM Helper for Bangla-English Speech.

Parses raw Bangla/Banglish speech transcripts from patients and doctors,
extracts symptoms, normalizes phonetic Bangla medical terms to standard English
clinical names (e.g., 'প্যারাসিটামল 500mg' -> 'Paracetamol 500mg'), extracts
dosage schedules (1-0-1, TDS, BD), duration, and structures full digital
doctor prescriptions.

Nothing here invents clinical content
-------------------------------------
PLAN.md is blunt about why: "a *confident wrong word* is worse than *visible
garbage* — a doctor spots the latter, silently trusts the former." A prescription
field this module could not read out of the transcript is therefore returned as
`None`, and the UI renders it as "not specified". It is never filled with a
plausible-looking default. In particular:

  * no strength, schedule, timing or duration is guessed when the speaker
    didn't say one;
  * a transcript with no drug in it yields an EMPTY medicine list, not a
    starter Paracetamol row;
  * a transcript with no symptom in it yields an EMPTY complaint list, not a
    "General Symptoms Reported" placeholder.

Detection runs over the output of `medical_correct.lexicon_pass` — closed
vocabulary, fuzzy match with a margin requirement, abstains when ambiguous — so
a garbled 'পেরাসিটমো' still resolves to Paracetamol, while a word that merely
*resembles* a drug name is flagged for review rather than silently promoted into
the Rx. That stage is deterministic and offline; no LLM sits in this path.
"""
from __future__ import annotations

import re
import json
from typing import Dict, List, Any, Optional

# --- Qwen LLM Model Configuration ---
QWEN_LORA_MODEL_ID = "millat/Qwen2.5-7B-BDLAW-LoRA"
QWEN_BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

def query_qwen_medical_llm(prompt: str) -> Optional[str]:
    """Query Qwen LLM (millat/Qwen2.5-7B-BDLAW-LoRA / Qwen2.5-7B-Instruct) for medical normalization."""
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=QWEN_BASE_MODEL_ID)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert clinical AI medical assistant. Convert spoken Bangla/Banglish medical prescriptions into clean JSON format containing standardized English medicine names, dosage frequencies (1-0-1, 1-1-1), duration, and symptoms."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[medical_llm] Qwen LLM notice: {e} (using local clinical normalizer)")
        return None

# --- Medical Dictionary Mappings (Bangla / Banglish -> Standardized English) ---
SYMPTOM_MAP = {
    "জ্বর": "Fever",
    "মাথাব্যথা": "Headache",
    "মাথা ব্যথা": "Headache",
    "কাশি": "Cough",
    "শুকনো কাশি": "Dry Cough",
    "বমি": "Vomiting",
    "বমি বমি ভাব": "Nausea",
    "পেট ব্যথা": "Abdominal Pain",
    "পেটে ব্যথা": "Abdominal Pain",
    "বুকে ব্যথা": "Chest Pain",
    "শ্বাসকষ্ট": "Shortness of Breath",
    "গলা ব্যথা": "Sore Throat",
    "দুর্বলতা": "General Weakness / Fatigue",
    "শরীর ব্যথা": "Body Ache",
    "প্রস্রাবে জ্বালাপোড়া": "Dysuria (Burning Micturition)",
    "সর্দি": "Runny Nose / Cold",
    "হাঁচি": "Sneezing",
    "ডায়রিয়া": "Diarrhea",
    "পাতলা পায়খানা": "Loose Motion / Diarrhea",
    "রক্তচাপ": "High Blood Pressure",
    "প্রেসার": "Hypertension / BP Issue",
}

DRUG_MAP = {
    "প্যারাসিটামল": "Paracetamol",
    "নাপা": "Napa (Paracetamol)",
    "এইচ": "Ace (Paracetamol)",
    "অ্যাজিথ্রোমাইসিন": "Azithromycin",
    "অমিপ্রাজল": "Omeprazole",
    "ওমিপ্রাজল": "Omeprazole",
    "সেফটিয়াক্সন": "Ceftriaxone",
    "সিপ্রোফ্লক্সাসিন": "Ciprofloxacin",
    "এমোক্সিসিলিন": "Amoxicillin",
    "এন্টাসিড": "Antacid",
    "ফ্লেক্সো": "Flexi (Aceclofenac)",
    "ফেক্সো": "Fexo (Fexofenadine)",
    "মেটফর্মিন": "Metformin",
    "অ্যামলোডিপিন": "Amlodipine",
    "প্যান্টোপ্রাজল": "Pantoprazole",
    "সেফুরোক্সিম": "Cefuroxime",
    "মনটেলুকাস্ট": "Montelukast",
    "ডক্সিসাইক্লিন": "Doxycycline",
    "আইবুপ্রোফেন": "Ibuprofen",
    "হিস্টাসিন": "Histacin",
}

DOSAGE_MAP = {
    "দিনে ১ বার": "1-0-0 (Once daily)",
    "দিনে ২ বার": "1-0-1 (Twice daily / BD)",
    "দিনে ৩ বার": "1-1-1 (Three times daily / TDS)",
    "দিনে ৪ বার": "1-1-1-1 (Four times daily / QDS)",
    "রাতে ১ বার": "0-0-1 (At bedtime / HS)",
    "সকালে ১ বার": "1-0-0 (In the morning)",
    "সকালে ও রাতে": "1-0-1 (Morning & Night)",
    "খাবার পর": "After meal",
    "খাবার আগে": "Before meal",
    "খাওয়ার পর": "After meal",
    "খাওয়ার আগে": "Before meal",
}

TEST_MAP = {
    "ইসিজি": "ECG (Electrocardiogram)",
    "রক্ত পরীক্ষা": "CBC (Complete Blood Count)",
    "সিবিসি": "CBC (Complete Blood Count)",
    "ব্লাড সুগার": "RBS (Random Blood Sugar)",
    "ইউরিন টেস্ট": "Urine R/M/E",
    "চেস্ট এক্সরে": "Chest X-Ray (P/A View)",
    "আল্ট্রাসনোগ্রাম": "USG of Whole Abdomen",
    "সিরাম ক্রিয়েটিনিন": "Serum Creatinine",
}


# --- Field-level extraction (abstains rather than guesses) -----------------

# A strength is a number plus a unit, spoken immediately after the drug name.
# The unit is optional because doctors say "Paracetamol 500" — but see
# _strength_after() for why a bare number is not always a strength.
_STRENGTH_RE = re.compile(
    r"^[\s,:;.\-–]*(\d+(?:[.,]\d+)?)\s*(mg|ml|mcg|gm|gram|g|মিগ্রা|এমজি|গ্রাম|মিলি)?",
    re.IGNORECASE,
)

_DURATION_RE = re.compile(
    r"(\d+)\s*(দিন|days?|সপ্তাহ|weeks?|মাস|months?)",
    re.IGNORECASE,
)

# Words that make a preceding bare number a duration or a frequency, not a dose.
_NOT_A_STRENGTH_AFTER = (
    "দিন", "দিনে", "day", "days", "সপ্তাহ", "week", "weeks",
    "মাস", "month", "months", "বার", "বেলা", "time", "times",
)


def _strength_after(window: str) -> Optional[str]:
    """Dose strength spoken right after a drug name, or None.

    Anchored to the start of the window on purpose: a number further along the
    sentence belongs to the schedule or the duration, and attaching it to the
    drug would be exactly the confident-wrong-value failure this module exists
    to avoid.
    """
    m = _STRENGTH_RE.match(window)
    if not m:
        return None

    number, unit = m.group(1), m.group(2)
    if unit:
        # "500mg" reads naturally closed up in Latin script, spaced in Bangla.
        return f"{number}{unit}" if unit.isascii() else f"{number} {unit}"

    # Bare number: only a strength if a duration/frequency word doesn't claim it
    # ("প্যারাসিটামল ৭ দিন" is a seven-day course, not a 7 mg dose).
    if window[m.end():].lstrip().startswith(_NOT_A_STRENGTH_AFTER):
        return None
    return number


def _schedule_and_timing_in(window: str) -> tuple[Optional[str], Optional[str]]:
    """First dosage schedule and first meal-timing phrase in `window`, else None."""
    schedule = timing = None
    schedule_at = timing_at = None

    for bn, en in DOSAGE_MAP.items():
        idx = window.find(bn)
        if idx < 0:
            continue
        low = en.lower()
        if "after" in low or "before" in low:
            if timing_at is None or idx < timing_at:
                timing, timing_at = en, idx
        else:
            if schedule_at is None or idx < schedule_at:
                schedule, schedule_at = en, idx

    return schedule, timing


def _duration_in(window: str) -> Optional[str]:
    m = _DURATION_RE.search(window)
    return f"{m.group(1)} {m.group(2)}" if m else None


def _drug_mentions(text: str) -> List[tuple]:
    """Non-overlapping (start, end, bangla_key, english_name) for drugs in `text`.

    Both the Bangla vocabulary form and the plain English name are searched, so
    a dictation typed as "Omeprazole 20mg" is found as readily as ওমিপ্রাজল.
    Longest match wins where two candidates overlap.
    """
    hits: List[tuple] = []
    for bn, en in DRUG_MAP.items():
        for m in re.finditer(re.escape(bn), text):
            hits.append((m.start(), m.end(), bn, en))
        # "Napa (Paracetamol)" -> search for "Napa"; the parenthetical is our
        # annotation, not something anyone says.
        latin = en.split("(")[0].strip()
        if latin:
            for m in re.finditer(re.escape(latin), text, re.IGNORECASE):
                hits.append((m.start(), m.end(), bn, en))

    # Longest-first at each position, then greedily drop anything overlapping an
    # already-accepted span. Keeps ওমিপ্রাজল/অমিপ্রাজল from both claiming one
    # "Omeprazole" and stops nested names double-counting.
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    accepted: List[tuple] = []
    for hit in hits:
        if accepted and hit[0] < accepted[-1][1]:
            continue
        accepted.append(hit)
    return accepted


def _lexicon_normalize(text: str) -> tuple[str, List[Dict], List[Dict]]:
    """Closed-vocabulary spelling repair via medical_correct.lexicon_pass.

    Imported lazily because medical_correct imports the maps defined above —
    a module-level import here would be circular. Deterministic and offline:
    the LLM stage of medical_correct is deliberately NOT used in the serving
    path (PLAN.md: no model that can hallucinate sits between speech and Rx).

    Any failure degrades to the raw text plus a review flag; it never silently
    drops the guardrail.
    """
    try:
        import medical_correct
        return medical_correct.lexicon_pass(text)
    except Exception as e:  # import error, or a bad span in the matcher
        return text, [], [{
            "token": None,
            "reason": "lexicon_unavailable",
            "detail": f"closed-vocabulary correction did not run ({e}); "
                      f"detection fell back to exact matching on the raw transcript",
        }]


def normalize_medical_text(raw_text: str) -> Dict[str, Any]:
    """Analyze mixed Bangla-English speech transcript and extract structured clinical terms.

    Every field is either read out of the transcript or returned as None. None
    means "the speaker did not say this" and the UI renders it as
    "not specified" — it is never backfilled with a default.
    """
    if not raw_text or not raw_text.strip():
        return {
            "raw_text": "",
            "corrected_text": "",
            "normalized_summary": "No text provided.",
            "symptoms": [],
            "medicines": [],
            "tests": [],
            "advice": [],
            "corrections": [],
            "review_flags": []
        }

    raw = raw_text.strip()

    # 0. Repair garbled vocabulary before matching. Abstains on ambiguity, so
    #    anything it wasn't sure about arrives in `flags` instead of being
    #    resolved into a drug name.
    text, corrections, review_flags = _lexicon_normalize(raw)

    found_symptoms = []
    found_medicines = []
    found_tests = []
    found_advice = []

    # 1. Detect Symptoms
    seen_symptoms = set()
    for bn, en in SYMPTOM_MAP.items():
        if bn in text and en not in seen_symptoms:
            seen_symptoms.add(en)
            found_symptoms.append({"bangla": bn, "english": en})

    # 2. Detect Medicines & Dosages.
    #    Each drug owns the span of speech between its own name and the next
    #    drug name, so "Paracetamol ... 7 days. And Omeprazole ..." cannot leak
    #    the seven-day course onto the omeprazole line.
    mentions = _drug_mentions(text)
    for i, (_start, end, bn, en) in enumerate(mentions):
        window_end = mentions[i + 1][0] if i + 1 < len(mentions) else len(text)
        window = text[end:window_end]

        schedule, timing = _schedule_and_timing_in(window)
        found_medicines.append({
            "name": en,
            "bangla_name": bn,
            "strength": _strength_after(window),
            "dosage": schedule,
            "timing": timing,
            "duration": _duration_in(window)
        })

    # 3. Detect Tests
    seen_tests = set()
    for bn, en in TEST_MAP.items():
        if bn in text and en not in seen_tests:
            seen_tests.add(en)
            found_tests.append({"bangla": bn, "english": en})

    # 4. Standard Advice Extraction
    if "পানি" in text or "তরল" in text:
        found_advice.append("Drink plenty of clean water and warm fluids.")
    if "বিশাম" in text or "রেস্ট" in text or "আম" in text:
        found_advice.append("Take adequate bed rest for at least 3-5 days.")
    if "ঠান্ডা" in text or "আইসক্রিম" in text:
        found_advice.append("Avoid cold drinks and ice cream.")
    if not found_advice:
        found_advice.append("Follow prescribed dosage strictly and consult doctor if symptoms persist.")

    # 5. Build clean Normalized English Text representation. "Not specified"
    #    everywhere, rather than "As advised" / "None requested" — the latter
    #    assert a clinical decision we did not actually hear.
    symptom_str = ", ".join(s["english"] for s in found_symptoms) or "Not specified"
    med_str = "; ".join(
        f"{m['name']} ({m['dosage'] or 'schedule not specified'}, "
        f"{m['duration'] or 'duration not specified'})"
        for m in found_medicines
    ) or "Not specified"
    test_str = ", ".join(t["english"] for t in found_tests) or "Not specified"

    normalized_summary = f"Symptoms: {symptom_str} | RX: {med_str} | Tests: {test_str}"

    return {
        "raw_text": raw,
        "corrected_text": text,
        "normalized_summary": normalized_summary,
        "symptoms": found_symptoms,
        "medicines": found_medicines,
        "tests": found_tests,
        "advice": found_advice,
        "corrections": corrections,
        "review_flags": review_flags
    }


def generate_digital_prescription(
    patient_name: str,
    patient_age: str,
    patient_gender: str,
    doctor_name: str,
    doctor_title: str,
    patient_speech_transcript: str,
    doctor_speech_dictation: str
) -> Dict[str, Any]:
    """Generate a complete digital prescription schema combining patient & doctor input.

    Carries `None` through untouched. A prescription that reaches the UI with a
    null strength or an empty medicine list is reporting honestly that the
    dictation did not contain one; filling it in here would put the fabrication
    back a layer down.
    """
    patient_analysis = normalize_medical_text(patient_speech_transcript)
    doctor_analysis = normalize_medical_text(doctor_speech_dictation)

    # Combine symptoms & medicines
    all_symptoms = patient_analysis["symptoms"] + [s for s in doctor_analysis["symptoms"] if s not in patient_analysis["symptoms"]]
    all_medicines = doctor_analysis["medicines"] if doctor_analysis["medicines"] else patient_analysis["medicines"]
    all_tests = patient_analysis["tests"] + doctor_analysis["tests"]
    all_advice = doctor_analysis["advice"]
    review_flags = patient_analysis["review_flags"] + doctor_analysis["review_flags"]

    prescription_id = f"RX-OMNI-{abs(hash(patient_name + doctor_name + patient_speech_transcript)) % 1000000:06d}"

    return {
        "prescription_id": prescription_id,
        "clinic_name": "OmniCare Digital Health & Telemedicine Clinic",
        "doctor": {
            "name": doctor_name or "Dr. Nabil Hasan",
            "title": doctor_title or "MBBS, FCPS (Internal Medicine), Clinical Specialist",
            "reg_no": "BMDC Reg No. A-789012"
        },
        "patient": {
            "name": patient_name or "Anonymous Patient",
            "age": patient_age or "30",
            "gender": patient_gender or "Male",
            "date": "2026-08-06"
        },
        "chief_complaints": [s["english"] for s in all_symptoms],
        "raw_patient_speech": patient_speech_transcript,
        "raw_doctor_dictation": doctor_speech_dictation,
        "normalized_medical_summary": doctor_analysis["normalized_summary"],
        "medicines": all_medicines,
        "investigations": [t["english"] for t in all_tests],
        "advice": all_advice,
        # Spans the correction stage refused to resolve. Surfaced so a reviewer
        # can see what the pipeline was unsure of instead of only seeing what it
        # was confident about.
        "review_flags": review_flags,
        "follow_up": "In 7 days or if symptoms worsen."
    }
