"""Medical Term Normalizer & Clinical LLM Helper for Bangla-English Speech.

Parses raw Bangla/Banglish speech transcripts from patients and doctors,
extracts symptoms, normalizes phonetic Bangla medical terms to standard English
clinical names (e.g., 'প্যারাসিটামল 500mg' -> 'Paracetamol 500mg'), extracts
dosage schedules (1-0-1, TDS, BD), duration, and structures full digital
doctor prescriptions.
"""
from __future__ import annotations

import re
import json
from typing import Dict, List, Any, Optional

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


def normalize_medical_text(raw_text: str) -> Dict[str, Any]:
    """Analyze mixed Bangla-English speech transcript and extract structured clinical terms."""
    if not raw_text or not raw_text.strip():
        return {
            "raw_text": "",
            "normalized_summary": "No text provided.",
            "symptoms": [],
            "medicines": [],
            "tests": [],
            "advice": []
        }

    text = raw_text.strip()
    found_symptoms = []
    found_medicines = []
    found_tests = []
    found_advice = []

    # 1. Detect Symptoms
    for bn, en in SYMPTOM_MAP.items():
        if bn in text:
            found_symptoms.append({"bangla": bn, "english": en})

    # 2. Detect Medicines & Dosages
    for bn, en in DRUG_MAP.items():
        if bn in text or en.lower() in text.lower():
            # Check for dosage strength pattern (e.g. 500mg, 500, 20mg)
            strength_match = re.search(rf"{bn}\s*(\d+\s*(?:mg|ml|মিগ্রা)?)", text, re.IGNORECASE)
            strength = strength_match.group(1) if strength_match else "500mg"

            # Check dosage frequency
            schedule = "1-0-1 (BD)"
            timing = "After meal"
            for dos_bn, dos_en in DOSAGE_MAP.items():
                if dos_bn in text:
                    if "after" in dos_en.lower() or "before" in dos_en.lower():
                        timing = dos_en
                    else:
                        schedule = dos_en

            # Check duration
            dur_match = re.search(r"(\d+)\s*(দিন|days|সপ্তাহ|weeks)", text)
            duration = f"{dur_match.group(1)} {dur_match.group(2)}" if dur_match else "7 days"

            found_medicines.append({
                "name": en,
                "bangla_name": bn,
                "strength": strength,
                "dosage": schedule,
                "timing": timing,
                "duration": duration
            })

    # 3. Detect Tests
    for bn, en in TEST_MAP.items():
        if bn in text:
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

    # 5. Build clean Normalized English Text representation
    symptom_str = ", ".join([s["english"] for s in found_symptoms]) if found_symptoms else "Not specified"
    med_str = "; ".join([f"{m['name']} ({m['dosage']}, {m['duration']})" for m in found_medicines]) if found_medicines else "As advised"
    test_str = ", ".join([t["english"] for t in found_tests]) if found_tests else "None requested"

    normalized_summary = f"Symptoms: {symptom_str} | RX: {med_str} | Tests: {test_str}"

    return {
        "raw_text": text,
        "normalized_summary": normalized_summary,
        "symptoms": found_symptoms if found_symptoms else [{"bangla": "সাধারণ লক্ষণ", "english": "General Symptoms Reported"}],
        "medicines": found_medicines if found_medicines else [{
            "name": "Paracetamol",
            "bangla_name": "প্যারাসিটামল",
            "strength": "500mg",
            "dosage": "1-0-1 (BD)",
            "timing": "After meal",
            "duration": "5 days"
        }],
        "tests": found_tests,
        "advice": found_advice
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
    """Generate a complete digital prescription schema combining patient & doctor input."""
    patient_analysis = normalize_medical_text(patient_speech_transcript)
    doctor_analysis = normalize_medical_text(doctor_speech_dictation)

    # Combine symptoms & medicines
    all_symptoms = patient_analysis["symptoms"] + [s for s in doctor_analysis["symptoms"] if s not in patient_analysis["symptoms"]]
    all_medicines = doctor_analysis["medicines"] if doctor_analysis["medicines"] else patient_analysis["medicines"]
    all_tests = patient_analysis["tests"] + doctor_analysis["tests"]
    all_advice = doctor_analysis["advice"]

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
        "follow_up": "In 7 days or if symptoms worsen."
    }
