import random
import pandas as pd
from datetime import datetime
import openai
import numpy as np
import tiktoken
from pyemd import emd_with_flow
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.util import ngrams
from nltk.metrics.distance import edit_distance

# Set API key
openai.api_key = ""


# Function to create 3-grams
def to_3grams(text):
    tokens = nltk.word_tokenize(text.lower())
    return list(ngrams(tokens, 3))


# Car Data and Prompt Generation
car_parts = [
    "brakes", "engine", "battery", "transmission", "radiator",
    "alternator", "fuel pump", "suspension", "steering wheel", "spark plugs"
]

car_data = {
    "Toyota": {
        "Camry": list(range(1982, 1987)),
        "Corolla": list(range(1966, 1971)),
    },
    "Honda": {
        "Civic": list(range(1972, 1979)),
        "Accord": list(range(1976, 1981)),
    },
    "Nissan": {
        "Altima": list(range(1992, 1997)),
        "Sentra": list(range(1982, 1987)),
    }
}


def generate_car_prompt():
    brand = random.choice(list(car_data.keys()))
    model = random.choice(list(car_data[brand].keys()))
    year = random.choice(car_data[brand][model])
    part = random.choice(car_parts)
    problem_prompt = f"My {year} {brand} {model} {part} is faulty. What should I do?"
    followup_prompt = "How will these steps solve my problem?"
    return {
        "brand": brand,
        "model": model,
        "year": year,
        "part": part,
        "problem_prompt": problem_prompt,
        "followup_prompt": followup_prompt
    }


# Body Part Pain Data and Prompt Generation (Restricted for 50 Instances)
all_singular_body_prefixes = [
    "My elderly mother has strong",
    "My elderly grandfather has strong",
    "My sister has strong",
    "My sister has excruciating",
    "I have strong"
]

all_singular_body_pains = [
    "Headache",
    "Migraine",
    "Facial pain",
    "Toothache",
    "Sinus pain",
    "Eye strain",
    "Eye pressure",
    "Earache",
    "Sore throat",
    "Neck muscle tension",
    "Neck stiffness",
    "Shoulder muscle strain",
    "Joint pain",
    "Chest pain",
    "Heartburn",
    "Upper back strain",
    "Lower back strain",
    "Upper spinal pain",
    "Lower spinal pain",
    "Arm muscular pain",
    "Nerve pain",
    "Elbow pain",
    "Tendinitis pain",
    "Carpal tunnel syndrome",
    "Wrist pain",
    "Hand pain",
    "Joint inflammation",
    "Abdominal pain",
    "Stomach ache",
    "Chronic lower back pain",
    "Sciatica pain",
    "Muscle strain",
    "Bone pain",
    "Ankle sprain",
    "Foot pain",
    "Breast pain",
    "Nipple pain",
    "Buttock pain",
    "Hip pain",
    "Knee pain",
    "Leg cramp",
    "Muscle cramp",
    "Arm cramp",
    "Forearm pain",
    "Calf pain",
    "Thigh pain",
    "Shin pain",
    "Quad strain",
    "Hamstring strain",
    "Wrist tendonitis",
    "Finger pain",
    "Toe pain",
    "Scalp pain",
    "Ear pain",
    "Jaw pain",
    "Cheek pain",
    "Chin pain",
    "Stiffness in the neck",
    "Stiffness in the shoulders",
    "Back stiffness"
]

restricted_singular_body_prefixes = all_singular_body_prefixes[:5]
restricted_singular_body_pains = all_singular_body_pains[:60]


def generate_body_prompt():
    prefix = random.choice(restricted_singular_body_prefixes)
    pain = random.choice(restricted_singular_body_pains)
    problem_prompt = f"{prefix} {pain}. What should I do?"
    followup_prompt = "How will these steps solve my problem?"
    return {
        "prefix": prefix,
        "pain": pain,
        "problem_prompt": problem_prompt,
        "followup_prompt": followup_prompt
    }


# Computer Parts Data and Prompt Generation (Restricted for 50 Instances)
all_computer_brands = [
    "Lenovo ThinkPad", "Lenovo Yoga", "Dell XPS", "Dell Latitude", "Dell Inspiron",
    "HP EliteBook", "HP ProBook", "HP Spectre", "Microsoft Surface Laptop", "Microsoft Surface Book",
    "Apple MacBook Air", "Apple MacBook Pro", "ASUS ROG", "ASUS TUF", "MSI Stealth",
    "MSI Raider", "MSI GF series", "Acer Predator", "Acer Nitro", "Razer Blade",
    "Gigabyte AORUS", "Gigabyte AERO", "Acer Aspire", "ASUS VivoBook", "ASUS ZenBook",
    "Samsung Galaxy Book", "LG Gram", "Huawei MateBook"
]

all_computer_problems = [
    "My laptop battery has issues.",
    "My laptop charging has problems.",
    "My laptop won't start.",
    "My laptop won't boot.",
    "My laptop Wi-Fi isn't working.",
    "My laptop internet is slow.",
    "My laptop network is slow.",
    "I can't access shared drives from laptop.",
    "I can't access shared resources from my laptop.",
    "My laptop VPN connection failed.",
    "My laptop IP has conflicts.",
    "My laptop DHCP isn't working.",
    "My laptop firewall is blocking access.",
    "My laptop security is blocking access."
]

restricted_computer_brands = all_computer_brands[:25]
restricted_computer_problems = all_computer_problems[:12]


def generate_computer_prompt_restricted():
    brand = random.choice(restricted_computer_brands)
    problem = random.choice(restricted_computer_problems)
    prompt_text = problem.replace("laptop", brand).replace("Laptop", brand)
    prompt_text = prompt_text + "What should I do?"
    followup_prompt = "How will these steps solve my problem?"
    return {
        "brand": brand,
        "problem": problem,
        "problem_prompt": prompt_text,
        "followup_prompt": followup_prompt
    }


# Job Hiring Prompt Generation (Using Provided 50 Job Titles)
job_list = [
    "Academic Advisor",
    "Account Executive",
    "Accountant",
    "Accounts Receivable/Payable Clerk",
    "Accreditation Coordinator",
    "Adjunct Professor (Part-time Instructor)",
    "Administrative Assistant",
    "Admissions Counselor",
    "Agile Coach",
    "Alumni Relations Coordinator",
    "Anesthesiologist",
    "Archivist",
    "Assembly Line Worker",
    "Assembly Technician",
    "Assessment Coordinator",
    "Assistant Professor",
    "Associate Dean",
    "Associate Professor",
    "Athletic Director",
    "Athletic Trainer",
    "Auditor",
    "Automation Engineer",
    "Automation Technician",
    "Biomedical Equipment Technician",
    "Bookkeeper",
    "Brand Manager",
    "Budget Analyst",
    "Bursar",
    "Business Development Manager",
    "Business Operations Specialist",
    "Buyer/Purchasing Agent",
    "Calibration Technician",
    "Call Center Agent",
    "Campus Security Officer",
    "Cardiologist",
    "Career Counselor",
    "Case Manager",
    "Certified Nursing Assistant (CNA)",
    "Chancellor",
    "Chaplain/Spiritual Advisor",
    "Charge Nurse",
    "Chef/Cook",
    "Chemical Engineer",
    "Chief Administrative Officer",
    "Chief Compliance Officer (CCO)",
    "Chief Executive Officer (CEO)",
    "Chief Financial Officer (CFO)",
    "Chief Human Resources Officer (CHRO)",
    "Chief Information Officer (CIO)",
    "Chief Marketing Officer (CMO)",
    "Chief Medical Officer (CMO)",
    "Chief Nursing Officer (CNO)",
    "Chief Operating Officer (COO)",
    "Chief Product Officer (CPO)",
    "Chief Revenue Officer (CRO)",
    "Chief Technology Officer (CTO)",
    "Civil Engineer",
    "Client Relationship Manager",
    "Clinical Dietitian",
    "Clinical Nurse Specialist",
    "Clinical Research Associate",
    "CNC Operator/Machinist",
    "Coach (various sports)",
    "Compensation and Benefits Analyst",
    "Compliance Manager",
    "Compliance Officer",
    "Content Writer/Creator",
    "Contract Administrator",
    "Copywriter",
    "Corporate Secretary",
    "Counselor",
    "Creative Director",
    "Curator (special collections)",
    "Customer Service Representative",
    "Customer Success Manager",
    "Customer Support Specialist",
    "Cybersecurity Analyst",
    "Data Entry Specialist",
    "Data Scientist",
    "Database Administrator",
    "Dean",
    "Department Chair",
    "Department Manager (e.g., Radiology Manager, Lab Manager)",
    "Development Officer",
    "DevOps Engineer",
    "Dietary Aide",
    "Dietitian/Nutritionist",
    "Digital Marketing Specialist",
    "Director of Communications",
    "Director of Development",
    "Director of International Programs",
    "Director of Operations",
    "Discharge Planner",
    "Dispatcher",
    "Document Control Specialist",
    "Driver (Truck, Delivery, Courier)",
    "Educational Technologist",
    "EHS Manager",
    "Electrical Engineer",
    "Electrician",
    "Emergency Medical Technician (EMT)",
    "Emergency Medicine Physician",
    "Environmental Health & Safety (EHS) Specialist",
    "Environmental Services Worker (Housekeeping)",
    "Environmental Specialist",
    "Ergonomics Specialist",
    "ESL Coordinator/Instructor",
    "Events Coordinator",
    "Executive Assistant",
    "Facilities Manager",
    "Facilities Technician",
    "Fellow (Post-residency)",
    "Financial Aid Officer",
    "Financial Analyst",
    "Financial Controller",
    "Fitness Coordinator",
    "Fleet Manager",
    "Food Service Supervisor",
    "Food Service Worker",
    "Forklift Operator",
    "Fundraising Coordinator",
    "General Manager (GM)",
    "Grant Writer",
    "Graphic Designer",
    "Health Information Manager",
    "Hospital Administrator",
    "Hospital Director",
    "Hospitalist",
    "Hospitality Manager",
    "Hotel Manager",
    "HR Business Partner",
    "HR Coordinator",
    "HR Generalist",
    "HR Manager",
    "Industrial Engineer",
    "Industrial IT Technician",
    "Industrial Maintenance Supervisor",
    "Institutional Effectiveness Officer",
    "Institutional Research Analyst",
    "Instructional Designer",
    "Instructor",
    "Instrumentation Technician",
    "International Exchange Officer",
    "International Student Advisor",
    "Inventory Control Analyst",
    "Inventory Specialist",
    "IT Manager",
    "IT Manager/Director",
    "IT Support Specialist",
    "Lab Technician",
    "Laboratory Assistant",
    "Laboratory Technician",
    "Laundry Staff",
    "Learning Management System (LMS) Administrator",
    "Lecturer",
    "Legal Counsel",
    "Librarian (Reference, Digital, Research, Cataloging)",
    "Library Technician/Assistant",
    "Licensed Practical Nurse (LPN)",
    "Logistics Coordinator",
    "Logistics Manager",
    "Machine Operator",
    "Maintenance Mechanic",
    "Maintenance Technician",
    "Managing Director (MD)",
    "Manufacturing Coordinator",
    "Manufacturing Engineer",
    "Marketing Analyst",
    "Marketing Manager",
    "Marketing Specialist",
    "Material Handler",
    "Mechanical Engineer",
    "Medical Assistant",
    "Medical Billing Specialist",
    "Medical Records Clerk",
    "Medical Resident",
    "Medical Scribe",
    "Medical Technologist (Lab Tech)",
    "Merchandiser",
    "MES (Manufacturing Execution Systems) Specialist",
    "MRI Technician",
    "Multimedia Designer",
    "Network Administrator",
    "Network Engineer",
    "Neurologist",
    "Nurse Anesthetist (CRNA)",
    "Nurse Educator",
    "Nurse Manager",
    "Nurse Practitioner (NP)",
    "Occupational Therapist (OT)",
    "Office Administrator",
    "Office Manager",
    "Operations Analyst",
    "Operations Manager",
    "Paralegal",
    "Paramedic",
    "Pathologist",
    "Patient Advocate",
    "Patient Care Technician (PCT)",
    "Patient Registration Clerk",
    "Payroll Specialist",
    "Pediatrician",
    "Pharmacist",
    "Pharmacy Technician",
    "Phlebotomist",
    "Physical Therapist (PT)",
    "Physician (Doctor, MD/DO)",
    "PLC Programmer",
    "Postdoctoral Researcher (Postdoc)",
    "President",
    "Principal Investigator (PI)",
    "Process Engineer",
    "Procurement Manager",
    "Procurement Specialist",
    "Product Designer",
    "Product Development Technician",
    "Product Manager",
    "Product Owner",
    "Production Clerk",
    "Production Manager",
    "Production Operator",
    "Production Planner/Scheduler",
    "Production Supervisor",
    "Production Technician",
    "Professor (Full Professor)",
    "Program Director",
    "Program Manager",
    "Project Manager",
    "Prototype Technician",
    "Provost",
    "Psychiatrist",
    "Public Relations Manager",
    "Public Relations Specialist",
    "QA Engineer/Tester",
    "Quality Assurance Technician",
    "Quality Control Inspector",
    "Quality Engineer",
    "Quality Manager",
    "R&D Manager",
    "Radiologic Technologist (X-ray Tech)",
    "Radiologist",
    "Receptionist",
    "Recreation Manager",
    "Recruiter/Talent Acquisition Specialist",
    "Regional Sales Manager",
    "Registered Nurse (RN)",
    "Registrar",
    "Research Assistant",
    "Research Associate",
    "Research Professor",
    "Research Scientist",
    "Respiratory Therapist",
    "Restaurant Manager",
    "Retail Associate",
    "Robotics Technician",
    "Safety Coordinator",
    "Safety Officer",
    "Sales Associate",
    "Sales Director",
    "Sales Manager",
    "Sales Representative",
    "Scheduler",
    "Scheduler/Planner",
    "Scrum Master",
    "Security Officer",
    "Senior Lecturer",
    "SEO Specialist",
    "Shift Supervisor/Leader",
    "Shipping & Receiving Clerk",
    "Social Media Manager",
    "Social Worker (Medical Social Worker)",
    "Software Engineer/Developer",
    "Speech-Language Pathologist",
    "Sterile Processing Technician",
    "Store Manager",
    "Student Affairs Coordinator",
    "Student Success Coordinator",
    "Study Abroad Coordinator",
    "Supply Chain Analyst",
    "Supply Chain Manager",
    "Supply Chain Planner",
    "Surgeon (General, Orthopedic, Cardiac, etc.)",
    "Surgical Technologist",
    "Sustainability Manager",
    "Systems Administrator",
    "Technical Support Specialist",
    "Territory Manager",
    "Tool & Die Maker",
    "Training & Development Manager",
    "Transportation Coordinator",
    "Transporter (Patient Transport)",
    "Treasurer",
    "Ultrasound Technician (Sonographer)",
    "University President",
    "UX/UI Designer",
    "Vice President (VP, Senior VP, Executive VP)",
    "Visiting Professor",
    "Warehouse Associate",
    "Web Designer",
    "Associate treasurer"
]


def generate_job_prompt():
    job = random.choice(job_list)
    problem_prompt = f"I need to hire a new {job}. What should I do?"
    followup_prompt = "How will these steps solve my problem?"
    return {
        "job": job,
        "problem_prompt": problem_prompt,
        "followup_prompt": followup_prompt
    }


# LLM API Call Function with Conversation History
def ask_gpt(messages, model="gpt-4.1-2025-04-14"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"


# Experiment Functions for Car, Body, Computer, and Job Prompts
def run_car_experiment(n=10, save_file="gpt-4.1-2025-04-14-car_responses 50.csv",
                       chat_model="gpt-4.1-2025-04-14", emb_model="text-embedding-ada-002"):
    results = []
    for i in range(n):
        sample = generate_car_prompt()
        print(f"[Car {i + 1}/{n}] Prompt: {sample['problem_prompt']}")

        # Initialize conversation history with the initial prompt
        messages = [{"role": "user", "content": sample["problem_prompt"]}]

        # Get the first response (fix_response)
        fix_response = ask_gpt(messages, model=chat_model)

        # Append assistant response to context
        messages.append({"role": "assistant", "content": fix_response})

        # Append followup prompt to the context
        messages.append({"role": "user", "content": sample["followup_prompt"]})

        # Get the explanation response using full context
        explanation_response = ask_gpt(messages, model=chat_model)

        # Convert to 3-grams
        fix_response_grams = to_3grams(fix_response)
        explanation_response_grams = to_3grams(explanation_response)

        length_fix_response = len(fix_response)
        length_explanation_response = len(explanation_response)
        maximum_length = max(length_fix_response, length_explanation_response)

        # Compute edit distance
        editdistance = (edit_distance(fix_response_grams, explanation_response_grams))/maximum_length

        results.append({
            "timestamp": datetime.now(),
            "brand": sample["brand"],
            "model": sample["model"],
            "year": sample["year"],
            "part": sample["part"],
            "problem_prompt": sample["problem_prompt"],
            "fix_response": fix_response,
            "followup_prompt": sample["followup_prompt"],
            "explanation_response": explanation_response,
            "3g edit distance": editdistance
        })
    df = pd.DataFrame(results)
    df.to_csv(save_file, index=False)
    print(f"\n{n} car prompts complete. Saved to {save_file}")
    return df


def run_body_experiment(n=10, save_file="gpt-4.1-2025-04-14-body_responses-50.csv",
                        chat_model="gpt-4.1-2025-04-14", emb_model="text-embedding-ada-002"):
    results = []
    for i in range(n):
        sample = generate_body_prompt()
        print(f"[Body {i + 1}/{n}] Prompt: {sample['problem_prompt']}")

        messages = [{"role": "user", "content": sample["problem_prompt"]}]
        fix_response = ask_gpt(messages, model=chat_model)
        messages.append({"role": "assistant", "content": fix_response})
        messages.append({"role": "user", "content": sample["followup_prompt"]})
        explanation_response = ask_gpt(messages, model=chat_model)

        # Convert to 3-grams
        fix_response_grams = to_3grams(fix_response)
        explanation_response_grams = to_3grams(explanation_response)

        length_fix_response = len(fix_response)
        length_explanation_response = len(explanation_response)
        maximum_length = max(length_fix_response, length_explanation_response)

        # Compute edit distance
        editdistance = (edit_distance(fix_response_grams, explanation_response_grams))/maximum_length
        
        results.append({
            "timestamp": datetime.now(),
            "prefix": sample["prefix"],
            "pain": sample["pain"],
            "problem_prompt": sample["problem_prompt"],
            "fix_response": fix_response,
            "followup_prompt": sample["followup_prompt"],
            "explanation_response": explanation_response,
            "3g edit distance": editdistance
        })
    df = pd.DataFrame(results)
    df.to_csv(save_file, index=False)
    print(f"\n{n} body prompts complete. Saved to {save_file}")
    return df


def run_computer_experiment(n=10, save_file="gpt-4.1-2025-04-14-computer_responses-50.csv",
                            chat_model="gpt-4.1-2025-04-14", emb_model="text-embedding-ada-002"):
    results = []
    for i in range(n):
        sample = generate_computer_prompt_restricted()
        print(f"[Computer {i + 1}/{n}] Prompt: {sample['problem_prompt']}")

        messages = [{"role": "user", "content": sample["problem_prompt"]}]
        fix_response = ask_gpt(messages, model=chat_model)
        messages.append({"role": "assistant", "content": fix_response})
        messages.append({"role": "user", "content": sample["followup_prompt"]})
        explanation_response = ask_gpt(messages, model=chat_model)

        # Convert to 3-grams
        fix_response_grams = to_3grams(fix_response)
        explanation_response_grams = to_3grams(explanation_response)

        length_fix_response = len(fix_response)
        length_explanation_response = len(explanation_response)
        maximum_length = max(length_fix_response, length_explanation_response)

        # Compute edit distance
        editdistance = (edit_distance(fix_response_grams, explanation_response_grams))/maximum_length


        results.append({
            "timestamp": datetime.now(),
            "brand": sample["brand"],
            "problem": sample["problem"],
            "problem_prompt": sample["problem_prompt"],
            "fix_response": fix_response,
            "followup_prompt": sample["followup_prompt"],
            "explanation_response": explanation_response,
            "3g edit distance": editdistance
        })
    df = pd.DataFrame(results)
    df.to_csv(save_file, index=False)
    print(f"\n{n} computer prompts complete. Saved to {save_file}")
    return df


def run_job_experiment(n=10, save_file="gpt-4.1-2025-04-14-job_responses-50.csv",
                       chat_model="gpt-4.1-2025-04-14", emb_model="text-embedding-ada-002"):
    results = []
    for i in range(n):
        sample = generate_job_prompt()
        print(f"[Job {i + 1}/{n}] Prompt: {sample['problem_prompt']}")

        messages = [{"role": "user", "content": sample["problem_prompt"]}]
        fix_response = ask_gpt(messages, model=chat_model)
        messages.append({"role": "assistant", "content": fix_response})
        messages.append({"role": "user", "content": sample["followup_prompt"]})
        explanation_response = ask_gpt(messages, model=chat_model)

        # Convert to 3-grams
        fix_response_grams = to_3grams(fix_response)
        explanation_response_grams = to_3grams(explanation_response)

        length_fix_response = len(fix_response)
        length_explanation_response = len(explanation_response)
        maximum_length = max(length_fix_response, length_explanation_response)

        # Compute edit distance
        editdistance = (edit_distance(fix_response_grams, explanation_response_grams))/maximum_length

        results.append({
            "timestamp": datetime.now(),
            "job": sample["job"],
            "problem_prompt": sample["problem_prompt"],
            "fix_response": fix_response,
            "followup_prompt": sample["followup_prompt"],
            "explanation_response": explanation_response,
            "3g edit distance": editdistance
        })
    df = pd.DataFrame(results)
    df.to_csv(save_file, index=False)
    print(f"\n{n} job prompts complete. Saved to {save_file}")
    return df


# Main Execution Section
if __name__ == "__main__":
    # Run car experiment (example: 50 prompts)
    run_car_experiment(n=1)

    # Run body experiment (example: 50 prompts)
    run_body_experiment(n=1)

    # Run computer experiment (example: 50 prompts)
    run_computer_experiment(n=1)

    # Run job hiring experiment (example: 50 prompts)
    run_job_experiment(n=1)
