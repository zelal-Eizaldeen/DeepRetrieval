# SCREENING_CRITERIA_GENERATION  = '''
# You are a clinical specialist. You are conducting a clinical trial systematic review.
# The research is defined by the following PICO elements:
# P (Patient, Problem or Population): {P}
# I (Intervention): {I}
# C (Comparison): {C}
# O (Outcome): {O}

# ## Task
# Your task is to design the eligibility criteria for selecting studies for this systematic review following these steps:

# ### Step 1
# Based on the PRISMA guidelines and the PICO elements of this research, please identify eligibility criteria for the studies to be included in the systematic review. Provide a rationale for each criterion.

# ELIGIBILITY_ANALYSIS: your items and reasons here...

# ### Step 2
# Next, create {num_eligibility_criteria} binary questions that will help you select studies based on their titles and abstracts. 
# These questions should be designed so that a "YES" answer indicates the study meets the criteria, while a "NO" answer means it doesn't. 
# The information required to answer these questions should be general and easily found in the study title and abstract.


# ## Reply Format
# You should reply in a JSON format like:

# {{
#     "ELIGIBILITY_ANALYSIS": ["rationale1", "rationale12", ...] \\ the bullet points of your analysis
#     "ELIGIBILITY_CRITERIA": ["criterion1", "criterion2", "..."] \\ the {num_eligibility_criteria} binary eligibility criteria
# }}
# '''

SCREENING_CRITERIA_GENERATION = """
You are a **clinical research specialist** conducting a **systematic review** of clinical trials based on the **PRISMA guidelines**. Your task is to **design eligibility criteria** for selecting studies relevant to this review.

The research is defined by the following **research topic/query** and **PICO elements**:
- **Research Topic/Query:** {research_topic}  
- **P (Population/Problem):** {P}  
- **I (Intervention):** {I}  
- **C (Comparison):** {C}  
- **O (Outcome):** {O}  

## **Task Instructions**

### **Step 1: Define Eligibility Criteria**
Using the PICO framework and PRISMA guidelines, identify **explicit eligibility criteria** for the inclusion and exclusion of studies. Each criterion should include a **clear rationale** to justify why the criterion is necessary. Consider factors such as:  

- **Study Type** (e.g., RCTs, observational studies, meta-analyses)  
- **Population Characteristics** (e.g., age, disease severity, geographic location)  
- **Intervention Specificity** (e.g., dosage, treatment duration, administration method)  
- **Comparators** (e.g., placebo, standard care, alternative interventions)  
- **Outcomes of Interest** (e.g., clinical endpoints, adverse effects, biomarkers)  
- **Study Type** (e.g., clinical trials)  

Please format the eligibility criteria as follows:  

**ELIGIBILITY ANALYSIS:**  
- **Inclusion Criteria:**  
  **[Criterion]** – [Rationale]  
  **[Criterion]** – [Rationale]  
  ...  
- **Exclusion Criteria:**  
  **[Criterion]** – [Rationale]  
  **[Criterion]** – [Rationale]  
  ...  

### **Step 2: Develop Screening Questions**  
To streamline the study selection process, create **{num_eligibility_criteria} binary screening questions** that will be used to determine study eligibility based on titles and abstracts.  

- These questions should be **binary (YES/NO)**, where **YES** means the study is eligible, and **NO** means it is not.  
- The information required to answer these questions should be **general** and **easily found** in the study title and abstract.  

## Reply Format
You should reply in a JSON format like:

{{
    "ELIGIBILITY_ANALYSIS": "your eligibility analysis here..." \\ the bullet points of your analysis
    "ELIGIBILITY_CRITERIA": ["criterion1", "criterion2", "..."] \\ the {num_eligibility_criteria} binary eligibility criteria
}}
"""
SCREENING_CRITERIA_GENERATION_CUSTOMFILTERS = """
You are a **clinical research specialist** conducting a **systematic review** of clinical trials based on the **PRISMA guidelines**. Your task is to **design eligibility criteria** for selecting studies relevant to this review.

The research is defined by the following **research topic/query** and **user-provided filters**:
- **Research Topic/Query:** {research_topic}  
{custom_filters_text}

## **Task Instructions**

### **Step 1: Define Eligibility Criteria**
Using the custom filters and PRISMA guidelines, identify **explicit eligibility criteria** for the inclusion and exclusion of studies. Each criterion should include a **clear rationale** to justify why the criterion is necessary. Consider factors such as:  

- **Study Type** (e.g., RCTs, observational studies, meta-analyses)  
- **Population Characteristics** (e.g., age, disease severity, geographic location)  
- **Intervention Specificity** (e.g., dosage, treatment duration, administration method)  
- **Comparators** (e.g., placebo, standard care, alternative interventions)  
- **Outcomes of Interest** (e.g., clinical endpoints, adverse effects, biomarkers)  
- **Study Type** (e.g., clinical trials)  

Please format the eligibility criteria as follows:  

**ELIGIBILITY ANALYSIS:**  
- **Inclusion Criteria:**  
  **[Criterion]** – [Rationale]  
  **[Criterion]** – [Rationale]  
  ...  
- **Exclusion Criteria:**  
  **[Criterion]** – [Rationale]  
  **[Criterion]** – [Rationale]  
  ...  

### **Step 2: Develop Screening Questions**  
To streamline the study selection process, create **{num_eligibility_criteria} binary screening questions** that will be used to determine study eligibility based on titles and abstracts.  

- These questions should be **binary (YES/NO)**, where **YES** means the study is eligible, and **NO** means it is not.  
- The information required to answer these questions should be **general** and **easily found** in the study title and abstract.  

## Reply Format
You should reply in a JSON format like:

{{
    "ELIGIBILITY_ANALYSIS": "your eligibility analysis here..." \\ the bullet points of your analysis
    "ELIGIBILITY_CRITERIA": ["criterion1", "criterion2", "..."] \\ the {num_eligibility_criteria} binary eligibility criteria
}}
"""