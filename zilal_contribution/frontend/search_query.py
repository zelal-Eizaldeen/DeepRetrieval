CATEGORY_DETERMINATION = '''You are a clinical specialist conducting a systematic literature review.

## Task
Based on the research topic and parameters provided, determine the most appropriate search categories for organizing search terms. 

Research Topic/Query: {research_topic}
P (Patient, Problem or Population): {P}
I (Intervention): {I}
C (Comparison): {C}
O (Outcome): {O}

## Guidelines
- Default categories are "conditions" and "treatments" 
- Consider if additional or alternative categories would better capture the research scope
- Examples of other useful categories: "outcomes", "populations", "biomarkers", "procedures", "exposures", "risk factors", "study types", "age groups", "geographic regions", etc.
- Remember that search uses AND across categories and OR within categories
- If the research needs to find co-occurring conditions, consider separate categories for each
- Typically return 2-4 categories

## Reply Format
Return a JSON object with:
- "categories": list of category names (as strings)
- "reasoning": brief explanation of why these categories were chosen

{{
    "categories": ["category1", "category2", "category3"],
    "reasoning": "Brief explanation of category choices"
}}
'''

CATEGORY_DETERMINATION_CUSTOMFILTERS = '''You are a clinical specialist conducting a systematic literature review.

## Task
Based on the research topic and custom filters provided, determine the most appropriate search categories for organizing search terms. 

Research Topic/Query: {research_topic}
{custom_filters_text}

## Guidelines
- Default categories are "conditions" and "treatments" 
- Consider if additional or alternative categories would better capture the research scope
- Examples of other useful categories: "outcomes", "populations", "biomarkers", "procedures", "exposures", "risk factors", "study types", "age groups", "geographic regions", etc.
- Remember that search uses AND across categories and OR within categories
- If the research needs to find co-occurring conditions, consider separate categories for each
- Typically return 2-4 categories

## Reply Format
Return a JSON object with:
- "categories": list of category names (as strings)
- "reasoning": brief explanation of why these categories were chosen

{{
    "categories": ["category1", "category2", "category3"],
    "reasoning": "Brief explanation of category choices"
}}
'''

PRIMARY_TERM_EXTRACTION = '''You are a clinical specialist. You are conducting a clinical study meta-analysis.
The research is defined by the following research topic/query and PICO elements:
Research Topic/Query: {research_topic}
P (Patient, Problem or Population): {P}
I (Intervention): {I}
C (Comparison): {C}
O (Outcome): {O}

## Task
Your task is to identify the primary clinical term(s) in this research. 
The clinical terms should be specific medical conditions, treatments, or procedures. 
General terms such as 'patients', or 'therapy' should not be included.

## Reply Format
You should only reply with 1~3 primary term. Your output should be in JSON format, like this:

{{
    "terms": ["term1", "term2", "term3"]
}}
'''

SEARCH_TERM_EXTRACTION = """
## background

You are a clinical specialist. You are conducting a clinical meta-analysis.
The research is defined by the following research topic/query and PICO elements:
Research Topic/Query: {research_topic}
P (Patient, Problem or Population): {P}
I (Intervention): {I}
C (Comparison): {C}
O (Outcome): {O}

## Reference

You've already gathered these related papers: 
{pubmed_reference_text}

## Task

Your task is to further your literature search by these 3 steps:

### Step 1
Extract related term in the reference papers.
Provide two lists of query terms: TREATMENTS, CONDITIONS.

CONDITIONS: words about any conditions or disease that is related to this meta-analysis (refering to Problem section)
TREATMENTS: primary related clinical terms/keywords showed in these reference papers (refering to Intervention section)

### Step 2
Double-check these query terms, remove the terms that is not directly related to the PICO elements of this research.
Provide two lists of refined core terms: CORE_CONDITIONS, CORE_TREATMENTS.

CORE_CONDITIONS: refined terms of conditions or disease
CORE_TREATMENTS: refined terms of primary related clinical terms/keywords

### Step 3
To expand the scope of query term searches, please extend each query term by: 
1. Synonyms and other names/forms; 
2. Possible abbreviations or full forms; 
3. Split into elements for compound phrases. 

Provide two lists of expanded query terms: EXPAND_CONDITIONS, EXPAND_TREATMENTS.

EXPAND_CONDITIONS: expanded terms of conditions or disease
EXPAND_TREATMENTS: expanded terms of primary related clinical terms/keywords


## Reply format
There should be no overlap between these each pair of lists

Your reply should be in a JSON format like: 

{{

"step 1": {{
    "CONDITIONS": [condition1, condition2, ..] \\ (~10 items)
    "TREATMENTS": [term1, term2 .. ] \\ (~10 items)
}},

\\ Refine according to P (Patient, Problem or Population): {P} and I (Intervention): {I} and O (Outcome): {O}
"step 2": {{
    "CORE_CONDITIONS": [condition1, condition2, ..] \\ (~5 items)
    "CORE_TREATMENTS": [term1, term2, .. ] \\ (~5 items)
}},

\\ Augumentation
"step 3": {{
    "EXPAND_CONDITIONS": [condition1, condition2, ..]  \\ (~10 items)
    "EXPAND_TREATMENTS": [term1, term2 ..] \\ (~10 items)
    }}
}}
"""

USER_REQUEST_SEARCH_QUERY_GENERATION = """

## Background

You are a clinical specialist conducting a systematic review. Your goal is to generate a comprehensive set of search terms that maximize recall while maintaining relevance.

## Task

You are given a user's request. Your task is to suggest relevant search terms, expanding beyond the exact phrasing of the request by considering:

- Synonyms and alternative terms  
- Related conditions, interventions, and outcomes  
- Abbreviations and full names of terms  
- Variations in medical terminology and phrasing  
- Broader and narrower concepts if relevant  

The user's request is:  
**{user_request}**  

## Reply Format

Your response should be in JSON format:

```json
{{
    "conditions": [ "term1", "term2", ... ],  \\ blank if not required by the user's request
    "treatments": [ "term1", "term2", ... ],   \\ blank if not required by the user's request
}}
```

- **conditions**: Relevant diseases, patient populations, or medical conditions mentioned or implied in the request. And any additional concepts, including mechanisms, biomarkers, outcome measures, or methodologies that might be useful for expanding the search  
- **treatments**: Drugs, procedures, or interventions directly or indirectly relevant to the request  
"""

CATEGORY_SPECIFIC_SEARCH_TERM_EXTRACTION = """
## background

You are a clinical specialist. You are conducting a clinical meta-analysis.
The research is defined by the following research topic/query and PICO elements:
Research Topic/Query: {research_topic}
P (Patient, Problem or Population): {P}
I (Intervention): {I}
C (Comparison): {C}
O (Outcome): {O}

## Reference

You've already gathered these related papers: 
{pubmed_reference_text}

## Task

Your task is to generate search terms specifically for the category: **{category_name}**

Follow these 3 steps:

### Step 1
Extract related terms in the reference papers that are relevant to the category "{category_name}".
Provide a list of query terms: {category_name.upper()}.

{category_name}: terms about {category_name} that are related to this meta-analysis

### Step 2
Double-check these query terms, remove the terms that are not directly related to the PICO elements of this research.
Provide a list of refined core terms: CORE_{category_name.upper()}.

CORE_{category_name}: refined terms of {category_name}

### Step 3
To expand the scope of query term searches, please extend each query term by: 
1. Synonyms and other names/forms; 
2. Possible abbreviations or full forms; 
3. Split into elements for compound phrases. 

Provide a list of expanded query terms: EXPAND_{category_name}.

EXPAND_{category_name}: expanded terms of {category_name}

## Reply format

Your reply should be in a JSON format like: 

{{
"step 1": {{
    "{category_name}": [term1, term2, ..] \\ (~10 items)
}},

\\ Refine according to P (Patient, Problem or Population): {P} and I (Intervention): {I} and O (Outcome): {O}
"step 2": {{
    "CORE_{category_name}": [term1, term2, ..] \\ (~5 items)
}},

\\ Augmentation
"step 3": {{
    "EXPAND_{category_name}": [term1, term2, ..] \\ (~10 items)
}}
}}
"""

CATEGORY_SPECIFIC_USER_REQUEST_SEARCH_QUERY_GENERATION = """
## Background

You are a clinical specialist conducting a systematic literature review. Your goal is to generate a comprehensive set of specific search terms for the category: **{category_name}**

## Task

You are given a user's request. Your task is to suggest the most relevant and specific search terms for the category "{category_name}" by considering:

- Synonyms and alternative terms if the request is a specific concept
- Related concepts within the "{category_name}" category (especially if the request is a category or description rather than a specific concept)
- Abbreviations and full names of terms
- Variations in terminology and phrasing
- Broader and narrower concepts if relevant

**IMPORTANT**: Generate only the most specific and directly relevant terms for {category_name}. Avoid broad or meaningless concepts. Focus on an array of precise terminology within this category.

The user's request is:  
**{user_request}**  

## Reply Format

Your response should be in JSON format:

```json
{{
    "terms": [ "term1", "term2", ... ]
}}
```

- **terms**: Specific {category_name} terms mentioned or implied in the request, plus any additional concepts that might be useful for expanding the search within this category. Focus on specific concepts or terminology within this category (~7-10 terms max)
"""

PRIMARY_TERM_EXTRACTION_CUSTOMFILTERS = '''You are a clinical specialist. You are conducting a clinical study meta-analysis.
The research is defined by the following research topic/query and user-provided filters:
Research Topic/Query: {research_topic}
{custom_filters_text}

## Task
Your task is to identify the primary clinical term(s) in this research. 
The clinical terms should be specific medical conditions, treatments, or procedures. 
General terms such as 'patients', or 'therapy' should not be included.

## Reply Format
You should only reply with 1~3 primary term. Your output should be in JSON format, like this:

{{
    "terms": ["term1", "term2", "term3"]
}}
'''

SEARCH_TERM_EXTRACTION_CUSTOMFILTERS = """
## background

You are a clinical specialist. You are conducting a clinical meta-analysis.
The research is defined by the following research topic/query and user-provided filters:
Research Topic/Query: {research_topic}
{custom_filters_text}

## Reference

You've already gathered these related papers: 
{pubmed_reference_text}

## Task

Your task is to further your literature search by these 3 steps:

### Step 1
Extract related term in the reference papers.
Provide two lists of query terms: TREATMENTS, CONDITIONS.

CONDITIONS: words about any conditions or disease that is related to this meta-analysis
TREATMENTS: primary related clinical terms/keywords showed in these reference papers

### Step 2
Double-check these query terms, remove the terms that are not directly related to the custom filters of this research.
Provide two lists of refined core terms: CORE_CONDITIONS, CORE_TREATMENTS.

CORE_CONDITIONS: refined terms of conditions or disease
CORE_TREATMENTS: refined terms of primary related clinical terms/keywords

### Step 3
To expand the scope of query term searches, please extend each query term by: 
1. Synonyms and other names/forms; 
2. Possible abbreviations or full forms; 
3. Split into elements for compound phrases. 

Provide two lists of expanded query terms: EXPAND_CONDITIONS, EXPAND_TREATMENTS.

EXPAND_CONDITIONS: expanded terms of conditions or disease
EXPAND_TREATMENTS: expanded terms of primary related clinical terms/keywords


## Reply format
There should be no overlap between these each pair of lists

Your reply should be in a JSON format like: 

{{

"step 1": {{
    "CONDITIONS": [condition1, condition2, ..] \ (~10 items)
    "TREATMENTS": [term1, term2 .. ] \ (~10 items)
}},

\
"step 2": {{
    "CORE_CONDITIONS": [condition1, condition2, ..] \ (~5 items)
    "CORE_TREATMENTS": [term1, term2, .. ] \ (~5 items)
}},

\ Augumentation
"step 3": {{
    "EXPAND_CONDITIONS": [condition1, condition2, ..]  \ (~10 items)
    "EXPAND_TREATMENTS": [term1, term2 ..] \ (~10 items)
    }}
}}
"""

CATEGORY_SPECIFIC_SEARCH_TERM_EXTRACTION_CUSTOMFILTERS = """
## background

You are a clinical specialist. You are conducting a clinical meta-analysis.
The research is defined by the following research topic/query and user-provided filters:
Research Topic/Query: {research_topic}
{custom_filters_text}

## Reference

You've already gathered these related papers: 
{pubmed_reference_text}

## Task

Your task is to generate search terms specifically for the category: **{category_name}**

Follow these 3 steps:

### Step 1
Extract related terms in the reference papers that are relevant to the category "{category_name}".
Provide a list of query terms: {category_name.upper()}.

{category_name}: terms about {category_name} that are related to this meta-analysis

### Step 2
Double-check these query terms, remove the terms that are not directly related to the custom filters of this research.
Provide a list of refined core terms: CORE_{category_name.upper()}.

CORE_{category_name}: refined terms of {category_name}

### Step 3
To expand the scope of query term searches, please extend each query term by: 
1. Synonyms and other names/forms; 
2. Possible abbreviations or full forms; 
3. Split into elements for compound phrases. 

Provide a list of expanded query terms: EXPAND_{category_name}.

EXPAND_{category_name}: expanded terms of {category_name}

## Reply format

Your reply should be in a JSON format like: 

{{
"step 1": {{
    "{category_name}": [term1, term2, ..] \ (~10 items)
}},

\ Refine according to custom filters
"step 2": {{
    "CORE_{category_name}": [term1, term2, ..] \ (~5 items)
}},

\ Augmentation
"step 3": {{
    "EXPAND_{category_name}": [term1, term2, ..] \ (~10 items)
}}
}}
"""

DYNAMIC_SEARCH_TERM_EXTRACTION = """
## background

You are a clinical specialist. You are conducting a clinical meta-analysis.
The research is defined by the following research topic/query and PICO elements:
Research Topic/Query: {research_topic}
P (Patient, Problem or Population): {P}
I (Intervention): {I}
C (Comparison): {C}
O (Outcome): {O}

## Reference

You've already gathered these related papers: 
{pubmed_reference_text}

## Task

Your task is to further your literature search by generating search terms for these categories: {categories}

Follow these 3 steps for EACH category:

### Step 1
Extract related terms in the reference papers for each category.
For each category, provide a list of query terms.

### Step 2
Double-check these query terms, remove the terms that are not directly related to the PICO elements of this research.
For each category, provide a list of refined core terms.

### Step 3
To expand the scope of query term searches, please extend each query term by: 
1. Synonyms and other names/forms; 
2. Possible abbreviations or full forms; 
3. Split into elements for compound phrases. 

For each category, provide a list of expanded query terms.

## Reply format

Your reply should be in a JSON format. For each category, provide three lists following the pattern below:

{{
    "category1": {{
        "step 1": {{
            "INITIAL": [term1, term2, ..] \\ (~10 items)
        }},
        "step 2": {{
            "CORE": [term1, term2, ..] \\ (~5 items)
        }},
        "step 3": {{
            "EXPANDED": [term1, term2, ..] \\ (~10 items)
        }}
    }},
    "category2": {{
        "step 1": {{
            "INITIAL": [term1, term2, ..] \\ (~10 items)
        }},
        "step 2": {{
            "CORE": [term1, term2, ..] \\ (~5 items)
        }},
        "step 3": {{
            "EXPANDED": [term1, term2, ..] \\ (~10 items)
        }}
    }}
}}
"""

DYNAMIC_SEARCH_TERM_EXTRACTION_CUSTOMFILTERS = """
## background

You are a clinical specialist. You are conducting a clinical meta-analysis.
The research is defined by the following research topic/query and user-provided filters:
Research Topic/Query: {research_topic}
{custom_filters_text}

## Reference

You've already gathered these related papers: 
{pubmed_reference_text}

## Task

Your task is to further your literature search by generating search terms for these categories: {categories}

Follow these 3 steps for EACH category:

### Step 1
Extract related terms in the reference papers for each category.
For each category, provide a list of query terms.

### Step 2
Double-check these query terms, remove the terms that are not directly related to the custom filters of this research.
For each category, provide a list of refined core terms.

### Step 3
To expand the scope of query term searches, please extend each query term by: 
1. Synonyms and other names/forms; 
2. Possible abbreviations or full forms; 
3. Split into elements for compound phrases. 

For each category, provide a list of expanded query terms.

## Reply format

Your reply should be in a JSON format. For each category, provide three lists following the pattern below:

{{
    "category1": {{
        "step 1": {{
            "INITIAL": [term1, term2, ..] \\ (~10 items)
        }},
        "step 2": {{
            "CORE": [term1, term2, ..] \\ (~5 items)
        }},
        "step 3": {{
            "EXPANDED": [term1, term2, ..] \\ (~10 items)
        }}
    }},
    "category2": {{
        "step 1": {{
            "INITIAL": [term1, term2, ..] \\ (~10 items)
        }},
        "step 2": {{
            "CORE": [term1, term2, ..] \\ (~5 items)
        }},
        "step 3": {{
            "EXPANDED": [term1, term2, ..] \\ (~10 items)
        }}
    }}
}}
"""

DYNAMIC_USER_REQUEST_SEARCH_QUERY_GENERATION = """
## Background

You are a clinical specialist conducting a systematic review. Your goal is to generate a comprehensive set of search terms that maximize recall while maintaining relevance.

## Task

You are given a user's request. Your task is to suggest relevant search terms for the following categories: {categories}

For each category, expand beyond the exact phrasing of the request by considering:

- Synonyms and alternative terms  
- Related concepts within that category
- Abbreviations and full names of terms  
- Variations in medical terminology and phrasing  
- Broader and narrower concepts if relevant  

The user's request is:  
**{user_request}**  

## Reply Format

Your response should be in JSON format with one key per category:

```json
{{
    "category1": [ "term1", "term2", ... ],
    "category2": [ "term1", "term2", ... ]
}}
```

Each category should contain relevant terms that might be useful for expanding the search. Leave a category as an empty list [] if not applicable to the user's request.
"""