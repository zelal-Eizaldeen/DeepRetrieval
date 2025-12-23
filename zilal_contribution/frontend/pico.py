PICO_GENERATION = """You are a clinical research specialist. Your task is to analyze a research topic/question and break it down into PICO elements.

The research topic/question is:
{research_topic}

## Task
Break down the research topic/question into the four PICO elements:

1. Population (P): The patient population or problem being studied
2. Intervention (I): The treatment, exposure, or diagnostic test being studied
3. Comparator (C): The alternative treatment, exposure, or diagnostic test being compared
4. Outcome (O): The outcome or endpoint being measured

## Guidelines
- Be specific and precise in identifying each PICO element
- If a PICO element is not explicitly mentioned, infer it based on clinical context
- For the comparator, if no explicit comparison is mentioned, use "standard care" or "placebo" as appropriate
- Outcomes should be measurable and clinically relevant

## Reply Format
You should reply in a JSON format like this:
{{
    "population": "description of the population/patient group",
    "intervention": "description of the intervention being studied",
    "comparator": "description of the comparison/control group",
    "outcome": "description of the outcome being measured"
}}
"""

PICOT_GENERATION = """You are a clinical research specialist. Your task is to analyze a research topic/question and break it down into PICOT elements.

The research topic/question is:
{research_topic}

## Task
Break down the research topic/question into the PICOT elements:

1. Population: The patient population or problem being studied
2. Intervention: The treatment, exposure, or diagnostic test being studied
3. Comparator: The alternative treatment, exposure, or diagnostic test being compared
4. Outcome: The outcome or endpoint being measured
5. Time: The time frame, duration, or temporal considerations

## Guidelines
- Be specific and precise in identifying each element
- If an element is not explicitly mentioned, infer it based on clinical context
- Time should include duration of intervention, follow-up period, or time to outcome
- Consider both short-term and long-term temporal aspects if relevant

## Reply Format
You should reply in a JSON format like this:
{{
    "Population": "description of the population/patient group",
    "Intervention": "description of the intervention being studied",
    "Comparator": "description of the comparison/control group",
    "Outcome": "description of the outcome being measured",
    "Time": "description of temporal considerations"
}}
"""

PICOS_GENERATION = """You are a clinical research specialist. Your task is to analyze a research topic/question and break it down into PICOS elements.

The research topic/question is:
{research_topic}

## Task
Break down the research topic/question into the PICOS elements:

1. Population: The patient population or problem being studied
2. Intervention: The treatment, exposure, or diagnostic test being studied
3. Comparator: The alternative treatment, exposure, or diagnostic test being compared
4. Outcome: The outcome or endpoint being measured
5. Study Design: The types of studies to include

## Guidelines
- Be specific and precise in identifying each element
- If an element is not explicitly mentioned, infer it based on clinical context
- For study design, suggest appropriate study types (e.g., RCT, cohort, case-control, systematic reviews)
- Consider what study designs would best answer the research question

## Reply Format
You should reply in a JSON format like this:
{{
    "Population": "description of the population/patient group",
    "Intervention": "description of the intervention being studied",
    "Comparator": "description of the comparison/control group",
    "Outcome": "description of the outcome being measured",
    "Study Design": "description of preferred study types"
}}
"""

SPIDER_GENERATION = """You are a research specialist. Your task is to analyze a research topic/question and break it down into SPIDER elements for qualitative or mixed-methods research.

The research topic/question is:
{research_topic}

## Task
Break down the research topic/question into the SPIDER elements:

1. Sample: The participants, people, organization, or culture being studied
2. Phenomenon of Interest: The behaviors, experiences, or issues being studied
3. Design: The research design or methods
4. Evaluation: The outcome measures or what is being assessed
5. Research type: The broad research methodology

## Guidelines
- SPIDER is particularly suited for qualitative and mixed-methods research
- Sample should describe who is being studied in detail
- Phenomenon of Interest should focus on experiences, perceptions, or behaviors
- Design might include qualitative approaches (ethnography, grounded theory, phenomenology)
- Research type should specify qualitative, quantitative, or mixed-methods

## Reply Format
You should reply in a JSON format like this:
{{
    "Sample": "description of the sample or participants",
    "Phenomenon of Interest": "description of what is being studied",
    "Design": "description of research design",
    "Evaluation": "description of outcome measures",
    "Research type": "description of research methodology"
}}
"""

SPICE_GENERATION = """You are a health services research specialist. Your task is to analyze a research topic/question and break it down into SPICE elements.

The research topic/question is:
{research_topic}

## Task
Break down the research topic/question into the SPICE elements:

1. Setting: The location, environment, or context
2. Perspective: The stakeholder perspective or viewpoint
3. Intervention: The intervention, service, program, or policy
4. Comparison: The alternative intervention or standard practice
5. Evaluation: The outcomes, impacts, or effects being measured

## Guidelines
- SPICE is particularly suited for service delivery, quality improvement, and policy questions
- Setting should describe where the intervention takes place
- Perspective should identify whose viewpoint matters (patients, providers, policymakers)
- Consider organizational and system-level factors
- Evaluation should focus on meaningful impacts and outcomes

## Reply Format
You should reply in a JSON format like this:
{{
    "Setting": "description of context or environment",
    "Perspective": "description of stakeholder perspective",
    "Intervention": "description of intervention or policy",
    "Comparison": "description of alternative",
    "Evaluation": "description of outcomes being evaluated"
}}
"""

ECLIPSE_GENERATION = """You are a health policy and management specialist. Your task is to analyze a research topic/question and break it down into ECLIPSE elements.

The research topic/question is:
{research_topic}

## Task
Break down the research topic/question into the ECLIPSE elements:

1. Expectation: The information need or expectation
2. Client group: The target population or recipients
3. Location: The care setting or geographic location
4. Impact: The desired outcomes or changes
5. Professionals: The healthcare professionals or service providers
6. Service: The specific service or intervention

## Guidelines
- ECLIPSE is particularly suited for health policy, management, and service delivery questions
- Expectation should clarify what information or knowledge is being sought
- Client group identifies who is affected
- Location provides context for where services are delivered
- Impact should focus on meaningful changes and improvements
- Consider the roles of different professionals and services

## Reply Format
You should reply in a JSON format like this:
{{
    "Expectation": "description of information need",
    "Client group": "description of target population",
    "Location": "description of setting",
    "Impact": "description of desired outcomes",
    "Professionals": "description of service providers",
    "Service": "description of intervention or service"
}}
"""

PEO_GENERATION = """You are a research specialist. Your task is to analyze a research topic/question and break it down into PEO elements.

The research topic/question is:
{research_topic}

## Task
Break down the research topic/question into the PEO elements:

1. Population: The population or group being studied
2. Exposure: The exposure, risk factor, or variable of interest
3. Outcome: The outcomes or effects being measured

## Guidelines
- PEO is particularly suited for observational studies, epidemiology, and social sciences
- Population should describe the group, demographic, or subjects being studied
- Exposure can include risk factors, characteristics, behaviors, or variables being examined
- Outcome should focus on the effects, associations, or results being investigated
- Consider both direct and indirect exposures and outcomes

## Reply Format
You should reply in a JSON format like this:
{{
    "Population": "description of population being studied",
    "Exposure": "description of exposure or risk factor",
    "Outcome": "description of outcomes being measured"
}}
"""

PECO_GENERATION = """You are an environmental health and toxicology specialist. Your task is to analyze a research topic/question and break it down into PECO elements.

The research topic/question is:
{research_topic}

## Task
Break down the research topic/question into the PECO elements:

1. Population: The population or organisms being studied
2. Exposure: The environmental exposure or factor
3. Comparator: The comparison group or exposure level
4. Outcome: The health or environmental outcomes

## Guidelines
- PECO is particularly suited for environmental health, toxicology, and observational environmental studies
- Population can include humans, animals, or ecosystems
- Exposure should describe environmental factors, pollutants, or chemicals
- Comparator might be unexposed groups, different exposure levels, or reference conditions
- Outcome should focus on health effects, environmental impacts, or measurable endpoints
- Consider dose-response relationships and temporal factors

## Reply Format
You should reply in a JSON format like this:
{{
    "Population": "description of population or organisms",
    "Exposure": "description of environmental exposure",
    "Comparator": "description of comparison group",
    "Outcome": "description of health or environmental outcomes"
}}
"""

PCC_GENERATION = """You are a research specialist. Your task is to analyze a research topic/question and break it down into PCC elements for a scoping review.

The research topic/question is:
{research_topic}

## Task
Break down the research topic/question into the PCC elements:

1. Population: The population, participants, or subject of interest
2. Concept: The core concept, topic, or phenomenon
3. Context: The context, setting, or circumstances

## Guidelines
- PCC is particularly suited for scoping reviews and broad exploratory research across any domain
- Population can be very broadly defined (people, organizations, systems, concepts)
- Concept should identify the central idea, phenomenon, or topic being explored
- Context provides the setting, environment, or circumstances that frame the research
- This framework is flexible and applicable to questions in any field
- Focus on breadth rather than specific interventions or outcomes

## Reply Format
You should reply in a JSON format like this:
{{
    "Population": "description of population or subject",
    "Concept": "description of core concept",
    "Context": "description of context or setting"
}}
""" 