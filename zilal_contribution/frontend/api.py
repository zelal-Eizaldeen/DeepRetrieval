import json
import logging

from trialmind.llm import call_llm_json_output
from trialmind.api.data_models import CustomFilter

def _rerank_studies_bm25(data, keyword_maps, topn=10):
    """Rerank the search results based on BM25.
    """
    from trialmind.TrialDesign.KBService.vectordb_utils.ranking import rerank_docs_by_bm25
    # get weighted docs, 10 * title + 1 * abstract
    docs = (data["Title"].fillna("").astype(str) * 10 + data["Abstract"].fillna("").astype(str)).tolist()
    id2score = {idx: 0 for idx in range(len(docs))}
    kw_idx = 0
    for k, v in keyword_maps.items():
        # multiple keywords fusion, each keyword is equally important
        ranked_indices, ranked_scores = rerank_docs_by_bm25(docs, v, topn=topn)
        top_ids = np.where(np.array(ranked_scores) > 0)[0]
        for rank, idx in enumerate(np.array(ranked_indices)[top_ids]):
            id2score[idx] += (1/(rank+1))
        kw_idx += 1
    id2score = sorted(id2score.items(), key=lambda x: -x[1])
    ranked_indices, ranked_scores = zip(*id2score)
    data = data.iloc[list(ranked_indices)].reset_index(drop=True)
    return data


class PICOGeneration:
    """
    Input the user's input research question, generate the PICO elements.

    Args:
        research_topic: The research topic or question to analyze.
        llm: The language model to use for PICO generation. Default is "gpt-4o".
    """

    def __init__(self):
        pass

    def run(self, research_topic: str, llm: str = "gpt-4o"):
        """Generate PICO elements from a research topic/question.

        Args:
            research_topic: The research topic or question to analyze.
            llm: The language model to use for PICO generation.

        Returns:
            dict: Dictionary containing the PICO elements:
                - population: The population/patient group
                - intervention: The intervention being studied
                - comparator: The comparison/control group
                - outcome: The outcome being measured
        """
        from trialmind.TrialMetaAnalysis.prompts.pico import PICO_GENERATION

        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                outputs = call_llm_json_output(
                    PICO_GENERATION,
                    {"research_topic": research_topic},
                    llm=llm,
                    temperature=0.01,
                    max_completion_tokens=1024,
                )
                outputs = json.loads(outputs)
                break
            except (Exception, json.JSONDecodeError) as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise e
        
        return outputs


class FilterGeneration:
    """
    Generate filter elements for various research frameworks (PICOT, PICOS, SPIDER, SPICE, ECLIPSE, PEO, PECO, PCC)
    
    Args:
        research_topic: The research topic or question to analyze.
        filter_type: The type of filter framework
        llm: The language model to use. Default is "gpt-4o".
    """
    
    SUPPORTED_TYPES = ["PICOT", "PICOS", "SPIDER", "SPICE", "ECLIPSE", "PEO", "PECO", "PCC"]
    
    def __init__(self):
        pass
    
    def run(self, research_topic: str, filter_type: str, llm: str = "gpt-4o"):
        """Generate filter elements from a research topic/question.
        
        Args:
            research_topic: The research topic or question to analyze.
            filter_type: The type of filter framework (PICOT, PICOS, SPIDER, SPICE, ECLIPSE, PEO, PECO, PCC)
            llm: The language model to use.
            
        Returns:
            dict: Dictionary containing the filter elements specific to the framework type.
        """
        from trialmind.TrialMetaAnalysis.prompts.pico import (
            PICOT_GENERATION, 
            PICOS_GENERATION, 
            SPIDER_GENERATION, 
            SPICE_GENERATION, 
            ECLIPSE_GENERATION,
            PEO_GENERATION,
            PECO_GENERATION,
            PCC_GENERATION
        )
        
        filter_type = filter_type.upper()
        if filter_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported filter type: {filter_type}. Supported types: {self.SUPPORTED_TYPES}")
        
        # Map filter type to prompt
        prompt_map = {
            "PICOT": PICOT_GENERATION,
            "PICOS": PICOS_GENERATION,
            "SPIDER": SPIDER_GENERATION,
            "SPICE": SPICE_GENERATION,
            "ECLIPSE": ECLIPSE_GENERATION,
            "PEO": PEO_GENERATION,
            "PECO": PECO_GENERATION,
            "PCC": PCC_GENERATION,
        }
        
        prompt = prompt_map[filter_type]
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                outputs = call_llm_json_output(
                    prompt,
                    {"research_topic": research_topic},
                    llm=llm,
                    temperature=0.01,
                    max_completion_tokens=1024,
                )
                outputs = json.loads(outputs)
                break
            except (Exception, json.JSONDecodeError) as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise e
        
        return outputs


class SearchQueryGeneration:
    """
    Input the user's input research question, generate the search query for the searching clinical studies.

    Args:
        population (str): The population of the research question.
        intervention (str): The intervention of the research question.
        comparator (str): The comparator of the research question.
        outcome (str): The outcome of the research question.
        llm (str): The language model to use for the search query generation. Default is "gpt-4o".
    """
    # TODO: https://training.cochrane.org/handbook/current/chapter-04#section-4-4-4
    # refer to the Cochrane Handbook for Systematic Reviews of Interventions
    # to draft the prompt for making the search query
    def __init__(self):
        pass

    def run(self,
            research_topic: str,
            population: str,
            intervention: str,
            comparator: str,
            outcome: str,
            custom_filters: list[CustomFilter],
            user_request: str,
            category_name: str,
            llm: str="gpt-4o"
        ):
        # Handle category-specific search query generation
        if category_name and category_name.strip() != "":
            # Special case: if category_name is "conditions" or "treatments", use existing pipeline but return single category
            if category_name.lower() in ["conditions", "condition", "treatments", "treatment"]:
                if len(user_request) == 0:
                    # Use normal search query generation
                    terms = self._run_init_term_generation(research_topic, population, intervention, comparator, outcome, llm=llm, custom_filters=custom_filters)
                    terms = 'AND'.join([f'({k})' for k in terms])
                    logging.info(f"Generate initial search terms: {terms}")

                    pmids = self._run_pubmed_id_search(terms)
                    logging.info(f"Fetch initial reference pubmed paper ids {pmids}")
                    pubmed_reference_text = self._run_pubmed_full_search(pmids)

                    outputs = self._run_final_search_query_generation(research_topic, population, intervention, comparator, outcome, pubmed_reference_text, llm=llm, custom_filters=custom_filters)
                    
                    # Return only the requested category as a list
                    if category_name.lower() in ["conditions", "condition"]:
                        return outputs["conditions"]
                    else:  # treatments or treatment
                        return outputs["treatments"]
                else:
                    # Use user request search query generation
                    outputs = self._run_user_request_search_query_generation(user_request, llm=llm)
                    
                    # Return only the requested category as a list
                    if category_name.lower() in ["conditions", "condition"]:
                        return outputs["conditions"]
                    else:  # treatments or treatment
                        return outputs["treatments"]
            else:
                # Use category-specific search query generation
                return self._run_category_specific_search_query_generation(research_topic, population, intervention, comparator, outcome, user_request, category_name, llm=llm)
        
        # Original pipeline for when category_name is empty (dynamic category determination)
        if len(user_request) == 0: # do the normal search query generation
            # Step 1: Determine appropriate categories
            categories = self._run_category_determination(research_topic, population, intervention, comparator, outcome, llm=llm, custom_filters=custom_filters)
            
            # Step 2: Get initial terms for PubMed search
            terms = self._run_init_term_generation(research_topic, population, intervention, comparator, outcome, llm=llm, custom_filters=custom_filters)
            terms = 'AND'.join([f'({k})' for k in terms])
            logging.info(f"Generate initial search terms: {terms}")

            # Step 3: Find reference pubmed papers
            pmids = self._run_pubmed_id_search(terms)
            logging.info(f"Fetch initial reference pubmed paper ids {pmids}")
            pubmed_reference_text = self._run_pubmed_full_search(pmids)

            # Step 4: Generate the final search query dynamically for all categories
            outputs = self._run_dynamic_final_search_query_generation(
                research_topic, population, intervention, comparator, outcome, 
                pubmed_reference_text, categories, llm=llm, custom_filters=custom_filters
            )

            # Return dynamic category structure
            return outputs
        else: # do the user request search query generation
            # Step 1: Determine appropriate categories
            categories = self._run_category_determination(research_topic, population, intervention, comparator, outcome, llm=llm, custom_filters=custom_filters)
            
            # Step 2: Generate the search query based on the user's request with dynamic categories
            outputs = self._run_dynamic_user_request_search_query_generation(user_request, categories, llm=llm)
            
            # Return dynamic category structure
            return outputs

    def _run_init_term_generation(self, research_topic, population, intervention, comparator, outcome, llm, custom_filters=None):
        from trialmind.TrialMetaAnalysis.prompts.search_query import PRIMARY_TERM_EXTRACTION, PRIMARY_TERM_EXTRACTION_CUSTOMFILTERS
        has_custom = custom_filters is not None and len(custom_filters) > 0
        if has_custom:
            def _fmt(cf):
                return f"{cf.name}: {cf.value}"
            custom_filters_text = "\n".join([_fmt(cf) for cf in custom_filters])
            outputs = call_llm_json_output(PRIMARY_TERM_EXTRACTION_CUSTOMFILTERS, { "research_topic": research_topic, "custom_filters_text": custom_filters_text }, llm=llm)
        else:
            outputs = call_llm_json_output(PRIMARY_TERM_EXTRACTION, { "research_topic": research_topic, "P": population, "I": intervention, "C": comparator, "O": outcome}, llm=llm)
        outputs = json.loads(outputs)
        terms = outputs.get("terms", [])
        return terms
    
    def _run_pubmed_id_search(self, terms):
        from trialmind.TrialMetaAnalysis.pubmed import ReqPubmedID
        req = ReqPubmedID()
        pmids = req.run(term=terms, retmax=7)
        return pmids
    
    def _run_pubmed_full_search(self, pmids):
        from trialmind.TrialMetaAnalysis.pubmed import ReqPubmedFull
        req = ReqPubmedFull()
        fetched_pubmed_data = req.run(pmids)
        pubmed_reference_text = '\n'.join(f"{idx+1}. {d['title']}\nAbstract: {d['abstract']}" 
                                          for idx, d in enumerate(fetched_pubmed_data))
        return pubmed_reference_text
    
    def _run_final_search_query_generation(self, research_topic, population, intervention, comparator, outcome, pubmed_reference_text, llm, custom_filters=None):
        from trialmind.TrialMetaAnalysis.prompts.search_query import SEARCH_TERM_EXTRACTION, SEARCH_TERM_EXTRACTION_CUSTOMFILTERS
        has_custom = custom_filters is not None and len(custom_filters) > 0
        if has_custom:
            def _fmt(cf):
                return f"{cf.name}: {cf.value}"
            custom_filters_text = "\n".join([_fmt(cf) for cf in custom_filters])
            payload = {"research_topic": research_topic, "custom_filters_text": custom_filters_text, "pubmed_reference_text": pubmed_reference_text}
            prompt = SEARCH_TERM_EXTRACTION_CUSTOMFILTERS
        else:
            payload = {"research_topic": research_topic, "P": population, "I": intervention, "C": comparator, "O": outcome, "pubmed_reference_text": pubmed_reference_text}
            prompt = SEARCH_TERM_EXTRACTION
        outputs = call_llm_json_output(
            prompt, 
            payload, 
            llm=llm,
            temperature=0.01,
            max_completion_tokens=1024
            )
        outputs = json.loads(outputs)
        logging.info(f"Final search query: {outputs}")

        # get the terms
        core_conditions = outputs.get("step 2", {}).get("CORE_CONDITIONS", [])
        core_treatments = outputs.get("step 2", {}).get("CORE_TREATMENTS", [])

        expand_conditions = outputs.get("step 3", {}).get("EXPAND_CONDITIONS", [])
        expand_treatments = outputs.get("step 3", {}).get("EXPAND_TREATMENTS", [])

        conditions = list(set(core_conditions + expand_conditions))
        treatments = list(set(core_treatments + expand_treatments))

        return {
            "conditions": conditions,
            "treatments": treatments,
        }

    def _run_user_request_search_query_generation(self, user_request, llm):
        from trialmind.TrialMetaAnalysis.prompts.search_query import USER_REQUEST_SEARCH_QUERY_GENERATION
        outputs = call_llm_json_output(
            USER_REQUEST_SEARCH_QUERY_GENERATION, 
            {"user_request": user_request}, 
            llm=llm,
            temperature=0.01,
            max_completion_tokens=1024
            )
        outputs = json.loads(outputs)
        conditions = outputs.get("conditions", [])
        treatments = outputs.get("treatments", [])
        return {
            "conditions": conditions,
            "treatments": treatments,
        }

    def _run_category_specific_search_query_generation(self, research_topic, population, intervention, comparator, outcome, user_request, category_name, llm, custom_filters=None):
        """
        Generate search terms specific to a given category.
        Returns a list of terms rather than a dictionary.
        """
        if len(user_request) == 0:
            # Use the normal pipeline but focus on the specific category
            terms = self._run_init_term_generation(research_topic, population, intervention, comparator, outcome, llm=llm, custom_filters=custom_filters)
            terms = 'AND'.join([f'({k})' for k in terms])
            logging.info(f"Generate initial search terms: {terms}")

            pmids = self._run_pubmed_id_search(terms)
            logging.info(f"Fetch initial reference pubmed paper ids {pmids}")
            pubmed_reference_text = self._run_pubmed_full_search(pmids)

            # Generate category-specific search terms
            return self._run_category_specific_final_search_query_generation(research_topic, population, intervention, comparator, outcome, pubmed_reference_text, category_name, llm=llm, custom_filters=custom_filters)
        else:
            # Use user request but focus on the specific category
            return self._run_category_specific_user_request_search_query_generation(user_request, category_name, llm=llm)

    def _run_category_specific_final_search_query_generation(self, research_topic, population, intervention, comparator, outcome, pubmed_reference_text, category_name, llm, custom_filters=None):
        """
        Generate category-specific search terms using the reference papers.
        Returns a list of terms for the specified category.
        """
        from trialmind.TrialMetaAnalysis.prompts.search_query import CATEGORY_SPECIFIC_SEARCH_TERM_EXTRACTION, CATEGORY_SPECIFIC_SEARCH_TERM_EXTRACTION_CUSTOMFILTERS
        has_custom = custom_filters is not None and len(custom_filters) > 0
        if has_custom:
            def _fmt(cf):
                return f"{cf.name}: {cf.value}"
            custom_filters_text = "\n".join([_fmt(cf) for cf in custom_filters])
            payload = {
                "research_topic": research_topic,
                "custom_filters_text": custom_filters_text,
                "pubmed_reference_text": pubmed_reference_text,
                "category_name": category_name.upper()
            }
            prompt = CATEGORY_SPECIFIC_SEARCH_TERM_EXTRACTION_CUSTOMFILTERS
        else:
            payload = {
                "research_topic": research_topic, 
                "P": population, 
                "I": intervention, 
                "C": comparator, 
                "O": outcome, 
                "pubmed_reference_text": pubmed_reference_text,
                "category_name": category_name.upper()
            }
            prompt = CATEGORY_SPECIFIC_SEARCH_TERM_EXTRACTION
        outputs = call_llm_json_output(
            prompt,
            payload,
            llm=llm,
            temperature=0.01,
            max_completion_tokens=1024
        )
        outputs = json.loads(outputs)
        logging.info(f"Category-specific search query for {category_name}: {outputs}")

        # Extract terms for the specific category
        core_terms = outputs.get("step 2", {}).get(f"CORE_{category_name.upper()}", [])
        expand_terms = outputs.get("step 3", {}).get(f"EXPAND_{category_name.upper()}", [])
        
        # Combine and deduplicate terms
        terms = list(set(core_terms + expand_terms))
        return terms

    def _run_category_specific_user_request_search_query_generation(self, user_request, category_name, llm):
        """
        Generate category-specific search terms based on user request.
        Returns a list of terms for the specified category.
        """
        from trialmind.TrialMetaAnalysis.prompts.search_query import CATEGORY_SPECIFIC_USER_REQUEST_SEARCH_QUERY_GENERATION
        outputs = call_llm_json_output(
            CATEGORY_SPECIFIC_USER_REQUEST_SEARCH_QUERY_GENERATION, 
            {
                "user_request": user_request,
                "category_name": category_name.upper()
            }, 
            llm=llm,
            temperature=0.01,
            max_completion_tokens=1024
        )
        outputs = json.loads(outputs)
        terms = outputs.get("terms", [])
        return terms

    def _run_category_determination(self, research_topic, population, intervention, comparator, outcome, llm, custom_filters=None):
        """
        Determine the most appropriate search categories based on the research topic and parameters.
        Returns a list of category names.
        """
        from trialmind.TrialMetaAnalysis.prompts.search_query import CATEGORY_DETERMINATION, CATEGORY_DETERMINATION_CUSTOMFILTERS
        has_custom = custom_filters is not None and len(custom_filters) > 0
        if has_custom:
            def _fmt(cf):
                return f"{cf.name}: {cf.value}"
            custom_filters_text = "\n".join([_fmt(cf) for cf in custom_filters])
            outputs = call_llm_json_output(
                CATEGORY_DETERMINATION_CUSTOMFILTERS,
                {
                    "research_topic": research_topic,
                    "custom_filters_text": custom_filters_text
                },
                llm=llm,
                temperature=0.01,
                max_completion_tokens=512
            )
        else:
            outputs = call_llm_json_output(
                CATEGORY_DETERMINATION,
                {
                    "research_topic": research_topic,
                    "P": population,
                    "I": intervention,
                    "C": comparator,
                    "O": outcome
                },
                llm=llm,
                temperature=0.01,
                max_completion_tokens=512
            )
        outputs = json.loads(outputs)
        categories = outputs.get("categories", ["conditions", "treatments"])
        reasoning = outputs.get("reasoning", "")
        logging.info(f"Determined categories: {categories}. Reasoning: {reasoning}")
        return categories

    def _run_dynamic_final_search_query_generation(self, research_topic, population, intervention, comparator, outcome, pubmed_reference_text, categories, llm, custom_filters=None):
        """
        Generate search terms dynamically for the determined categories.
        Returns a dictionary with category names as keys and lists of terms as values.
        """
        from trialmind.TrialMetaAnalysis.prompts.search_query import DYNAMIC_SEARCH_TERM_EXTRACTION, DYNAMIC_SEARCH_TERM_EXTRACTION_CUSTOMFILTERS
        has_custom = custom_filters is not None and len(custom_filters) > 0
        
        # Format categories for the prompt
        categories_str = ", ".join(categories)
        
        if has_custom:
            def _fmt(cf):
                return f"{cf.name}: {cf.value}"
            custom_filters_text = "\n".join([_fmt(cf) for cf in custom_filters])
            payload = {
                "research_topic": research_topic,
                "custom_filters_text": custom_filters_text,
                "pubmed_reference_text": pubmed_reference_text,
                "categories": categories_str
            }
            prompt = DYNAMIC_SEARCH_TERM_EXTRACTION_CUSTOMFILTERS
        else:
            payload = {
                "research_topic": research_topic,
                "P": population,
                "I": intervention,
                "C": comparator,
                "O": outcome,
                "pubmed_reference_text": pubmed_reference_text,
                "categories": categories_str
            }
            prompt = DYNAMIC_SEARCH_TERM_EXTRACTION
        
        outputs = call_llm_json_output(
            prompt,
            payload,
            llm=llm,
            temperature=0.01,
            max_completion_tokens=2048
        )
        outputs = json.loads(outputs)
        logging.info(f"Dynamic search query generation output: {outputs}")
        
        # Extract and combine terms for each category
        result = {}
        for category in categories:
            core_terms = outputs.get(category, {}).get("step 2", {}).get("CORE", [])
            expand_terms = outputs.get(category, {}).get("step 3", {}).get("EXPANDED", [])
            result[category] = list(set(core_terms + expand_terms))
        
        return result

    def _run_dynamic_user_request_search_query_generation(self, user_request, categories, llm):
        """
        Generate search terms dynamically for the determined categories based on user request.
        Returns a dictionary with category names as keys and lists of terms as values.
        """
        from trialmind.TrialMetaAnalysis.prompts.search_query import DYNAMIC_USER_REQUEST_SEARCH_QUERY_GENERATION
        
        # Format categories for the prompt
        categories_str = ", ".join(categories)
        
        outputs = call_llm_json_output(
            DYNAMIC_USER_REQUEST_SEARCH_QUERY_GENERATION,
            {
                "user_request": user_request,
                "categories": categories_str
            },
            llm=llm,
            temperature=0.01,
            max_completion_tokens=1024
        )
        outputs = json.loads(outputs)
        
        # Ensure all categories have entries (even if empty)
        result = {}
        for category in categories:
            result[category] = outputs.get(category, [])
        
        return result


class ScreeningCriteriaGeneration:
    """
    Input the user's input research question, generate the screening criteria for the screening clinical studies.

    Args:
        population (str): The population of the research question.
        intervention (str): The intervention of the research question.
        comparator (str): The comparator of the research question.
        outcome (str): The outcome of the research question.
        num_title_criteria (int): The number of title criteria to generate. Default is 3.
        num_abstract_criteria (int): The number of abstract criteria to generate. Default is 3.
        llm (str): The language model to use for the screening criteria generation. Default is "gpt-4o".
    """
    # TODO: https://training.cochrane.org/handbook/current/chapter-03
    # refer to the Cochrane Handbook for Systematic Reviews of Interventions
    # to draft the prompt for making the screening criteria
    def __init__(self):
        pass

    def run(
        self,
        research_topic: str,
        population: str,
        intervention: str,
        comparator: str,
        outcome: str,
        custom_filters: list[CustomFilter],
        num_title_criteria: int=3,
        num_abstract_criteria: int=3,
        llm: str="gpt-4o"
        ):
        from trialmind.TrialMetaAnalysis.prompts.screen_criteria import SCREENING_CRITERIA_GENERATION, SCREENING_CRITERIA_GENERATION_CUSTOMFILTERS
        has_custom = custom_filters is not None and len(custom_filters) > 0
        if has_custom:
            def _fmt(cf):
                return f"- **{cf.name}:** {cf.value}"
            custom_filters_text = "\n".join([_fmt(cf) for cf in custom_filters])
            prompt = SCREENING_CRITERIA_GENERATION_CUSTOMFILTERS
            payload = {
                "research_topic": research_topic,
                "custom_filters_text": custom_filters_text,
                "num_eligibility_criteria": num_title_criteria+num_abstract_criteria,
            }
        else:
            prompt = SCREENING_CRITERIA_GENERATION
            payload = {
                "research_topic": research_topic,
                "P": population,
                "I": intervention,
                "C": comparator,
                "O": outcome,
                "num_eligibility_criteria": num_title_criteria+num_abstract_criteria,
            }
        outputs = call_llm_json_output(
            prompt,
            payload,
            llm=llm,
            temperature=0.01,
            max_completion_tokens=1024
            )
        outputs = json.loads(outputs)
        eligibility_criteria = outputs.get("ELIGIBILITY_CRITERIA", [])
        eligibility_analysis = outputs.get("ELIGIBILITY_ANALYSIS", [])
        return {
            "criteria": eligibility_criteria, # list of criteria
            "eligibility_analysis": eligibility_analysis
        }
