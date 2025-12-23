import pytest
from unittest.mock import patch, AsyncMock
from trialmind.api.shared_apis import ma_search_query_generation
from trialmind.api.data_models import MASearchQueryGenerationBody

@pytest.mark.asyncio
async def test_ma_search_query_generation_success():
    # 1. Prepare a mock request body
    request_data = MASearchQueryGenerationBody(
        research_topic="Impact of aspirin on heart disease",
        population="Adults with hypertension"
    )

    # 2. Mock the deep retrieval model call
    # We patch the actual function that calls the vLLM server
    with patch("zilal_contribution.app.query_service.query_generator.call_deepretrieval_model", 
               new_callable=AsyncMock) as mock_vllm:
        
        # Define what the "fake" model should return
        mock_vllm.return_value = "<answer>{\"final_query\": \"aspirin AND heart disease\"}</answer>"

        # 3. Call the API function
        response = await ma_search_query_generation(request_data)

        # 4. Assert the results
        assert response.status_code == 200
        content = response.body.decode()
        assert "aspirin AND heart disease" in content
        
        
@pytest.mark.asyncio
async def test_ma_search_query_generation_fallback():
    request_data = MASearchQueryGenerationBody(research_topic="Test topic")

    # Mock the vLLM to FAIL, forcing a fallback
    with patch("zilal_contribution.app.query_service.query_generator.generate_search_terms", 
               side_effect=Exception("vLLM Offline")):
        
        # Mock the legacy searchQueryGenAPI to succeed
        with patch("trialmind.api.shared_apis.searchQueryGenAPI.run") as mock_legacy:
            mock_legacy.return_value = {"Condition": ["Aspirin"]}

            response = await ma_search_query_generation(request_data)
            
            assert response.status_code == 200
            # Verify the response format matches your fallback normalization