# API's for literature review product
from fastapi import (
    APIRouter,
    Depends,
    Header,
    BackgroundTasks,
    Request,
)
from trialmind import (
    AWS_datasets_ACCESS_KEY_ID,
    AWS_datasets_SECRET_ACCESS_KEY,
    AWS_datasets_DEFAULT_REGION,
    SLR_USER_DOCUMENTS_S3_BUCKET,
    TEXTRACT_SLR_SNS_ROLE_ARN,
    TEXTRACT_SLR_SNS_TOPIC_ARN,
    TEXTRACT_SLR_SQS_QUEUE_URL,
)
from trialmind.api.routes.authorization import check_token
from trialmind.api.permissions import check_user_permissions_and_bind
from trialmind.api.utils import Module, get_route_path
from trialmind.api.data_models import MASearchQueryGenerationBody, MARetrievePapersBody, MACriteriaGenerationBody, MAPICOGenerationBody, MAFilterGenerationBody, SearchQueryInputWrapper
from trialmind.api.shared_apis import (
    ma_pico_generation,
    ma_filter_generation,
    ma_search_query_generation,
    ma_retrieve_papers,
    ma_criteria_generation,
    get_retrieve_papers_status
)
from trialmind.LiteratureSearch.DocumentParsingService.DocumentParsingService import DocumentParsingService
from trialmind.api.rate_limit import set_rate_limit_category

router = APIRouter(prefix="", tags=["literature-review"])


documentParsingService = DocumentParsingService(
    s3_bucket_name=SLR_USER_DOCUMENTS_S3_BUCKET,
    aws_access_key_id=AWS_datasets_ACCESS_KEY_ID, # same as protocol vanilla KB
    aws_secret_access_key=AWS_datasets_SECRET_ACCESS_KEY, # same as protocol vanilla KB
    aws_default_region=AWS_datasets_DEFAULT_REGION,
    textract_sns_role_arn=TEXTRACT_SLR_SNS_ROLE_ARN,
    textract_sns_topic_arn=TEXTRACT_SLR_SNS_TOPIC_ARN,
    textract_sqs_queue_url=TEXTRACT_SLR_SQS_QUEUE_URL,
)

ALLOWED_MODULES = [Module.LR]

@router.post("/ma_pico_generation", dependencies=[Depends(check_token), Depends(set_rate_limit_category("llm"))])
async def ma_pico_generation_route(
    request: Request,
    request_body: MAPICOGenerationBody,
    provider: Annotated[str | None, Header()] = None,
    access_token: Annotated[str | None, Header()] = None,
    route: str = Depends(get_route_path),
):
    """Generate PICO elements for meta-analysis"""
    user_id, organizations, groups, modules, tier, credit_cost, error_response = await check_user_permissions_and_bind(request, provider, access_token, ALLOWED_MODULES, route, request_body.dict())
    if error_response is not  None: return error_response
    return await ma_pico_generation(request_body)


@router.post("/ma_filter_generation", dependencies=[Depends(check_token), Depends(set_rate_limit_category("llm"))])
async def ma_filter_generation_route(
    request: Request,
    request_body: MAFilterGenerationBody,
    provider: Annotated[str | None, Header()] = None,
    access_token: Annotated[str | None, Header()] = None,
    route: str = Depends(get_route_path),
):
    """Generate filter elements for various research frameworks (PICOT, PICOS, SPIDER, SPICE, ECLIPSE)"""
    user_id, organizations, groups, modules, tier, credit_cost, error_response = await check_user_permissions_and_bind(request, provider, access_token, ALLOWED_MODULES, route, request_body.dict())
    if error_response is not  None: return error_response
    return await ma_filter_generation(request_body)


@router.post("/ma_search_query_generation", dependencies=[Depends(check_token), Depends(set_rate_limit_category("llm"))])
async def ma_search_query_generation_route(
    request: Request,
    request_body: MASearchQueryGenerationBody,
    provider: Annotated[str | None, Header()] = None,
    access_token: Annotated[str | None, Header()] = None,
    route: str = Depends(get_route_path),
):
    """Generate search query for meta-analysis"""
    user_id, organizations, groups, modules, tier, credit_cost, error_response = await check_user_permissions_and_bind(request, provider, access_token, ALLOWED_MODULES, route, request_body.dict())
    if error_response is not  None: return error_response
    return await ma_search_query_generation(request_body)

@router.post("/ma_retrieve_papers", dependencies=[Depends(check_token), Depends(set_rate_limit_category("external-api"))])
async def ma_retrieve_papers_route(
    request: Request,
    request_body: MARetrievePapersBody,
    background_tasks: BackgroundTasks,
    provider: Annotated[str | None, Header()] = None,
    access_token: Annotated[str | None, Header()] = None,
    route: str = Depends(get_route_path),
):
    """Retrieve papers for meta-analysis"""
    user_id, organizations, groups, modules, tier, credit_cost, error_response = await check_user_permissions_and_bind(request, provider, access_token, ALLOWED_MODULES, route, request_body.dict())
    if error_response is not  None: return error_response
    return await ma_retrieve_papers(request_body, background_tasks)

@router.get("/ma_retrieve_papers/{task_id}", dependencies=[Depends(check_token)])
async def get_retrieve_papers_status_route(
    request: Request,
    task_id: str,
    provider: Annotated[str | None, Header()] = None,
    access_token: Annotated[str | None, Header()] = None,
    route: str = Depends(get_route_path),
):
    """Get retrieve papers status for meta-analysis"""
    user_id, organizations, groups, modules, tier, credit_cost, error_response = await check_user_permissions_and_bind(request, provider, access_token, ALLOWED_MODULES, route, None)
    if error_response is not  None: return error_response
    return await get_retrieve_papers_status(task_id)

@router.post("/ma_criteria_generation", dependencies=[Depends(check_token), Depends(set_rate_limit_category("llm"))])
async def ma_criteria_generation_route(
    request: Request,
    request_body: MACriteriaGenerationBody,
    provider: Annotated[str | None, Header()] = None,
    access_token: Annotated[str | None, Header()] = None,
    route: str = Depends(get_route_path),
):
    """Generate criteria for meta-analysis"""
    user_id, organizations, groups, modules, tier, credit_cost, error_response = await check_user_permissions_and_bind(request, provider, access_token, ALLOWED_MODULES, route, request_body.dict())
    if error_response is not  None: return error_response
    return await ma_criteria_generation(request_body)
