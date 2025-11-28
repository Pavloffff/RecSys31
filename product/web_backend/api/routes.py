from fastapi import APIRouter, Depends, HTTPException, status
from starlette.requests import Request

from domain.yandexgpt.request import Request as YandexGptRequest
from services.logger.logger import logger
from services.kafka.producer import Producer

router = APIRouter(prefix='/llm')

@router.post('/post')
async def post_message_to_llm(
    request: Request, 
    yandex_gpt_request: YandexGptRequest,
):
    # try:
        producer: Producer = request.app.state.producer
        request_json = yandex_gpt_request.dict()
        
        logger.info(request_json)
        
        producer.produce(request_json, 'utf-8')
        return {
            "status": "success",
        }
    # except Exception as e:
    #     logger.error(f"Error processing request: {str(e)}")
    #     raise HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail="Internal server error"
    #     )
