from fastapi import APIRouter, Depends, HTTPException, status
from starlette.requests import Request

from domain.message.message import Message
from services.logger.logger import logger
from services.kafka.producer import Producer
from usecases.user_data.get_user_data import GetUserData

router = APIRouter(prefix='/recsys')


@router.get('/send/{user_id}')
async def send_recomend_request(
    request: Request, 
    user_id: int
):
    try:
        producer: Producer = request.app.state.producer
        
        async with request.app.state.postgres_session() as session:
            message = Message(user_id=user_id, context={})
            producer.produce(message.dict(), 'utf-8')

        return {
            "status": "success",
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# @router.post('/post')
# async def post_message_to_llm(
#     request: Request, 
#     yandex_gpt_request: YandexGptRequest,
# ):
#     try:
#         producer: Producer = request.app.state.producer
#         request_json = yandex_gpt_request.dict()
        
#         logger.info(request_json)
        
#         producer.produce(request_json, 'utf-8')
#         return {
#             "status": "success",
#         }
#     except Exception as e:
#         logger.error(f"Error processing request: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Internal server error"
#         )
