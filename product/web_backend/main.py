import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from api.routes import router
from services.codec.json_codec import JsonCodec
from services.config.config import Config
from services.kafka.producer import Producer
from services.logger.logger import logger
from services.postgres.session import PostgresSession


def main():
    config = Config.from_env()
    
    codec = JsonCodec()
    kafka_producer = Producer(config.kafka, codec)
    
    postgres_session = PostgresSession(config=config.database)
    
    app = FastAPI(
        title=config.app.title,
        openapi_url=f'{config.app.openapi_str}/openapi.json'        
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.add_event_handler('startup', _startup_handler(app, config, kafka_producer, postgres_session))

    uvicorn.run(
        app,
        host=config.app.host,
        port=config.app.port
    )
    

def _startup_handler(
    app: FastAPI, 
    config: Config, 
    producer: Producer,
    session: PostgresSession
):
    async def startup() -> None:
        logger.info("Running startup handler.")
        await _startup(app, config, producer, session)
    return startup


async def _startup(
    app: FastAPI, 
    config: Config, 
    producer: Producer, 
    session: PostgresSession
):
    app.state.config = config
    app.state.producer = producer
    app.state.postgres_session = await session.create()


if __name__ == '__main__':
    main()
