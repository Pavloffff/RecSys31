from itertools import chain

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from domain.recsys.user_data import UserData, EventData
from services.postgres.tables import User, Brand, MarketplaceItem, RetailItem, OffersItem, Event


class Item(BaseModel):
    item_id: str
    category: str | None
    brand_id: int | None
    brand: str | None
    price: float | None
    

class GetUserData:
    @staticmethod
    async def get(session: AsyncSession, user_id: int) -> UserData | None:
        user_stmt = select(User).where(
            User.user_id == user_id
        )
        user_res = await session.execute(user_stmt)
        user = user_res.scalar_one_or_none()
        if user is None:
            return None
        
        events_stmt = select(Event).where(Event.user_id == user.user_id)
        events_res = await session.scalars(statement=events_stmt)
        events = events_res.all()
        
        events_data: list[EventData] = []
        for event in events:
            item_id = event.item_id
            
            offers_item_stmt = select(OffersItem).where(OffersItem.item_id == item_id)
            offers_item_res = await session.execute(statement=offers_item_stmt)
            offers_item = offers_item_res.scalar_one_or_none()
            
            retail_item_stmt = select(RetailItem).where(RetailItem.item_id == item_id)
            retail_item_res = await session.execute(statement=retail_item_stmt)
            retail_item = retail_item_res.scalar_one_or_none()
            
            marketplace_item_stmt = select(MarketplaceItem).where(MarketplaceItem.item_id == item_id)
            marketplace_item_res = await session.execute(statement=marketplace_item_stmt)
            marketplace_item = marketplace_item_res.scalar_one_or_none()
            
            item_data = None
            item_type_data = None
            for item, item_type in chain(
                (offers_item, 'offers'), 
                (retail_item, 'retail'), 
                (marketplace_item, 'marketplace')
            ):
                if item is not None:
                    brand_id = None
                    brand = None
                    if item.brand_id is not None:
                        brand_stmt = select(Brand).where(Brand.brand_id == item.brand_id)
                        brand_res = await session.execute(statement=brand_stmt)
                        brand_obj = brand_res.scalar_one_or_none()
                        brand_id = brand.brand_id
                        brand = brand_obj.name 
                    item_data = Item(
                        item_id=item.item_id,
                        brand_id=brand_id,
                        brand=brand,
                        category=item.category,
                        price=item.price
                    )
                    item_type_data = item_type
                    break
                
            brand_stmt = select(Brand).where(Brand.brand_id == event.brand_id)
            brand_res = await session.execute(statement=brand_stmt)
            brand_obj = brand_res.scalar_one_or_none()
            
            events_data.append(
                EventData(
                    event_id=event.event_id,
                    action_type=event.action_type,
                    item_type=item_type_data,
                    item_id=item_data.item_id if item_data is not None else None,
                    item_category=item_data.category if item_data is not None else None,
                    item_brand=item_data.brand if item_data is not None else None,
                    item_brand_id=item_data.brand_id if item_data is not None else None,
                    item_price=item_data.price if item_data is not None else None,
                    price=event.price,
                    brand_id=event.brand_id,
                    brand=brand_obj.name if brand_obj is not None else None,
                    category=event.category,
                    channel=event.channel,
                    timestamp=event.timestamp
                )
            )

        return UserData(
            user_id=user.user_id,
            name=user.name,
            events=events_data
        )