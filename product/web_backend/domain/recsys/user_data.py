from pydantic import BaseModel


class EventData(BaseModel):
    event_id: int
    item_type: str
    item_id: str
    item_category: str
    item_brand_id: int
    item_brand: str
    item_price: float
    channel: str
    action_type: str
    timestamp: int
    price: float | None
    category: str | None
    brand_id: int | None
    brand: str | None


class UserData(BaseModel):
    user_id: int
    name: str | None
    events: list[EventData]
