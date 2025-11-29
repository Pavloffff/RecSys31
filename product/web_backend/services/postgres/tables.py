from sqlalchemy import BigInteger, Numeric, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class BaseTable(DeclarativeBase):
    pass


class User(BaseTable):
    __tablename__ = 'user'
    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String)


class Brand(BaseTable):
    __tablename__ = 'brand'
    brand_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String)


class MarketplaceItem(BaseTable):
    __tablename__ = 'marketplace_item'
    item_id: Mapped[str] = mapped_column(String, primary_key=True)
    category: Mapped[str] = mapped_column(String)
    brand_id: Mapped[int] = mapped_column(ForeignKey('brand.brand_id'))
    price: Mapped[float] = mapped_column(Numeric(10, 2))


class RetailItem(BaseTable):
    __tablename__ = 'retail_item'
    item_id: Mapped[str] = mapped_column(String, primary_key=True)
    category: Mapped[str] = mapped_column(String)
    brand_id: Mapped[int] = mapped_column(ForeignKey('brand.brand_id'))
    price: Mapped[float] = mapped_column(Numeric(10, 2))


class OffersItem(BaseTable):
    __tablename__ = 'offers_item'
    item_id: Mapped[str] = mapped_column(String, primary_key=True)
    category: Mapped[str] = mapped_column(String)
    brand_id: Mapped[int] = mapped_column(ForeignKey('brand.brand_id'))
    price: Mapped[float] = mapped_column(Numeric(10, 2))


class Event(BaseTable):
    __tablename__ = 'event'
    event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('user.user_id'))
    item_id: Mapped[str] = mapped_column(String)
    channel: Mapped[str] = mapped_column(String(50))
    action_type: Mapped[str] = mapped_column(String(50))
    timestamp: Mapped[int] = mapped_column(BigInteger)
    price: Mapped[float] = mapped_column(Numeric(10, 2))
    category: Mapped[str] = mapped_column(String)
    brand_id: Mapped[int] = mapped_column(ForeignKey('brand.brand_id'))
