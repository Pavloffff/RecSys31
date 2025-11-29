DROP SCHEMA public;

CREATE SCHEMA public AUTHORIZATION postgres;

-- Таблица пользователей
CREATE TABLE public."user" (
	user_id int8 NOT NULL,
	"name" varchar,
	CONSTRAINT user_pk PRIMARY KEY (user_id)
);

-- Таблица брендов
CREATE TABLE public.brand (
	brand_id int8 NOT NULL,
	"name" varchar,
	CONSTRAINT brand_pk PRIMARY KEY (brand_id)
);

-- Таблица товаров для marketplace
CREATE TABLE public.marketplace_item (
	item_id varchar NOT NULL,
	category varchar,
	brand_id int8,
	price numeric(10, 2),
	CONSTRAINT marketplace_item_pk PRIMARY KEY (item_id),
	CONSTRAINT marketplace_item_brand_fk FOREIGN KEY (brand_id) REFERENCES public.brand(brand_id)
);

-- Таблица товаров для retail
CREATE TABLE public.retail_item (
	item_id varchar NOT NULL,
	category varchar,
	brand_id int8,
	price numeric(10, 2),
	CONSTRAINT retail_item_pk PRIMARY KEY (item_id),
	CONSTRAINT retail_item_brand_fk FOREIGN KEY (brand_id) REFERENCES public.brand(brand_id)
);

-- Таблица товаров для offers
CREATE TABLE public.offers_item (
	item_id varchar NOT NULL,
	category varchar,
	brand_id int8,
	price numeric(10, 2),
	CONSTRAINT offers_item_pk PRIMARY KEY (item_id),
	CONSTRAINT offers_item_brand_fk FOREIGN KEY (brand_id) REFERENCES public.brand(brand_id)
);

-- Таблица событий
CREATE TABLE public.event (
	event_id bigserial NOT NULL,
	user_id int8 NOT NULL,
	item_id varchar NOT NULL,
	channel varchar(50) NOT NULL,
	action_type varchar(50) NOT NULL,
	timestamp int8 NOT NULL,
	price numeric(10, 2),
	category varchar,
	brand_id int8,
	CONSTRAINT event_pk PRIMARY KEY (event_id),
	CONSTRAINT event_user_fk FOREIGN KEY (user_id) REFERENCES public."user"(user_id)
);

CREATE INDEX idx_event_user_id ON public.event(user_id);
CREATE INDEX idx_event_item_id ON public.event(item_id);
CREATE INDEX idx_event_channel ON public.event(channel);
CREATE INDEX idx_event_timestamp ON public.event(timestamp);
CREATE INDEX idx_event_user_channel ON public.event(user_id, channel);
CREATE INDEX idx_marketplace_item_category ON public.marketplace_item(category);
CREATE INDEX idx_retail_item_category ON public.retail_item(category);
CREATE INDEX idx_offers_item_category ON public.offers_item(category);