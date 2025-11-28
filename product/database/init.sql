DROP SCHEMA public;

CREATE SCHEMA public AUTHORIZATION postgres;
-- public."user" определение

-- Drop table

-- DROP TABLE public."user";

CREATE TABLE public."user" (
	user_id int8 GENERATED ALWAYS AS IDENTITY NOT NULL,
	"name" varchar NOT NULL,
	CONSTRAINT user_pk PRIMARY KEY (user_id)
);
