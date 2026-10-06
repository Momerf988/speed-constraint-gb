-- One row per calendar day, for every full year in the data.
-- Generated rather than loaded: a date dimension gives every fact table the same
-- calendar attributes (year, month, weekday) without recalculating them each time.

with bounds as (
    select
        date_trunc('year', min(collision_date))::date                          as first_day,
        (date_trunc('year', max(collision_date)) + interval '1 year - 1 day')::date as last_day
    from {{ ref('stg_collision') }}
),

days as (
    select generate_series(first_day, last_day, interval '1 day')::date as calendar_date
    from bounds
)

select
    to_char(calendar_date, 'YYYYMMDD')::int             as date_key,
    calendar_date,
    extract(year from calendar_date)::smallint          as year,
    extract(quarter from calendar_date)::smallint       as quarter,
    extract(month from calendar_date)::smallint         as month,
    to_char(calendar_date, 'FMMonth')                   as month_name,
    extract(day from calendar_date)::smallint           as day_of_month,
    extract(isodow from calendar_date)::smallint        as iso_day_of_week,  -- 1 = Monday
    to_char(calendar_date, 'FMDay')                     as day_name,
    extract(isodow from calendar_date) in (6, 7)        as is_weekend
from days