-- Collisions per casualty, by severity and year. TAG values police, insurance and property
-- damage per collision, not per casualty, so preventing a casualty also prevents this many
-- collisions' worth of those costs. A collision takes the severity of its worst casualty.
-- Reported (not adjusted) severity, because DfT's collision costs are built on it.

with collisions as (
    select collision_year as year, collision_severity as severity, count(*) as collisions
    from {{ ref('fct_collision') }}
    where collision_year >= 2010
    group by 1, 2
),

casualties as (
    select collision_year as year, casualty_severity as severity, count(*) as casualties
    from {{ ref('fct_casualty') }}
    where collision_year >= 2010
    group by 1, 2
)

select
    year,
    case severity when 1 then 'Fatal' when 2 then 'Serious' when 3 then 'Slight' end   as severity,
    collisions,
    casualties,
    collisions::double precision / casualties                  as collisions_per_casualty
from collisions
join casualties using (year, severity)
