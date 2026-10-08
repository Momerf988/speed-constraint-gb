-- DfT traffic (TRA0202) in long form: one row per year and road category.
-- Category names match int_collision_enriched.traffic_road_category exactly,
-- because Part B joins the two on year and category.
-- '[no notes]' becomes null; the remaining notes flag years DfT says need caution
-- (2000 fuel protest, 2001 foot and mouth, 2020-2022 COVID).

with source as (
    select * from {{ source('raw', 'traffic_tra0202') }}
)

select year, nullif(notes, '[no notes]') as notes, 'Motorway' as traffic_road_category, motorway as billion_vehicle_km
from source
union all
select year, nullif(notes, '[no notes]'), 'Urban A road', urban_a_roads
from source
union all
select year, nullif(notes, '[no notes]'), 'Urban minor road', urban_minor_roads
from source
union all
select year, nullif(notes, '[no notes]'), 'Rural A road', rural_a_roads
from source
union all
select year, nullif(notes, '[no notes]'), 'Rural minor road', rural_minor_roads
from source