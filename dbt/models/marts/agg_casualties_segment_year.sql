-- Casualties per year on the four road types that have DfT free-flow speed distributions
-- (the "modelled segments", defined once in macros/modelled_segment.sql), and how many were
-- in collisions involving a car, taxi or van: the vehicles a speed limiter acts on. Every
-- other road is grouped as "not modelled", so the share of harm the casualty model covers
-- stays visible.

with car_van_collisions as (
    select distinct collision_index
    from {{ ref('stg_vehicle') }}
    where collision_year >= 2010
      and vehicle_type_code in (8, 9, 19)   -- taxi or private hire car, car, van up to 3.5 t
),

casualties as (
    select
        f.collision_year                                        as year,
        {{ modelled_segment('r') }}                             as segment,
        f.fatal,
        f.ksi_adjusted,
        v.collision_index is not null                           as car_van_involved
    from {{ ref('fct_casualty') }} as f
    join {{ ref('dim_road') }} as r
      on r.road_key = f.road_key
    left join car_van_collisions as v
      on v.collision_index = f.collision_index
    where f.collision_year >= 2010
)

select
    year,
    segment,
    count(*)                                                    as casualties,
    sum(fatal)                                                  as killed,
    sum(ksi_adjusted)                                           as ksi_adjusted,
    count(*) filter (where car_van_involved)                    as casualties_car_van,
    sum(fatal) filter (where car_van_involved)                  as killed_car_van,
    sum(ksi_adjusted) filter (where car_van_involved)           as ksi_adjusted_car_van
from casualties
group by 1, 2
