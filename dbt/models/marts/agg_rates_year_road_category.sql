-- Casualty, death and KSI rates per billion vehicle-km, by year and road category.
-- Numerators come from agg_casualties_year_road; the denominator is DfT motor-vehicle
-- traffic (TRA0202). Both are kept, not just the rate, so any group of categories
-- (e.g. all rural roads) can be re-rated correctly: sum the numerators, sum the
-- denominators, then divide. Never average rates.
-- Starts in 1994, when STATS19 began recording urban/rural. The "Unknown" road category
-- has no traffic figure and is excluded (under 1% of casualties in every year, tested).
-- DfT's own published rates divide by traffic including pedal cycles, so they run
-- 1-3% lower than these; counts are reconciled exactly instead.

{% set first_year = var('urban_rural_first_year') %}

with casualties as (
    select
        year,
        traffic_road_category,
        sum(casualties)     as casualties,
        sum(killed)         as killed,
        sum(ksi_adjusted)   as ksi_adjusted
    from {{ ref('agg_casualties_year_road') }}
    where traffic_road_category <> 'Unknown'
      and year >= {{ first_year }}
    group by 1, 2
),

traffic as (
    select year, traffic_road_category, notes, billion_vehicle_km
    from {{ ref('stg_traffic_road_class') }}
)

select
    c.year,
    c.traffic_road_category,
    case
        when c.traffic_road_category = 'Motorway' then 'Motorway'
        when c.traffic_road_category like 'Urban%' then 'Urban'
        else 'Rural'
    end                                                                  as road_environment,
    t.notes,
    c.casualties,
    c.killed,
    c.ksi_adjusted,
    t.billion_vehicle_km,
    (c.casualties / t.billion_vehicle_km)::double precision              as casualties_per_bn_vkm,
    (c.killed / t.billion_vehicle_km)::double precision                  as killed_per_bn_vkm,
    (c.ksi_adjusted / t.billion_vehicle_km::double precision)            as ksi_adjusted_per_bn_vkm
from casualties as c
join traffic as t
  on t.year = c.year
 and t.traffic_road_category = c.traffic_road_category