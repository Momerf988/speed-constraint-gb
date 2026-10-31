-- Car and van traffic on each modelled road type, per year, in billion vehicle-km.
-- DfT publishes traffic by road class (TRA0204), not by speed limit. So each road class's
-- traffic is split across the modelled road types in proportion to that class's casualties
-- in the same year: if 32% of rural A road casualties are on 97 km/h single carriageways,
-- so is 32% of rural A road traffic.
-- Single carriageways have more casualties per km than dual carriageways, so this overstates
-- their traffic, and with it the time a cap costs on 97 km/h roads: the cautious direction
-- for judging a cap. The appraisal tests how much it matters.

with casualties as (
    select
        f.collision_year                                        as year,
        r.traffic_road_category,
        {{ modelled_segment('r') }}                             as segment,
        count(*)                                                as casualties
    from {{ ref('fct_casualty') }} as f
    join {{ ref('dim_road') }} as r
      on r.road_key = f.road_key
    where f.collision_year >= 2010
      and r.traffic_road_category <> 'Unknown'                 -- 38 of 396,729 casualties in 2022-2024
    group by 1, 2, 3
),

shares as (
    select
        year,
        traffic_road_category,
        segment,
        casualties::double precision
            / sum(casualties) over (partition by year, traffic_road_category)   as share
    from casualties
),

traffic as (
    select
        year,
        traffic_road_category,
        sum(billion_vehicle_km) filter (where vehicle_type = 'Cars and Taxis')              as cars,
        sum(billion_vehicle_km) filter (where vehicle_type = 'Light Commercial Vehicles')   as vans
    from {{ ref('stg_traffic_vehicle_road_class') }}
    group by 1, 2
)

select
    s.year,
    s.segment,
    sum(s.share * t.cars)                                       as car_billion_vehicle_km,
    sum(s.share * t.vans)                                       as van_billion_vehicle_km
from shares as s
join traffic as t
  on t.year = s.year
 and t.traffic_road_category = s.traffic_road_category
group by 1, 2
