-- Splitting traffic across road types must not create or lose any: in every year, the car
-- and van traffic in agg_traffic_segment must add up to the staging totals for the five road
-- categories. Returns the years that differ.

with allocated as (
    select year, sum(car_billion_vehicle_km) as cars, sum(van_billion_vehicle_km) as vans
    from {{ ref('agg_traffic_segment') }}
    group by year
),

published as (
    select
        year,
        sum(billion_vehicle_km) filter (where vehicle_type = 'Cars and Taxis')              as cars,
        sum(billion_vehicle_km) filter (where vehicle_type = 'Light Commercial Vehicles')   as vans
    from {{ ref('stg_traffic_vehicle_road_class') }}
    where year in (select year from allocated)
    group by year
)

select p.year, p.cars, a.cars as allocated_cars, p.vans, a.vans as allocated_vans
from published as p
join allocated as a using (year)
where abs(p.cars - a.cars) > 0.001
   or abs(p.vans - a.vans) > 0.001
