-- TRA0204 (by vehicle type) and TRA0202 (all motor vehicles) are two cuts of the same DfT
-- traffic estimate, so each road category's total must agree in every year both cover.
-- A roads are two rounded rows in TRA0204 (trunk and principal), so the totals can differ by
-- up to 2 x 0.05 + 0.05 = 0.15 billion vehicle-km.

with tra0204 as (
    select year, traffic_road_category, sum(billion_vehicle_km) as billion_vehicle_km
    from {{ ref('stg_traffic_vehicle_road_class') }}
    group by 1, 2
),

tra0204_totals as (
    select
        year,
        case
            when road_type = 'Motorway' then 'Motorway'
            when road_type = '''A'' Road' and rural_urban = 'Rural' then 'Rural A road'
            when road_type = '''A'' Road' and rural_urban = 'Urban' then 'Urban A road'
            when road_type = 'Minor Road' and rural_urban = 'Rural' then 'Rural minor road'
            when road_type = 'Minor Road' and rural_urban = 'Urban' then 'Urban minor road'
        end                                                     as traffic_road_category,
        sum(billion_vehicle_km)                                 as all_motor_vehicles
    from {{ source('raw', 'traffic_tra0204') }}
    where vehicle = 'All Motor Vehicles'
      and road_type <> 'All Roads'
    group by 1, 2
)

select t.year, t.traffic_road_category, t.all_motor_vehicles, r.billion_vehicle_km as tra0202
from tra0204_totals as t
join {{ ref('stg_traffic_road_class') }} as r
  on r.year = t.year
 and r.traffic_road_category = t.traffic_road_category
where abs(t.all_motor_vehicles - r.billion_vehicle_km) > 0.15
