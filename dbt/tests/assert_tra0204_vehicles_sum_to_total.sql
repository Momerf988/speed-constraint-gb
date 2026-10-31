-- In every TRA0204 row group, the five vehicle types must add up to DfT's "All Motor Vehicles"
-- row. Each figure is rounded to 0.1 billion vehicle-km, so five parts can differ from the
-- rounded total by at most 5 x 0.05 + 0.05 = 0.3. A larger gap means a misread row.

with parts as (
    select road_type, road_management, rural_urban, year, sum(billion_vehicle_km) as sum_of_parts
    from {{ source('raw', 'traffic_tra0204') }}
    where vehicle <> 'All Motor Vehicles'
    group by 1, 2, 3, 4
),

totals as (
    select road_type, road_management, rural_urban, year, billion_vehicle_km as total
    from {{ source('raw', 'traffic_tra0204') }}
    where vehicle = 'All Motor Vehicles'
)

select p.*, t.total
from parts as p
join totals as t using (road_type, road_management, rural_urban, year)
where abs(p.sum_of_parts - t.total) > 0.3
