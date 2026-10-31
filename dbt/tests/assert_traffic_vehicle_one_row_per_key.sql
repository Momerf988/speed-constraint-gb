-- agg_traffic_segment sums this view by year, category and vehicle type. A duplicate here
-- would double that traffic, and every time cost built on it.

select year, traffic_road_category, vehicle_type, count(*) as rows
from {{ ref('stg_traffic_vehicle_road_class') }}
group by 1, 2, 3
having count(*) > 1
