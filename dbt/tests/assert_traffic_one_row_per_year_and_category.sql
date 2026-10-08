-- The join in Part B needs exactly one traffic figure per year and road category.
-- A duplicate here would fan out every casualty count it touches.

select year, traffic_road_category, count(*) as rows
from {{ ref('stg_traffic_road_class') }}
group by 1, 2
having count(*) > 1