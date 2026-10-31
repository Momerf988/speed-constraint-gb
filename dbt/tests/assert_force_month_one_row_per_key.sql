-- The panel needs exactly one row per force, month and road group. A duplicate would count a
-- month twice in the regression.

select month, police_force_code, road_group, count(*) as rows
from {{ ref('agg_casualties_force_month') }}
group by 1, 2, 3
having count(*) > 1
