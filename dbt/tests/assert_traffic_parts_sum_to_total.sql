-- The five road categories must add up to DfT's own "All roads" total.
-- Each published figure is rounded to 0.1 billion vehicle-km, so five parts can differ
-- from the rounded total by at most 5 x 0.05 + 0.05 = 0.3. A larger gap is not rounding:
-- it means a column was misread or the table changed.

select
    year,
    motorway + rural_a_roads + urban_a_roads + rural_minor_roads + urban_minor_roads as sum_of_parts,
    all_roads
from {{ source('raw', 'traffic_tra0202') }}
where abs(motorway + rural_a_roads + urban_a_roads + rural_minor_roads + urban_minor_roads - all_roads) > 0.3