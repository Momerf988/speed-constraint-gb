-- Casualties per year and road environment: about 13,000 rows summarising 12 million.
-- Small enough to chart directly, and the table Phase 4 joins to traffic volumes.
-- Only answers questions about year and road; anything else goes back to fct_casualty.

select
    f.collision_year                as year,
    r.traffic_road_category,
    r.urban_rural,
    r.first_road_class,
    r.road_type,
    r.speed_limit_mph,
    r.speed_limit_kmh,
    r.is_standard_speed_limit,
    count(*)                        as casualties,
    sum(f.fatal)                    as killed,
    sum(f.serious_reported)         as serious_reported,
    sum(f.ksi_reported)             as ksi_reported,
    sum(f.serious_adjusted)         as serious_adjusted,
    sum(f.ksi_adjusted)             as ksi_adjusted
from {{ ref('fct_casualty') }} as f
join {{ ref('dim_road') }} as r
  on r.road_key = f.road_key
group by 1, 2, 3, 4, 5, 6, 7, 8