-- The road type -> speed limit mapping is written by hand in stg_speed_bands_cars. Where DfT
-- states the limit itself (motorways, national speed limit roads), the two must agree.

select distinct b.year, b.road_type, b.speed_limit_mph, s.published_speed_limit_mph
from {{ ref('stg_speed_bands_cars') }} as b
join {{ ref('stg_speed_summary_cars') }} as s
  on s.year = b.year
 and s.road_type = b.road_type
where s.published_speed_limit_mph is not null
  and s.published_speed_limit_mph <> b.speed_limit_mph