-- The model assumes DfT publishes van bands on every road type except national-limit single
-- carriageways. If DfT starts publishing them there, this fails, and the model should use them
-- instead of the car distribution moved to the van average.

select year, road_type, count(*) as bands
from {{ ref('stg_speed_bands_vans') }}
where road_type = 'National Speed Limit Single Carriageways'
group by 1, 2

union all

select s.year, s.road_type, 0
from {{ ref('stg_speed_summary_vans') }} as s
left join {{ ref('stg_speed_bands_vans') }} as b
  on b.year = s.year
 and b.road_type = s.road_type
where s.road_type <> 'National Speed Limit Single Carriageways'
  and b.year is null
