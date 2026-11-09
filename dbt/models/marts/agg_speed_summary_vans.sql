-- Average free-flow speed of vans and cars per year and road type, in km/h. On national-limit
-- single carriageways this is all DfT publishes for vans, so the model moves the car
-- distribution by the difference between the two averages.

select
    v.year,
    v.road_type,
    (v.avg_free_flow_mph * 1.609344)::double precision           as van_avg_free_flow_kmh,
    (c.avg_free_flow_mph * 1.609344)::double precision           as car_avg_free_flow_kmh,
    v.pct_exceeding::double precision                            as van_pct_exceeding,
    v.observed_thousands::double precision                       as van_observed_thousands
from {{ ref('stg_speed_summary_vans') }} as v
join {{ ref('stg_speed_summary_cars') }} as c
  on c.year = v.year
 and c.road_type = v.road_type
