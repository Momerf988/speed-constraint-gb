-- Free-flow van speed distributions by year, road type and 5 mph band, where DfT publishes them:
-- 32 km/h roads, 48 km/h roads and motorways. Same columns as agg_speed_bands_cars, so the same
-- Python code turns both into speed distributions.

select
    b.year,
    b.road_type,
    b.speed_limit_mph,
    round(b.speed_limit_mph * 1.609344)::smallint                as speed_limit_kmh,
    b.band_lower_mph,
    b.band_upper_mph,
    (b.band_lower_mph * 1.609344)::double precision              as band_lower_kmh,
    (b.band_upper_mph * 1.609344)::double precision              as band_upper_kmh,
    b.share_pct::double precision                                as share_pct,
    b.is_low,
    s.pct_exceeding::double precision                            as pct_exceeding,
    s.avg_free_flow_mph::double precision                        as avg_free_flow_mph,
    (s.avg_free_flow_mph * 1.609344)::double precision           as avg_free_flow_kmh,
    s.observed_thousands::double precision                       as observed_thousands
from {{ ref('stg_speed_bands_vans') }} as b
join {{ ref('stg_speed_summary_vans') }} as s
  on s.year = b.year
 and s.road_type = b.road_type
