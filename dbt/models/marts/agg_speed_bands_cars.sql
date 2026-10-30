-- Free-flow car speed distributions by year, road type and 5 mph band, with DfT's own
-- summary figures alongside: the input for fitting continuous speed distributions in Python.
-- Bands stay in mph, as published; km/h bounds are added for reporting.
-- DfT says its sites overstate speeding across the network, so treat these as an upper bound.

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
    s.pct_exceeding_by_5_mph::double precision                   as pct_exceeding_by_5_mph,
    s.pct_exceeding_by_10_mph::double precision                  as pct_exceeding_by_10_mph,
    s.avg_free_flow_mph::double precision                        as avg_free_flow_mph,
    (s.avg_free_flow_mph * 1.609344)::double precision           as avg_free_flow_kmh,
    s.observed_thousands::double precision                       as observed_thousands
from {{ ref('stg_speed_bands_cars') }} as b
join {{ ref('stg_speed_summary_cars') }} as s
  on s.year = b.year
 and s.road_type = b.road_type