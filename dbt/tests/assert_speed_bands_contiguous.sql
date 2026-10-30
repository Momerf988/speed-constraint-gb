-- Each band must start exactly where the previous one ends: no gaps, no overlaps.
-- lag() looks back one row in the window, here the previous band of the same year and road type.

select *
from (
    select
        year,
        road_type,
        band_lower_mph,
        lag(band_upper_mph) over (partition by year, road_type order by band_lower_mph) as previous_upper_mph
    from {{ ref('stg_speed_bands_cars') }}
) as b
where previous_upper_mph is not null
  and previous_upper_mph <> band_lower_mph