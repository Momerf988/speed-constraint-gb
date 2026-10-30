-- Documents a known defect in DfT's file: the motorway band printed "60 to 54" (every year,
-- 2018-2024) is really 60 to 64. Bands are built from their lower bound, so the data is
-- unaffected; this test reports the bad labels as a warning rather than blocking the build.

{{ config(severity='warn') }}

select year, road_type, band_label
from {{ ref('stg_speed_bands_cars') }}
where printed_upper_mph is not null
  and printed_upper_mph <> band_lower_mph + 4