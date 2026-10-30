-- DfT free-flow car speed distributions (SPE0102) in long form: one row per year, road type
-- and 5 mph speed band. Bands are rebuilt from their lower bound, not read from the printed
-- label, because DfT's motorway label "60 to 54" is a typo for 60 to 64 (in every year).
-- "[low]" means a share too small to print but not zero: share_pct is null and is_low is true.

with source as (
    select * from {{ source('raw', 'speed_spe0102') }}
)

select
    year,
    road_type,
    case road_type
        when 'Motorways' then 70
        when 'National Speed Limit Single Carriageways' then 60
        when '30 mph speed limit roads' then 30
        when '20 mph speed limit roads' then 20
    end                                                                  as speed_limit_mph,
    row_order,
    measure                                                              as band_label,
    case when measure like '%under%' then 0
         else (regexp_match(measure, 'vehicles (\d+)'))[1]::int end      as band_lower_mph,
    case when measure like '%under%' then (regexp_match(measure, 'under (\d+)'))[1]::int
         when measure like '%and over%' then null
         else (regexp_match(measure, 'vehicles (\d+)'))[1]::int + 5 end  as band_upper_mph,
    (regexp_match(measure, ' to (\d+)'))[1]::int                         as printed_upper_mph,
    case when cars = '[low]' then null else cars::numeric end            as share_pct,
    cars = '[low]'                                                       as is_low
from source
where measure like 'Percentage of vehicles %'
  and measure not like '%exceeding%'