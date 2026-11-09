-- DfT free-flow van speed distributions (SPE0102) in long form: one row per year, road type
-- and 5 mph speed band. Same rules as stg_speed_bands_cars. DfT publishes no van bands on
-- national-limit single carriageways ("[x]"): van limits there depend on the van's weight, which
-- its counters cannot measure (SPE0102 note 6). Those rows are left out here; the van's average
-- speed on those roads is in stg_speed_summary_vans.

with source as (
    select * from {{ source('raw', 'speed_spe0102') }}
)

select
    year,
    road_type,
    case road_type
        when 'Motorways' then 70
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
    case when light_commercial_vehicles = '[low]' then null
         else light_commercial_vehicles::numeric end                     as share_pct,
    light_commercial_vehicles = '[low]'                                  as is_low
from source
where measure like 'Percentage of vehicles %'
  and measure not like '%exceeding%'
  and light_commercial_vehicles <> '[x]'
