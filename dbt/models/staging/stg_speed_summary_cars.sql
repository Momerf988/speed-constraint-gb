-- DfT's own summary figures for cars (SPE0102), one row per year and road type:
-- the share exceeding the limit, the average free-flow speed and the number observed.
-- Rows become columns with max(case ...), a pivot.

with source as (
    select * from {{ source('raw', 'speed_spe0102') }}
)

select
    year,
    road_type,
    max(case when measure = 'Applicable speed limit (miles per hour)' then cars::numeric end)                as published_speed_limit_mph,
    max(case when measure = 'Percentage of vehicles exceeding speed limit' then cars::numeric end)            as pct_exceeding,
    max(case when measure like 'Percentage of vehicles exceeding speed limit by 5 %' then cars::numeric end)  as pct_exceeding_by_5_mph,
    max(case when measure like 'Percentage of vehicles exceeding speed limit by 10 %' then cars::numeric end) as pct_exceeding_by_10_mph,
    max(case when measure = 'Average free flow speed (miles per hour)' then cars::numeric end)                as avg_free_flow_mph,
    max(case when measure = 'Number observed (thousands)' then cars::numeric end)                             as observed_thousands
from source
group by 1, 2