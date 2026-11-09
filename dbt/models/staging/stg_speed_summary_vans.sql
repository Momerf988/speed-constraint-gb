-- DfT's own summary figures for vans (SPE0102), one row per year and road type. On
-- national-limit single carriageways only the average speed and the number observed are
-- published; "[x]" and "[z]" become nulls.

with source as (
    select
        year,
        road_type,
        measure,
        case when light_commercial_vehicles ~ '^[0-9.]+$' then light_commercial_vehicles::numeric end as value
    from {{ source('raw', 'speed_spe0102') }}
)

select
    year,
    road_type,
    max(case when measure = 'Percentage of vehicles exceeding speed limit' then value end)            as pct_exceeding,
    max(case when measure = 'Average free flow speed (miles per hour)' then value end)                as avg_free_flow_mph,
    max(case when measure = 'Number observed (thousands)' then value end)                             as observed_thousands
from source
group by 1, 2
