-- The bands at or above the limit must reproduce DfT's published "% exceeding". They fall
-- short by 0.3-0.6 points in most years because the fastest bands include "[low]" cells, so
-- the gap must be zero or more and no bigger than the total "[low]" share. A larger gap would
-- mean the bands or the speed-limit mapping are wrong.

with bands as (
    select
        year,
        road_type,
        coalesce(sum(share_pct) filter (where band_lower_mph >= speed_limit_mph), 0) as at_or_above_limit_pct,
        100 - sum(share_pct)                                                          as missing_pct
    from {{ ref('stg_speed_bands_cars') }}
    group by 1, 2
)

select
    b.year,
    b.road_type,
    s.pct_exceeding - b.at_or_above_limit_pct  as gap_pct,
    b.missing_pct
from bands as b
join {{ ref('stg_speed_summary_cars') }} as s
  on s.year = b.year
 and s.road_type = b.road_type
where s.pct_exceeding - b.at_or_above_limit_pct < -0.05
   or s.pct_exceeding - b.at_or_above_limit_pct > b.missing_pct + 0.05