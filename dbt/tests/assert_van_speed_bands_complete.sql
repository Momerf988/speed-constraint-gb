-- Van bands should cover almost all vans, as the car bands do. Fail if the shortfall exceeds 1.5
-- points, if the shares exceed 100, or if there are no "[low]" cells to explain a shortfall.

select
    year,
    road_type,
    100 - sum(share_pct)  as missing_pct,
    bool_or(is_low)       as has_low_cells
from {{ ref('stg_speed_bands_vans') }}
group by 1, 2
having 100 - sum(share_pct) < -0.05
    or 100 - sum(share_pct) > 1.5
    or (not bool_or(is_low) and abs(100 - sum(share_pct)) > 0.05)
