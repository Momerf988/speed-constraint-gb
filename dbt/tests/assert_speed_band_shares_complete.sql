-- The bands should cover almost all vehicles. "[low]" cells are not printed, so the known
-- shares fall short of 100% by up to about 1% (observed 0.29-1.04 in 2018-2024). Fail if the
-- shortfall exceeds 1.5 points, if the shares exceed 100, or if there are no "[low]" cells
-- to explain a shortfall.

select
    year,
    road_type,
    100 - sum(share_pct)  as missing_pct,
    bool_or(is_low)       as has_low_cells
from {{ ref('stg_speed_bands_cars') }}
group by 1, 2
having 100 - sum(share_pct) < -0.05
    or 100 - sum(share_pct) > 1.5
    or (not bool_or(is_low) and abs(100 - sum(share_pct)) > 0.05)