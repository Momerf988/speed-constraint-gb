-- Rates exclude casualties whose road category is unknown, because no traffic figure
-- exists for them. That is only safe while they are a small share. The worst year since
-- 1994 is 1999 at 0.67%; fail if any year exceeds 1%.

select
    year,
    sum(casualties) filter (where traffic_road_category = 'Unknown')    as unknown_casualties,
    sum(casualties)                                                     as all_casualties
from {{ ref('agg_casualties_year_road') }}
where year >= {{ var('urban_rural_first_year') }}
group by 1
having coalesce(sum(casualties) filter (where traffic_road_category = 'Unknown'), 0) > 0.01 * sum(casualties)