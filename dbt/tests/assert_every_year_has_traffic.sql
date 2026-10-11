-- Every casualty year from 1994 must have traffic figures, or the inner join in
-- agg_rates_year_road_category silently drops that year. This will fail when a new
-- STATS19 year is loaded before DfT publishes its traffic table, which is the point.

select distinct c.year
from {{ ref('agg_casualties_year_road') }} as c
left join {{ ref('stg_traffic_road_class') }} as t
  on t.year = c.year
where c.year >= {{ var('urban_rural_first_year') }}
  and t.year is null