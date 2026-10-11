-- The road categories must reproduce DfT's published counts by road type and class
-- (RAS0302, 2025). This proves the urban/rural x A-road/minor/motorway mapping in
-- int_collision_enriched is DfT's own.

with ours as (
    select year, traffic_road_category, 'killed' as measure, sum(killed)::numeric as our_value
    from {{ ref('agg_casualties_year_road') }} group by 1, 2
    union all
    select year, traffic_road_category, 'ksi_adjusted', sum(ksi_adjusted)::numeric
    from {{ ref('agg_casualties_year_road') }} group by 1, 2
    union all
    select year, traffic_road_category, 'casualties', sum(casualties)::numeric
    from {{ ref('agg_casualties_year_road') }} group by 1, 2
)

select
    p.year,
    p.traffic_road_category,
    p.measure,
    p.published_value,
    o.our_value
from {{ ref('dft_published_road_type_counts') }} as p
left join ours as o
  on o.year = p.year
 and o.traffic_road_category = p.traffic_road_category
 and o.measure = p.measure
where o.our_value is null
   or abs(o.our_value - p.published_value) > p.tolerance