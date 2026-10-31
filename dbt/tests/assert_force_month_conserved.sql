-- The Wales panel must hold every dated casualty from 2015 exactly once: per year, its total
-- must equal fct_casualty's. Returns the years that differ.

with panel as (
    select extract(year from month)::int as year, sum(casualties) as casualties
    from {{ ref('agg_casualties_force_month') }}
    group by 1
),

facts as (
    select collision_year as year, count(*) as casualties
    from {{ ref('fct_casualty') }}
    where collision_year >= 2015
      and date_key is not null
    group by 1
)

select facts.year, facts.casualties as fact_casualties, panel.casualties as panel_casualties
from facts
full join panel on panel.year = facts.year
where facts.casualties is distinct from panel.casualties
