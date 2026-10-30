-- Every casualty from 2010 onwards must land in exactly one segment: per year, the segment
-- totals must equal fct_casualty's, for casualties and for deaths. Returns the years that differ.

with segments as (
    select year, sum(casualties) as casualties, sum(killed) as killed
    from {{ ref('agg_casualties_segment_year') }}
    group by year
),

facts as (
    select collision_year as year, count(*) as casualties, sum(fatal) as killed
    from {{ ref('fct_casualty') }}
    where collision_year >= 2010
    group by collision_year
)

select
    year,
    f.casualties                                                as fact_casualties,
    s.casualties                                                as segment_casualties,
    f.killed                                                    as fact_killed,
    s.killed                                                    as segment_killed
from facts as f
full join segments as s using (year)
where f.casualties is distinct from s.casualties
   or f.killed is distinct from s.killed