-- The marts must reproduce DfT's own published headline totals.
-- This is the project's noise-floor check: if the pipeline cannot reproduce the
-- official numbers, nothing built on it can be trusted.
-- A singular test passes when it returns no rows.

with ours as (
    select collision_year as year, 'casualties' as measure, count(*)::numeric as our_value
    from {{ ref('fct_casualty') }} group by 1
    union all
    select collision_year, 'killed', sum(fatal)::numeric
    from {{ ref('fct_casualty') }} group by 1
    union all
    select collision_year, 'ksi_adjusted', sum(ksi_adjusted)::numeric
    from {{ ref('fct_casualty') }} group by 1
)

select
    p.year,
    p.measure,
    p.published_value,
    o.our_value,
    p.source
from {{ ref('dft_published_totals') }} as p
left join ours as o
  on o.year = p.year and o.measure = p.measure
where o.our_value is null
   or abs(o.our_value - p.published_value) > p.tolerance