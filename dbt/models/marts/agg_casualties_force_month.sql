-- Casualties per police force, month and road group from 2015: the panel for the Wales
-- difference-in-differences study. "32 or 48 km/h roads" holds the roads Wales's September
-- 2023 default limit applies to, before and after: many moved from 48 to 32 km/h, so 32 km/h
-- roads alone would change membership because of the policy itself.
-- Every force, month and road group appears, with zeros where there were no casualties: to a
-- count model, a month with no casualties is information, not a gap.

with casualties as (
    select
        date_trunc('month', d.calendar_date)::date              as month,
        f.police_force_code,
        case when r.speed_limit_kmh in (32, 48) then '32 or 48 km/h roads'
             else 'other roads' end                             as road_group,
        count(*)                                                as casualties,
        sum(f.fatal)                                            as killed,
        sum(f.ksi_adjusted)                                     as ksi_adjusted
    from {{ ref('fct_casualty') }} as f
    join {{ ref('dim_date') }} as d
      on d.date_key = f.date_key
    join {{ ref('dim_road') }} as r
      on r.road_key = f.road_key
    where f.collision_year >= 2015
    group by 1, 2, 3
),

grid as (
    select m.month, p.police_force_code, p.country, g.road_group
    from (select distinct month from casualties) as m
    cross join {{ ref('dim_police_force') }} as p
    cross join (values ('32 or 48 km/h roads'), ('other roads')) as g (road_group)
    where p.police_force_code in (select police_force_code from casualties)   -- forces active since 2015
)

select
    grid.month,
    grid.police_force_code,
    grid.country,
    grid.road_group,
    coalesce(c.casualties, 0)                                   as casualties,
    coalesce(c.killed, 0)                                       as killed,
    coalesce(c.ksi_adjusted, 0)                                 as ksi_adjusted
from grid
left join casualties as c
  on c.month = grid.month
 and c.police_force_code = grid.police_force_code
 and c.road_group = grid.road_group
