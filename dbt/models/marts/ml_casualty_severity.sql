-- One row per casualty from 2021 to 2025, with what was recorded at the scene: the input for
-- the severity model. Fields recorded only because of the outcome, or that describe police
-- recording rather than risk, are left out (model/severity.py lists them and why).
-- The label is DfT's adjusted probability of being killed or seriously injured (1 for a death),
-- so the switch of police forces to injury-based recording does not move it.

with vehicles_involved as (
    select
        collision_index,
        bool_or(vehicle_type_code in (8, 9))                    as car_involved,
        bool_or(vehicle_type_code = 19)                         as van_involved,
        bool_or(vehicle_type_code in (20, 21, 98))              as hgv_involved,
        bool_or(vehicle_type_code in (10, 11))                  as bus_involved,
        bool_or(vehicle_type_code in (2, 3, 4, 5, 23, 97))      as motorcycle_involved,
        bool_or(vehicle_type_code = 1)                          as pedal_cycle_involved
    from {{ ref('stg_vehicle') }}
    where collision_year between 2021 and 2025
    group by 1
)

select
    f.casualty_key,
    f.collision_index,
    f.collision_year,
    f.ksi_adjusted,
    f.fatal,
    -- the person
    p.casualty_class,
    p.casualty_type,
    p.sex,
    f.age,
    -- their vehicle; for a pedestrian, the vehicle that hit them
    v.vehicle_type_code                                         as vehicle_type,
    v.vehicle_manoeuvre_code                                    as vehicle_manoeuvre,
    v.first_point_of_impact_code                                as first_point_of_impact,
    v.vehicle_age,
    v.engine_capacity_cc,
    v.driver_age,
    -- the other vehicles in the collision
    c.number_of_vehicles,
    i.car_involved,
    i.van_involved,
    i.hgv_involved,
    i.bus_involved,
    i.motorcycle_involved,
    i.pedal_cycle_involved,
    -- the road and conditions
    r.speed_limit_kmh,
    r.road_type,
    r.first_road_class,
    r.urban_rural,
    c.junction_detail_code                                      as junction_detail,
    c.light_conditions_code                                     as light_conditions,
    c.weather_conditions_code                                   as weather_conditions,
    c.road_surface_code                                         as road_surface,
    c.hour_of_day,
    d.iso_day_of_week                                           as day_of_week
from {{ ref('fct_casualty') }} as f
join {{ ref('fct_collision') }} as c
  on c.collision_index = f.collision_index
join {{ ref('dim_road') }} as r
  on r.road_key = f.road_key
join {{ ref('dim_casualty_profile') }} as p
  on p.casualty_profile_key = f.casualty_profile_key
left join {{ ref('dim_date') }} as d
  on d.date_key = f.date_key
left join {{ ref('stg_vehicle') }} as v
  on v.collision_index = f.collision_index
 and v.vehicle_reference = f.vehicle_reference
left join vehicles_involved as i
  on i.collision_index = f.collision_index
where f.collision_year between 2021 and 2025
