-- Fact table at casualty grain: one row per injured person.
-- This is the main table for the project: casualties are what the policy is meant to
-- prevent. Each casualty inherits its collision's date, road and police force keys.

with casualties as (
    select * from {{ ref('int_casualty_severity') }}
),

collisions as (
    select collision_index, date_key, road_key, police_force_code
    from {{ ref('int_collision_enriched') }}
)

select
    s.collision_index || '-' || s.vehicle_reference || '-' || s.casualty_reference  as casualty_key,
    s.collision_index,
    s.vehicle_reference,
    s.casualty_reference,
    s.collision_year,
    c.date_key,
    c.road_key,
    c.police_force_code,
    {{ surrogate_key(['s.casualty_class_code', 's.casualty_type_code', 's.sex_code', 's.age_band_code']) }} as casualty_profile_key,
    s.age,
    s.casualty_severity,
    s.fatal,
    s.serious_reported,
    s.slight_reported,
    s.ksi_reported,
    s.serious_adjusted,
    s.slight_adjusted,
    s.ksi_adjusted
from casualties as s
join collisions as c
  on c.collision_index = s.collision_index