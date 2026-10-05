-- One row per vehicle, as published, with clear names and -1 turned into null.
-- Not used by the Phase 3 marts; it is here so later phases (the severity model)
-- start from a cleaned table rather than raw.

with source as (
    select * from {{ source('raw', 'vehicle') }}
)

select
    collision_index,
    vehicle_reference,
    collision_year,
    {{ unknown_to_null('vehicle_type') }}               as vehicle_type_code,
    {{ unknown_to_null('vehicle_manoeuvre') }}          as vehicle_manoeuvre_code,
    {{ unknown_to_null('sex_of_driver') }}              as driver_sex_code,
    {{ unknown_to_null('age_of_driver') }}              as driver_age,
    {{ unknown_to_null('age_band_of_driver') }}         as driver_age_band_code,
    {{ unknown_to_null('engine_capacity_cc') }}         as engine_capacity_cc,
    {{ unknown_to_null('propulsion_code') }}            as propulsion_code,
    {{ unknown_to_null('age_of_vehicle') }}             as vehicle_age,
    {{ unknown_to_null('first_point_of_impact') }}      as first_point_of_impact_code,
    generic_make_model
from source