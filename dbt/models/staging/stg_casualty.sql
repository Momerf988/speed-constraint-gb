-- One row per casualty, as published, with clear names and -1 turned into null.

with source as (
    select * from {{ source('raw', 'casualty') }}
)

select
    collision_index,
    vehicle_reference,
    casualty_reference,
    collision_year,
    {{ unknown_to_null('casualty_class') }}             as casualty_class_code,
    {{ unknown_to_null('casualty_type') }}              as casualty_type_code,
    {{ unknown_to_null('sex_of_casualty') }}            as sex_code,
    {{ unknown_to_null('age_of_casualty') }}            as age,
    {{ unknown_to_null('age_band_of_casualty') }}       as age_band_code,
    casualty_severity,
    casualty_adjusted_severity_serious                  as adjusted_serious_probability,
    casualty_adjusted_severity_slight                   as adjusted_slight_probability
from source