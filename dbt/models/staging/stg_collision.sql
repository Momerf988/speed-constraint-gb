-- One row per collision, as published, with clear names and -1 turned into null.
-- The pre-2024 "_historic" columns are dropped: DfT has back-filled the 2024-format
-- columns for earlier years (under 0.2% of pre-2024 rows are missing them).

with source as (
    select * from {{ source('raw', 'collision') }}
)

select
    collision_index,
    collision_year,
    date                                                as collision_date,
    time                                                as collision_time,
    {{ unknown_to_null('police_force') }}               as police_force_code,
    collision_severity,
    number_of_vehicles,
    number_of_casualties,
    {{ unknown_to_null('speed_limit') }}                as speed_limit_mph,
    {{ unknown_to_null('road_type') }}                  as road_type_code,
    {{ unknown_to_null('first_road_class') }}           as first_road_class_code,
    {{ unknown_to_null('urban_or_rural_area') }}        as urban_rural_code,
    {{ unknown_to_null('junction_detail') }}            as junction_detail_code,
    {{ unknown_to_null('light_conditions') }}           as light_conditions_code,
    {{ unknown_to_null('weather_conditions') }}         as weather_conditions_code,
    {{ unknown_to_null('road_surface_conditions') }}    as road_surface_code,
    latitude,
    longitude,
    location_easting_osgr,
    location_northing_osgr,
    local_authority_ons_district,
    lsoa_of_accident_location                           as lsoa_code,
    collision_adjusted_severity_serious                 as adjusted_serious_probability,
    collision_adjusted_severity_slight                  as adjusted_slight_probability
from source