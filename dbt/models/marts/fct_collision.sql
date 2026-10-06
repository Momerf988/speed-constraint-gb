-- Fact table at collision grain: one row per collision.
-- Keys point to the date, road and police force dimensions; the rest are measures
-- (things you add up) and collision-level details.
-- Adjusted probabilities are null before 2004 for every row: raw data carries a stray
-- 0 on pre-2004 fatal collisions, which would look like a real value.

{% set first_year = var('adjusted_severity_first_year') %}

select
    collision_index,
    collision_year,
    date_key,
    hour_of_day,
    road_key,
    police_force_code,
    collision_severity,
    (collision_severity = 1)::int                                       as fatal_collision,
    case when collision_year >= {{ first_year }}
         then adjusted_serious_probability end                          as serious_adjusted,
    case when collision_year >= {{ first_year }}
         then adjusted_slight_probability end                           as slight_adjusted,
    number_of_vehicles,
    number_of_casualties,
    junction_detail_code,
    light_conditions_code,
    weather_conditions_code,
    road_surface_code,
    latitude,
    longitude,
    local_authority_ons_district,
    lsoa_code
from {{ ref('int_collision_enriched') }}