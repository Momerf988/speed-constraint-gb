-- One row per road environment: a combination of speed limit, road type, road class
-- and urban/rural. This is the dimension every speed-related question goes through.

with roads as (
    select distinct
        road_key,
        speed_limit_mph,
        speed_limit_kmh,
        is_standard_speed_limit,
        road_type_code,
        first_road_class_code,
        urban_rural_code,
        traffic_road_category
    from {{ ref('int_collision_enriched') }}
)

select
    road_key,
    speed_limit_mph,
    speed_limit_kmh,
    coalesce(is_standard_speed_limit, false)                                    as is_standard_speed_limit,
    road_type_code,
    {{ code_label('collision', 'road_type', 'road_type_code') }}                as road_type,
    first_road_class_code,
    {{ code_label('collision', 'first_road_class', 'first_road_class_code') }}  as first_road_class,
    urban_rural_code,
    {{ code_label('collision', 'urban_or_rural_area', 'urban_rural_code') }}    as urban_rural,
    traffic_road_category
from roads