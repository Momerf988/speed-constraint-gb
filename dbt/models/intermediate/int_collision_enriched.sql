-- Collisions with the derived attributes the marts need: keys into the date and road
-- dimensions, the speed limit in km/h, and the road category DfT uses for traffic volumes.

with collisions as (
    select * from {{ ref('stg_collision') }}
),

enriched as (
    select
        *,
        speed_limit_mph in (20, 30, 40, 50, 60, 70)                 as is_standard_speed_limit,
        case
            when speed_limit_mph in (20, 30, 40, 50, 60, 70)
            then round(speed_limit_mph * 1.609344)::smallint
        end                                                          as speed_limit_kmh,
        -- Matches the road categories in DfT's traffic volume tables, so casualties can
        -- be divided by vehicle-km in Phase 4. Motorways include A(M) roads.
        -- To be checked against the traffic table definitions in Phase 4.
        case
            when first_road_class_code in (1, 2)                        then 'Motorway'
            when urban_rural_code = 1 and first_road_class_code = 3     then 'Urban A road'
            when urban_rural_code = 1 and first_road_class_code > 3     then 'Urban minor road'
            when urban_rural_code = 2 and first_road_class_code = 3     then 'Rural A road'
            when urban_rural_code = 2 and first_road_class_code > 3     then 'Rural minor road'
            else 'Unknown'
        end                                                          as traffic_road_category,
        to_char(collision_date, 'YYYYMMDD')::int                     as date_key,
        extract(hour from collision_time)::smallint                  as hour_of_day
    from collisions
)

select
    *,
    {{ surrogate_key(['speed_limit_mph', 'road_type_code', 'first_road_class_code', 'urban_rural_code']) }} as road_key
from enriched