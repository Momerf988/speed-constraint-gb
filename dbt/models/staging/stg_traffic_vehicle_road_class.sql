-- DfT traffic by vehicle type (TRA0204), one row per year, road category and vehicle type.
-- A roads are published split by management (trunk, principal); they are added together here.
-- DfT's own total rows ("All Motor Vehicles", "All Roads") are dropped, so summing this view
-- never counts the same traffic twice. Category names match stg_traffic_road_class and dim_road.
-- DfT writes A roads as 'A' Road, quotes included; in SQL that literal is '''A'' Road'.

with source as (
    select * from {{ source('raw', 'traffic_tra0204') }}
    where vehicle <> 'All Motor Vehicles'
      and road_type <> 'All Roads'
)

select
    year,
    case
        when road_type = 'Motorway' then 'Motorway'
        when road_type = '''A'' Road' and rural_urban = 'Rural' then 'Rural A road'
        when road_type = '''A'' Road' and rural_urban = 'Urban' then 'Urban A road'
        when road_type = 'Minor Road' and rural_urban = 'Rural' then 'Rural minor road'
        when road_type = 'Minor Road' and rural_urban = 'Urban' then 'Urban minor road'
    end                                                         as traffic_road_category,
    vehicle                                                     as vehicle_type,
    sum(billion_vehicle_km)::double precision                   as billion_vehicle_km
from source
group by 1, 2, 3
