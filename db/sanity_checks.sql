-- Sanity checks on the raw tables, run after db/load.py.

-- 1. Column counts match the CSV headers.
-- Expected: collision 44, vehicle 32, casualty 23.
select table_name, count(*) from information_schema.columns
where table_schema = 'raw' and table_name in ('collision', 'vehicle', 'casualty')
group by table_name;

-- 2. Vehicles whose collision does not exist. Expected: 0.
select count(*) from raw.vehicle v
where not exists (select 1 from raw.collision c where c.collision_index = v.collision_index);

-- 3. Casualties whose vehicle does not exist. Expected: 0.
select count(*) from raw.casualty cas
where not exists (
    select 1 from raw.vehicle v
    where v.collision_index = cas.collision_index
      and v.vehicle_reference = cas.vehicle_reference
);

-- 4. Duplicate casualty keys. Expected: no rows.
-- (Once db/constraints.sql has run, the primary keys make duplicates impossible.)
select collision_index, vehicle_reference, casualty_reference, count(*)
from raw.casualty
group by collision_index, vehicle_reference, casualty_reference
having count(*) > 1;

-- 5. Every collision severity code has a label. Expected: 1 Fatal, 2 Serious, 3 Slight.
select c.collision_severity, l.label, count(*) as collisions
from raw.collision c
left join raw.code_list l
  on l.table_name = 'collision' and l.field_name = 'collision_severity'
 and l.code = c.collision_severity::text
group by 1, 2 order by 1;
