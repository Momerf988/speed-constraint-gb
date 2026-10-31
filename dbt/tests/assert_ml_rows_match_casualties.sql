-- The severity model's input must have one row per casualty from 2021 to 2025: the joins to
-- vehicles and collisions must neither drop nor duplicate anyone.

select
    (select count(*) from {{ ref('fct_casualty') }} where collision_year between 2021 and 2025) as fact_rows,
    (select count(*) from {{ ref('ml_casualty_severity') }})                                    as model_rows
where (select count(*) from {{ ref('fct_casualty') }} where collision_year between 2021 and 2025)
   <> (select count(*) from {{ ref('ml_casualty_severity') }})
