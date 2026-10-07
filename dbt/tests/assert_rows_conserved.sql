-- No rows may be lost or duplicated between the raw tables and the fact tables.
-- A join in the wrong place silently drops or multiplies rows; this catches it.

select *
from (
    select
        'collision'                                             as table_name,
        (select count(*) from {{ source('raw', 'collision') }}) as raw_rows,
        (select count(*) from {{ ref('fct_collision') }})       as model_rows
    union all
    select
        'casualty',
        (select count(*) from {{ source('raw', 'casualty') }}),
        (select count(*) from {{ ref('fct_casualty') }})
) as counts
where raw_rows <> model_rows