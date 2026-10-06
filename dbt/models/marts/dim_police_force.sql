-- One row per police force, with the country it covers.
-- Country matters later: the Wales 20 mph study compares Welsh forces with English ones.
-- Scotland's eight regional codes (91-98) end in 2019 and Police Scotland (99) starts in
-- 2020, so Scottish figures are only comparable over time at country level.

select
    code::smallint                                  as police_force_code,
    label                                           as police_force,
    case
        when code::int between 60 and 63 then 'Wales'
        when code::int between 91 and 99 then 'Scotland'
        else 'England'
    end                                             as country
from {{ ref('stg_code_list') }}
where table_name = 'collision'
  and field_name = 'police_force'