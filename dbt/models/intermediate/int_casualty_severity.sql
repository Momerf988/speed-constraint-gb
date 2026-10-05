-- Severity measures for each casualty, as numbers that can be summed.
--
-- Reported severity is the police label: fatal, serious or slight.
-- Adjusted severity corrects for police forces moving to injury-based recording
-- (CRASH and COPA), which labels more injuries as serious. Each non-fatal casualty
-- carries a probability of being serious; adjusted counts are sums of those
-- probabilities. Fatal casualties are unaffected. Available from 2004 only.

{% set first_year = var('adjusted_severity_first_year') %}

with casualties as (
    select * from {{ ref('stg_casualty') }}
)

select
    *,
    (casualty_severity = 1)::int                                         as fatal,
    (casualty_severity = 2)::int                                         as serious_reported,
    (casualty_severity = 3)::int                                         as slight_reported,
    (casualty_severity in (1, 2))::int                                   as ksi_reported,
    case when collision_year >= {{ first_year }}
         then adjusted_serious_probability end                            as serious_adjusted,
    case when collision_year >= {{ first_year }}
         then adjusted_slight_probability end                             as slight_adjusted,
    case when collision_year >= {{ first_year }}
         then (casualty_severity = 1)::int + adjusted_serious_probability end as ksi_adjusted
from casualties