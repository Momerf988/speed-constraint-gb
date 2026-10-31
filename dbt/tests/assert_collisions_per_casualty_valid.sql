-- A collision of a given severity has at least one casualty of that severity, so there can
-- never be more collisions than casualties of the same severity: the ratio lies in (0, 1].
-- A value outside that range means the two counts were joined or filtered differently.

select *
from {{ ref('agg_collisions_per_casualty') }}
where collisions_per_casualty <= 0
   or collisions_per_casualty > 1
