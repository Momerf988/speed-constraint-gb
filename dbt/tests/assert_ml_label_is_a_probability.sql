-- The model's label is a probability of being killed or seriously injured: it must exist and
-- lie between 0 and 1, and be exactly 1 for a death.

select casualty_key, ksi_adjusted, fatal
from {{ ref('ml_casualty_severity') }}
where ksi_adjusted is null
   or ksi_adjusted < 0
   or ksi_adjusted > 1
   or (fatal = 1 and ksi_adjusted <> 1)
