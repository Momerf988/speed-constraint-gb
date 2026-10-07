-- Checks the severity adjustment behaves as DfT describes it:
--   * a fatal casualty has a probability of 0 of being serious (it is counted as fatal);
--   * for every other casualty, P(serious) + P(slight) = 1, to within rounding.
-- Probabilities are published to 5 decimal places; the largest observed rounding
-- error is about 0.0000014, so the tolerance is one unit in the 5th place.

select
    casualty_key,
    casualty_severity,
    serious_adjusted,
    slight_adjusted
from {{ ref('fct_casualty') }}
where collision_year >= {{ var('adjusted_severity_first_year') }}
  and (
        (casualty_severity = 1 and serious_adjusted <> 0)
     or (casualty_severity > 1 and abs(serious_adjusted + slight_adjusted - 1) > 0.00001)
  )