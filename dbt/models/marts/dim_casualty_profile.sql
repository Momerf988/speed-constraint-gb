-- One row per combination of casualty class, casualty type, sex and age band.
-- A "junk dimension": it gathers several small descriptive codes into one table, so a
-- casualty query needs one join to get all four labels.

with profiles as (
    select distinct
        casualty_class_code,
        casualty_type_code,
        sex_code,
        age_band_code
    from {{ ref('stg_casualty') }}
)

select
    {{ surrogate_key(['casualty_class_code', 'casualty_type_code', 'sex_code', 'age_band_code']) }} as casualty_profile_key,
    casualty_class_code,
    {{ code_label('casualty', 'casualty_class', 'casualty_class_code') }}      as casualty_class,
    casualty_type_code,
    {{ code_label('casualty', 'casualty_type', 'casualty_type_code') }}        as casualty_type,
    sex_code,
    {{ code_label('casualty', 'sex_of_casualty', 'sex_code') }}                as sex,
    age_band_code,
    {{ code_label('casualty', 'age_band_of_casualty', 'age_band_code') }}      as age_band
from profiles