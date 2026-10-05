-- Code-to-label lookups from the DfT data guide.

select
    table_name,
    field_name,
    code,
    label
from {{ source('raw', 'code_list') }}