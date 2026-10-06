{#- STATS19 uses -1 for "data missing or out of range". Turn it into a real null. -#}
{% macro unknown_to_null(column) -%}
    nullif({{ column }}, -1)
{%- endmacro %}


{#-
    A surrogate key: one stable hash built from several columns.
    Nulls become '~' first, so (null, 1) and (1, null) get different keys.
-#}
{% macro surrogate_key(columns) -%}
    md5(concat_ws('|'
    {%- for column in columns %}, coalesce({{ column }}::text, '~'){% endfor -%}
    ))
{%- endmacro %}


{#- The label for a STATS19 code, looked up in the data guide's code list. -#}
{% macro code_label(table_name, field_name, code_column) -%}
    (
        select label from {{ ref('stg_code_list') }} as l
        where l.table_name = '{{ table_name }}'
          and l.field_name = '{{ field_name }}'
          and l.code = {{ code_column }}::text
    )
{%- endmacro %}