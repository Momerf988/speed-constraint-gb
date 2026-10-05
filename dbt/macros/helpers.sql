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