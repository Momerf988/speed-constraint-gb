{#
    By default dbt names schemas "<profile schema>_<custom schema>", e.g. analytics_marts.
    This override uses the custom schema name on its own: staging, intermediate, marts.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}