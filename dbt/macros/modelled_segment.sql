{#-
    Which modelled road type a road belongs to: the four with DfT free-flow speed data
    (SPE0102), or 'not modelled'. One definition, used by every model that splits by road
    type, so casualties and traffic are always grouped the same way.
      32 km/h and 48 km/h roads       all roads with that limit
      97 km/h single carriageways     DfT's "national speed limit single carriageway"
      113 km/h motorways              motorways only; 113 km/h dual carriageways have no speed data
-#}
{% macro modelled_segment(road) -%}
    case
        when {{ road }}.speed_limit_kmh = 32 then '32 km/h roads'
        when {{ road }}.speed_limit_kmh = 48 then '48 km/h roads'
        when {{ road }}.speed_limit_kmh = 97 and {{ road }}.road_type = 'Single carriageway' then '97 km/h single carriageways'
        when {{ road }}.speed_limit_kmh = 113 and {{ road }}.traffic_road_category = 'Motorway' then '113 km/h motorways'
        else 'not modelled'
    end
{%- endmacro %}
