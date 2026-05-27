from jinja2 import Template


LEARNER = ["DER", "ER", "EWC", "FT", "LWF", "PN", "RAR", "SI"]

DRIFT_DETECTORS = [
    "ADWIN",
    "CUSUM",
    "DDM",
    "PH",
    "ORACLE",
]
BOUNDARIES = ["A", "G", "S"]

column_count = len(DRIFT_DETECTORS) * len(BOUNDARIES)
table = []
for method in LEARNER:
    row = [method] + ["?"] * column_count
    table.append(row)

latex_template = Template(
    r"""
\begin{tabular}{ {{ col_spec }} }
{{ top_rule }}
Learner
{% for dd in drift_detectors -%}
 & \multicolumn{ {{ boundaries|length }} }{c}{ {{ dd }} }
{%- endfor %} \\
{% for cmidrule in cmidrules -%}
{{ cmidrule }}
{% endfor -%}

{% for dd in drift_detectors -%}
{% for boundary in boundaries -%}
 & {{ boundary }}
{%- endfor %}
{%- endfor %} \\
\midrule
{% for row in rows -%}
{{ row | join(' & ') }} \\
{% endfor -%}
\bottomrule
\end{tabular}
""".strip()
)

col_spec = "l " + " ".join(["c" * len(BOUNDARIES)] * len(DRIFT_DETECTORS))
cmidrules = []
top_rule = "\\toprule"
start_col = 2
for _ in DRIFT_DETECTORS:
    end_col = start_col + len(BOUNDARIES) - 1
    cmidrules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
    start_col = end_col + 1

print(
    latex_template.render(
        drift_detectors=DRIFT_DETECTORS,
        boundaries=BOUNDARIES,
        top_rule=top_rule,
        cmidrules=cmidrules,
        rows=table,
        col_spec=col_spec,
    )
)
