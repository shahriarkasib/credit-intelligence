#!/usr/bin/env python3
"""
Generate comprehensive ERD diagram for Credit Intelligence database.

Shows all tables with columns, highlighting:
- Primary keys (yellow)
- Foreign keys (light blue)
- Hierarchy/join columns (pale blue)
- Regular columns (white)
"""

import os
from pathlib import Path

# Define complete schema with all tables and columns
SCHEMA = {
    # Workflow Tables (wf_*)
    "wf_runs": {
        "category": "workflow",
        "color": "#1565C0",  # Blue
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "UK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("model", "VARCHAR", ""),
            ("temperature", "DECIMAL", ""),
            ("status", "VARCHAR", ""),
            ("started_at", "TIMESTAMPTZ", ""),
            ("completed_at", "TIMESTAMPTZ", ""),
            ("risk_level", "VARCHAR", ""),
            ("credit_score", "INT", ""),
            ("confidence", "DECIMAL", ""),
            ("total_time_ms", "DECIMAL", ""),
            ("total_steps", "INT", ""),
            ("total_llm_calls", "INT", ""),
            ("tools_used", "JSONB", ""),
            ("evaluation_score", "DECIMAL", ""),
            ("workflow_correct", "BOOL", ""),
            ("output_correct", "BOOL", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "wf_llm_calls": {
        "category": "workflow",
        "color": "#1565C0",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("call_type", "VARCHAR", ""),
            ("model", "VARCHAR", ""),
            ("temperature", "DECIMAL", ""),
            ("prompt", "TEXT", ""),
            ("response", "TEXT", ""),
            ("prompt_tokens", "INT", ""),
            ("completion_tokens", "INT", ""),
            ("total_tokens", "INT", ""),
            ("input_cost", "DECIMAL", ""),
            ("output_cost", "DECIMAL", ""),
            ("total_cost", "DECIMAL", ""),
            ("execution_time_ms", "DECIMAL", ""),
            ("status", "VARCHAR", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "wf_tool_calls": {
        "category": "workflow",
        "color": "#1565C0",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("tool_name", "VARCHAR", ""),
            ("tool_input", "JSONB", ""),
            ("tool_output", "JSONB", ""),
            ("execution_time_ms", "DECIMAL", ""),
            ("status", "VARCHAR", ""),
            ("error", "TEXT", ""),
            ("parent_node", "VARCHAR", ""),
            ("workflow_phase", "VARCHAR", ""),
            ("call_depth", "INT", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "wf_assessments": {
        "category": "workflow",
        "color": "#1565C0",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("risk_level", "VARCHAR", ""),
            ("credit_score", "INT", ""),
            ("confidence", "DECIMAL", ""),
            ("reasoning", "TEXT", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "wf_plans": {
        "category": "workflow",
        "color": "#1565C0",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("full_plan", "JSONB", ""),
            ("num_tasks", "INT", ""),
            ("plan_summary", "TEXT", ""),
            ("task_1", "TEXT", ""),
            ("task_2", "TEXT", ""),
            ("task_3", "TEXT", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "wf_data_sources": {
        "category": "workflow",
        "color": "#1565C0",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("source_name", "VARCHAR", ""),
            ("status", "VARCHAR", ""),
            ("records_found", "INT", ""),
            ("data_summary", "TEXT", ""),
            ("execution_time_ms", "DECIMAL", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "wf_state_dumps": {
        "category": "workflow",
        "color": "#1565C0",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("company_info", "JSONB", ""),
            ("plan", "JSONB", ""),
            ("api_data", "JSONB", ""),
            ("search_data", "JSONB", ""),
            ("assessment", "JSONB", ""),
            ("evaluation", "JSONB", ""),
            ("coalition_score", "DECIMAL", ""),
            ("agent_metrics_score", "DECIMAL", ""),
            ("duration_ms", "DECIMAL", ""),
            ("status", "VARCHAR", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },

    # LangGraph Tables (lg_*)
    "lg_events": {
        "category": "langgraph",
        "color": "#2E7D32",  # Green
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("event_type", "VARCHAR", ""),
            ("event_name", "VARCHAR", ""),
            ("model", "VARCHAR", ""),
            ("tokens", "INT", ""),
            ("input_preview", "TEXT", ""),
            ("output_preview", "TEXT", ""),
            ("duration_ms", "DECIMAL", ""),
            ("status", "VARCHAR", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },

    # Evaluation Tables (eval_*)
    "eval_results": {
        "category": "evaluation",
        "color": "#E65100",  # Orange
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("evaluation_type", "VARCHAR", ""),
            ("overall_score", "DECIMAL", ""),
            ("tool_selection_score", "DECIMAL", ""),
            ("data_quality_score", "DECIMAL", ""),
            ("synthesis_score", "DECIMAL", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "eval_tool_selection": {
        "category": "evaluation",
        "color": "#E65100",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("selected_tools", "JSONB", ""),
            ("expected_tools", "JSONB", ""),
            ("precision_score", "DECIMAL", ""),
            ("recall_score", "DECIMAL", ""),
            ("f1_score", "DECIMAL", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "eval_consistency": {
        "category": "evaluation",
        "color": "#E65100",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("model_name", "VARCHAR", ""),
            ("num_runs", "INT", ""),
            ("risk_level_consistency", "DECIMAL", ""),
            ("score_consistency", "DECIMAL", ""),
            ("overall_consistency", "DECIMAL", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "eval_cross_model": {
        "category": "evaluation",
        "color": "#E65100",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("models_compared", "JSONB", ""),
            ("risk_level_agreement", "DECIMAL", ""),
            ("cross_model_agreement", "DECIMAL", ""),
            ("best_model", "VARCHAR", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "eval_llm_judge": {
        "category": "evaluation",
        "color": "#E65100",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("model_used", "VARCHAR", ""),
            ("accuracy_score", "DECIMAL", ""),
            ("completeness_score", "DECIMAL", ""),
            ("overall_score", "DECIMAL", ""),
            ("suggestions", "JSONB", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "eval_agent_metrics": {
        "category": "evaluation",
        "color": "#E65100",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("intent_correctness", "DECIMAL", ""),
            ("plan_quality", "DECIMAL", ""),
            ("tool_choice_correctness", "DECIMAL", ""),
            ("tool_completeness", "DECIMAL", ""),
            ("trajectory_match", "DECIMAL", ""),
            ("final_answer_quality", "DECIMAL", ""),
            ("overall_score", "DECIMAL", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "eval_coalition": {
        "category": "evaluation",
        "color": "#E65100",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("is_correct", "BOOL", ""),
            ("correctness_score", "DECIMAL", ""),
            ("confidence", "DECIMAL", ""),
            ("correctness_category", "VARCHAR", ""),
            ("num_evaluators", "INT", ""),
            ("votes_json", "JSONB", ""),
            ("status", "VARCHAR", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "eval_log_tests": {
        "category": "evaluation",
        "color": "#E65100",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("verification_status", "VARCHAR", ""),
            ("total_tables_logged", "INT", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },

    # Metadata Tables (meta_*)
    "meta_prompts": {
        "category": "metadata",
        "color": "#6A1B9A",  # Purple
        "columns": [
            ("id", "BIGINT", "PK"),
            ("run_id", "VARCHAR(64)", "FK"),
            ("company_name", "VARCHAR", "JOIN"),
            ("node", "VARCHAR", "JOIN"),
            ("node_type", "VARCHAR", "JOIN"),
            ("agent_name", "VARCHAR", "JOIN"),
            ("master_agent", "VARCHAR", "JOIN"),
            ("step_number", "INT", "JOIN"),
            ("prompt_id", "VARCHAR", ""),
            ("prompt_name", "VARCHAR", ""),
            ("category", "VARCHAR", ""),
            ("system_prompt", "TEXT", ""),
            ("user_prompt", "TEXT", ""),
            ("variables", "JSONB", ""),
            ("model", "VARCHAR", ""),
            ("temperature", "DECIMAL", ""),
            ("timestamp", "TIMESTAMPTZ", ""),
        ]
    },
    "meta_api_keys": {
        "category": "metadata",
        "color": "#6A1B9A",
        "columns": [
            ("id", "BIGINT", "PK"),
            ("provider", "VARCHAR", ""),
            ("key_name", "VARCHAR", ""),
            ("encrypted_key", "TEXT", ""),
            ("is_active", "BOOL", ""),
            ("created_at", "TIMESTAMPTZ", ""),
            ("updated_at", "TIMESTAMPTZ", ""),
        ]
    },
}

# Column type colors
COLORS = {
    "PK": "#FFEB3B",  # Yellow - Primary Key
    "FK": "#BBDEFB",  # Light Blue - Foreign Key
    "JOIN": "#E1F5FE",  # Pale Blue - Join column
    "": "#FFFFFF",  # White - Regular column
}

CATEGORY_COLORS = {
    "workflow": "#1565C0",
    "langgraph": "#2E7D32",
    "evaluation": "#E65100",
    "metadata": "#6A1B9A",
}


def generate_dot(compact=False):
    """Generate Graphviz DOT file content with balanced layout."""

    # For compact version, show fewer columns per table
    def get_columns(table_info, compact_mode):
        if not compact_mode:
            return table_info["columns"]
        # Show only key columns in compact mode
        return [col for col in table_info["columns"] if col[2] in ("PK", "FK", "JOIN") or col[0] in ("tool_name", "model", "status", "timestamp")][:12]

    lines = [
        'digraph Credit_Intelligence_ERD {',
        '    // Graph settings - balanced 2x2 grid layout',
        '    rankdir=TB;',
        '    nodesep=0.3;',
        '    ranksep=0.5;',
        '    splines=polyline;',
        '    compound=true;',
        '    newrank=true;',
        f'    node [fontname="Arial" fontsize={"7" if compact else "8"} shape=none];',
        '    edge [fontname="Arial" fontsize=7];',
        '',
        '    // Legend at top',
        '    legend [label=<',
        '        <TABLE BORDER="2" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" BGCOLOR="#FAFAFA">',
        '            <TR><TD COLSPAN="4" BGCOLOR="#37474F"><FONT COLOR="white"><B>LEGEND</B></FONT></TD></TR>',
        '            <TR>',
        '                <TD BGCOLOR="#FFEB3B"><B>PK</B></TD><TD>Primary Key</TD>',
        '                <TD BGCOLOR="#BBDEFB"><B>FK</B></TD><TD>Foreign Key</TD>',
        '            </TR>',
        '            <TR>',
        '                <TD BGCOLOR="#E1F5FE"><B>J</B></TD><TD>Join Column</TD>',
        '                <TD></TD><TD></TD>',
        '            </TR>',
        '            <TR><TD COLSPAN="4" BGCOLOR="#EEEEEE"><FONT POINT-SIZE="6">Table Categories:</FONT></TD></TR>',
        '            <TR>',
        '                <TD BGCOLOR="#1565C0"><FONT COLOR="white">wf_*</FONT></TD><TD>Workflow</TD>',
        '                <TD BGCOLOR="#2E7D32"><FONT COLOR="white">lg_*</FONT></TD><TD>LangGraph</TD>',
        '            </TR>',
        '            <TR>',
        '                <TD BGCOLOR="#E65100"><FONT COLOR="white">eval_*</FONT></TD><TD>Evaluation</TD>',
        '                <TD BGCOLOR="#6A1B9A"><FONT COLOR="white">meta_*</FONT></TD><TD>Metadata</TD>',
        '            </TR>',
        '        </TABLE>',
        '    >];',
        '',
        '    // Force 2x2 grid layout using invisible edges and ranks',
        '    { rank=same; cluster_workflow; cluster_langgraph; }',
        '    { rank=same; cluster_evaluation; cluster_metadata; }',
        '',
    ]

    # Group tables by category for subgraphs
    categories = {
        "workflow": [],
        "langgraph": [],
        "evaluation": [],
        "metadata": []
    }

    for table_name, table_info in SCHEMA.items():
        categories[table_info["category"]].append(table_name)

    # Generate table nodes within subgraphs - 2x2 grid
    subgraph_configs = [
        ("workflow", "Workflow Tables (wf_*)", "#E3F2FD"),
        ("langgraph", "LangGraph Tables (lg_*)", "#E8F5E9"),
        ("evaluation", "Evaluation Tables (eval_*)", "#FFF3E0"),
        ("metadata", "Metadata Tables (meta_*)", "#F3E5F5"),
    ]

    for cat_id, cat_label, bg_color in subgraph_configs:
        lines.append(f'    subgraph cluster_{cat_id} {{')
        lines.append(f'        label="{cat_label}";')
        lines.append(f'        style="filled,rounded";')
        lines.append(f'        fillcolor="{bg_color}";')
        lines.append(f'        fontname="Arial Bold";')
        lines.append(f'        fontsize=9;')
        lines.append(f'        margin=10;')
        lines.append('')

        for table_name in categories[cat_id]:
            table_info = SCHEMA[table_name]
            header_color = CATEGORY_COLORS[cat_id]
            columns = get_columns(table_info, compact)

            # Start table HTML
            lines.append(f'        {table_name} [label=<')
            lines.append(f'            <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">')
            lines.append(f'                <TR><TD COLSPAN="2" BGCOLOR="{header_color}"><FONT COLOR="white"><B>{table_name}</B></FONT></TD></TR>')

            for col_name, col_type, col_key in columns:
                bg_color_col = COLORS.get(col_key, "#FFFFFF")
                # Key indicator
                key_icon = ""
                if col_key == "PK":
                    key_icon = '<FONT COLOR="#B8860B"><B>PK</B> </FONT>'
                elif col_key == "FK":
                    key_icon = '<FONT COLOR="#1976D2"><B>FK</B> </FONT>'
                elif col_key == "JOIN":
                    key_icon = '<FONT COLOR="#0288D1"><B>J</B> </FONT>'

                lines.append(f'                <TR><TD BGCOLOR="{bg_color_col}" ALIGN="LEFT">{key_icon}{col_name}</TD><TD BGCOLOR="{bg_color_col}" ALIGN="LEFT"><FONT COLOR="#666">{col_type}</FONT></TD></TR>')

            # Show ellipsis if columns were truncated
            if compact and len(table_info["columns"]) > len(columns):
                lines.append(f'                <TR><TD COLSPAN="2" ALIGN="CENTER"><FONT COLOR="#999">... +{len(table_info["columns"]) - len(columns)} more</FONT></TD></TR>')

            lines.append('            </TABLE>')
            lines.append('        >];')
            lines.append('')

        lines.append('    }')
        lines.append('')

    # Add relationships (FK to wf_runs) - only main ones to reduce clutter
    lines.append('    // Foreign Key Relationships (solid blue lines)')
    main_tables = ["wf_llm_calls", "wf_tool_calls", "wf_assessments", "lg_events", "eval_results", "eval_coalition", "meta_prompts"]
    for table_name in main_tables:
        if table_name in SCHEMA:
            lines.append(f'    wf_runs -> {table_name} [color="#1976D2" style="solid" arrowhead="crow" penwidth=1.2];')

    # Add logical join relationships
    lines.append('')
    lines.append('    // Logical Join Relationships (dashed gray lines)')
    join_pairs = [
        ("wf_llm_calls", "lg_events"),
        ("wf_tool_calls", "lg_events"),
        ("eval_agent_metrics", "eval_coalition"),
    ]
    for src, dst in join_pairs:
        lines.append(f'    {src} -> {dst} [color="#9E9E9E" style="dashed" constraint=false arrowhead="none" penwidth=1];')

    lines.append('}')

    return '\n'.join(lines)


def main():
    """Generate ERD files."""
    print("Generating ERD diagrams...")

    docs_dir = Path(__file__).parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)

    try:
        import subprocess

        # 1. Full ERD (all columns)
        print("\n1. Full ERD (all columns):")
        dot_content_full = generate_dot(compact=False)
        dot_file_full = docs_dir / "credit_intelligence_erd_full.dot"
        with open(dot_file_full, "w") as f:
            f.write(dot_content_full)

        png_file_full = docs_dir / "credit_intelligence_erd_full.png"
        result = subprocess.run(
            ["dot", "-Tpng", "-Gdpi=120", str(dot_file_full), "-o", str(png_file_full)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"   Generated: {png_file_full.name}")

        # 2. Compact ERD (key columns only - best for docs)
        print("\n2. Compact ERD (key columns only - RECOMMENDED for docs):")
        dot_content_compact = generate_dot(compact=True)
        dot_file_compact = docs_dir / "credit_intelligence_erd.dot"
        with open(dot_file_compact, "w") as f:
            f.write(dot_content_compact)

        # PNG
        png_file = docs_dir / "credit_intelligence_erd.png"
        result = subprocess.run(
            ["dot", "-Tpng", "-Gdpi=150", str(dot_file_compact), "-o", str(png_file)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"   Generated: {png_file.name}")

        # SVG
        svg_file = docs_dir / "credit_intelligence_erd.svg"
        result = subprocess.run(
            ["dot", "-Tsvg", str(dot_file_compact), "-o", str(svg_file)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"   Generated: {svg_file.name}")

        # PDF
        pdf_file = docs_dir / "credit_intelligence_erd.pdf"
        result = subprocess.run(
            ["dot", "-Tpdf", str(dot_file_compact), "-o", str(pdf_file)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"   Generated: {pdf_file.name}")

    except FileNotFoundError:
        print("  WARNING: graphviz 'dot' command not found. Install with: brew install graphviz")

    print("\n" + "="*60)
    print("ERD GENERATION COMPLETE")
    print("="*60)
    print(f"\nFiles saved to: {docs_dir}")
    print("\nRecommended for Google Docs:")
    print("  - credit_intelligence_erd.png (compact, balanced layout)")
    print("  - credit_intelligence_erd.pdf (vector, scalable)")
    print("\nDetailed version:")
    print("  - credit_intelligence_erd_full.png (all columns)")


if __name__ == "__main__":
    main()
