import json
import os
from pathlib import Path


def config_inference():
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("Snowflake/Arctic-Text2SQL-R1-7B")
    print(config.max_position_embeddings)


def generate_omnistyle_schema(db_folder: str, output_dir: str = "omnistyle_schema_sqlite") -> str:
    """
    Generate a schema file in omni-style format from a database folder containing JSON table definitions.
    
    Args:
        db_folder: Path to the database folder (e.g., .../sqlite/AdventureWorks)
        output_dir: Directory to save the output schema file
    
    Returns:
        Path to the generated schema file
    """
    db_folder = Path(db_folder)
    db_name = db_folder.name
    
    # Collect all JSON files (table definitions)
    json_files = sorted(db_folder.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {db_folder}")
    
    tables = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            table_data = json.load(f)
        
        table_name = table_data.get("table_name", json_file.stem)
        column_names = table_data.get("column_names", [])
        column_types = table_data.get("column_types", [])
        descriptions = table_data.get("description", [""] * len(column_names))
        sample_rows_raw = table_data.get("sample_rows", [])
        
        # Convert sample rows from list of dicts to list of lists
        sample_rows = []
        for row in sample_rows_raw[:2]:  # Take first 2 sample rows
            if isinstance(row, dict):
                sample_rows.append([row.get(col) for col in column_names])
            else:
                sample_rows.append(row)
        
        # Build table schema
        table_schema = {
            "table_name": table_name,
            "table_description": table_data.get("table_description", f"Table containing {table_name} data."),
            "column_names": column_names,
            "column_types": column_types,
            "column_descriptions": descriptions if any(d for d in descriptions) else [f"Column {col}" for col in column_names],
            "primary_key": _infer_primary_key(column_names),
            "sample_rows": sample_rows
        }
        tables.append(table_schema)
    
    # Build the full schema
    schema = {
        "table_num": len(tables),
        "tables": tables,
        "foreign_keys": _infer_foreign_keys(tables)
    }
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write schema file
    output_file = output_path / f"{db_name}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    
    print(f"Generated schema: {output_file}")
    return str(output_file)


def _infer_primary_key(column_names: list[str]) -> list[str]:
    """Infer primary key from column names."""
    # Common primary key patterns
    for col in column_names:
        col_lower = col.lower()
        if col_lower == "id" or col_lower.endswith("_id"):
            return [col]
    # If first column looks like an ID
    if column_names and ("id" in column_names[0].lower()):
        return [column_names[0]]
    return []


def _infer_foreign_keys(tables: list[dict]) -> list[dict]:
    """Infer foreign keys based on column naming conventions."""
    foreign_keys = []
    
    # Build a map of table names to their primary keys
    table_pk_map = {}
    for table in tables:
        table_name = table["table_name"]
        pk = table.get("primary_key", [])
        if pk:
            table_pk_map[table_name] = pk[0]
    
    # Look for foreign key patterns
    for table in tables:
        table_name = table["table_name"]
        for col in table["column_names"]:
            col_lower = col.lower()
            
            # Skip if this is likely the table's own primary key
            if col in table.get("primary_key", []):
                continue
            
            # Check if column references another table
            for ref_table_name, ref_pk in table_pk_map.items():
                if ref_table_name == table_name:
                    continue
                    
                # Match patterns like: customer_id -> customers.customer_id
                # or order_id -> orders.order_id
                ref_table_singular = ref_table_name.rstrip('s')
                if col_lower == ref_pk.lower() or col_lower == f"{ref_table_singular}_{ref_pk}".lower():
                    foreign_keys.append({
                        "source_table": table_name,
                        "column_in_source_table": col,
                        "referenced_table": ref_table_name,
                        "column_in_referenced_table": ref_pk
                    })
                    break
    
    return foreign_keys


if __name__ == "__main__":
    config_inference()
