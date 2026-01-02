"""
Arctic SQL Generation - Reproduction Script
Uses the Snowflake/arctic model to generate SQL from natural language questions.
"""

import llm

# Sample database schema - loaded from file
SAMPLE_SCHEMA_PATH = "schema_E_commerce.txt"

def load_schema(path: str) -> str:
    """Load database schema from file."""
    with open(path, "r") as f:
        return f.read()

SAMPLE_SCHEMA = load_schema(SAMPLE_SCHEMA_PATH)

# Sample question with evidence
SAMPLE_QUESTION = '''
Can you calculate the 5-day symmetric moving average of predicted toy sales for December 5 to 8, 2018, using daily sales data from January 1, 2017, to August 29, 2018, with a simple linear regression model? Finally provide the sum of those four 5-day moving averages?
'''

# Template file paths
SYSTEM_TEMPLATE_PATH = "arctic_system_template.txt"
USER_TEMPLATE_PATH = "arctic_user_template.txt"

# Assistant prefix to prime the model's response format
ASSISTANT_PREFIX = """Let me solve this step by step.
<think>
"""


def load_template(path: str) -> str:
    """Load a template from file."""
    with open(path, "r") as f:
        return f.read()


def build_messages(
    schema: str, 
    question: str, 
    use_assistant_prefix: bool = True
) -> list[dict[str, str]]:
    """
    Build the messages list for the chat API.
    
    Structure:
    - System: Expert role description
    - User: Database schema, question, and instructions with output format
    - Assistant (optional): Prefix to prime the model's response format
    
    Args:
        schema: Database schema (CREATE TABLE statements)
        question: Natural language question
        use_assistant_prefix: If True, add assistant prefix to guide response format
    """
    # Load templates
    system_template = load_template(SYSTEM_TEMPLATE_PATH)
    user_template = load_template(USER_TEMPLATE_PATH)
    
    # Fill in placeholders in user template
    user_content = user_template.replace("{Database Schema}", schema.strip())
    user_content = user_content.replace("{evidence + question}", question.strip())
    
    messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": user_content},
    ]
    
    # Add assistant prefix to prime the model's response
    if use_assistant_prefix and ASSISTANT_PREFIX:
        messages.append({"role": "assistant", "content": ASSISTANT_PREFIX})
    
    return messages


def generate_sql(
    schema: str, 
    question: str, 
    temperature: float = 0.0,
    use_assistant_prefix: bool = True
) -> str:
    messages = build_messages(schema, question, use_assistant_prefix=use_assistant_prefix)
    response = llm.chat(messages, temperature=temperature)
    
    # If we used assistant prefix, prepend it to the response for proper parsing
    # (the model continues from the prefix, so its response doesn't include it)
    if use_assistant_prefix and ASSISTANT_PREFIX:
        response = ASSISTANT_PREFIX + response
    
    return response


def extract_sql(response: str) -> str:
    """Extract the SQL query from the model's response."""
    # Look for SQL in <answer> tags first
    if "<answer>" in response and "</answer>" in response:
        answer_start = response.find("<answer>") + len("<answer>")
        answer_end = response.find("</answer>")
        answer = response[answer_start:answer_end]
    else:
        answer = response
    
    # Extract SQL from ```sql blocks
    if "```sql" in answer:
        sql_start = answer.find("```sql") + len("```sql")
        sql_end = answer.find("```", sql_start)
        return answer[sql_start:sql_end].strip()
    
    return answer.strip()


if __name__ == "__main__":
    print("=" * 60)
    print("Arctic SQL Generation Demo")
    print("=" * 60)
    print("\n📋 Schema:")
    print(SAMPLE_SCHEMA)
    print("\n❓ Question:")
    print(SAMPLE_QUESTION)
    print("\n🤖 Generating SQL...\n")
    
    response = generate_sql(SAMPLE_SCHEMA, SAMPLE_QUESTION)
    
    print("=" * 60)
    print("Full Response:")
    print("=" * 60)
    print(response)
    
    print("\n" + "=" * 60)
    print("Extracted SQL:")
    print("=" * 60)
    sql = extract_sql(response)
    print(sql)
