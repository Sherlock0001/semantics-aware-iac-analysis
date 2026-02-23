import pandas as pd
import time
from tqdm import tqdm
#from openai import OpenAI
from typing import Optional
import openai
import re
from dotenv import load_dotenv
import os
import openpyxl

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Error: OpenAI API key not found in environment variables.")
    exit(1)

openai.api_key = openai_api_key
# Initialize OpenAI client
#client = OpenAI()

# ---------- Prompt template ----------
PROMPT_TEMPLATE = """
You are an expert in Infrastructure as Code (IaC) security.
You are analyzing a smelly Puppet code snippet.

Task (extractive only):
1- Identify the exact line number(s) that contain the security smell.
2- Provide the corresponding CWE ID and its official CWE description.

Constraints:
- Do not explain your reasoning.
- Return only the requested fields.

Input:
{code}
"""

# ---------- LLM call ----------
def label_cwe(code_snippet: str) -> Optional[str]:
    """
    Calls GPT-4o to extract smelly lines and CWE.
    Returns raw text output or None if failed.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": PROMPT_TEMPLATE.format(code=code_snippet)}
            ],
            temperature=0.0,
            max_tokens=200,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"ERROR: {str(e)}"

# ---------- Main pipeline ----------
def process_xlsx(
    input_path: str,
    output_path: str,
    code_column: str = "Script.Content",
):
    df = pd.read_excel(input_path)

    if code_column not in df.columns:
        raise ValueError(f"Column '{code_column}' not found in input file.")

    results = []

    for idx, code in tqdm(df[code_column].items(), total=len(df)):
        if not isinstance(code, str) or not code.strip():
            results.append("ERROR: empty or invalid code snippet")
            continue

        output = label_cwe(code)
        results.append(output)

        # Gentle rate limiting (important for stability)
        time.sleep(0.2)

    df["cwe_labeling_output"] = results
    df.to_excel(output_path, index=False)

    print(f"✅ Processing completed. Results saved to: {output_path}")

# ---------- Run ----------
if __name__ == "__main__":
    process_xlsx(
        input_path="puppet_unseen_dataset.xlsx",
        output_path="puppet_code_with_cwe_labels.xlsx",
        code_column="Script.Content",
    )
