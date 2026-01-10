import litellm
from dotenv import load_dotenv
import os
import warnings


def get_LLM_response(query: str, retrieved_info: str) -> str:
    """Get a concise LLM response grounded on retrieved info."""

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    prompt = (
        "Give the simplest summary or explanation for the user query using the retrieved "
        "information from their documents. Limit the answer to under 2 sentences.\n\n"
        f"Query: {query}\n"
        f"Retrieved info: {retrieved_info}"
    )

    completion = litellm.completion(
        model="openai/gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key,
    )

    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

    # Avoid printing the pydantic object (emits warnings); log only the text content
    return completion.choices[0].message["content"].strip()