"""
Example: Parallel debate execution using async LLM calls.

This demonstrates how to run Advocate and Opposer agents in parallel
to reduce total debate time while respecting rate limits.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm import call_llm_async
from prompts import ADVOCATE_SYSTEM_PROMPT, OPPOSER_SYSTEM_PROMPT


async def run_parallel_debate(claim: str):
    """Run advocate and opposer in parallel."""
    
    advocate_msg = (
        f"Debate Claim: \"{claim}\"\n\n"
        "Formulate your formal opening case. Present EXACTLY 3 arguments strictly IN FAVOR of this claim. "
        "Output your arguments in the specified JSON format."
    )
    
    opposer_msg = (
        f"Debate Claim: \"{claim}\"\n\n"
        "Formulate your formal opening case. Present EXACTLY 3 arguments strictly AGAINST this claim. "
        "Output your arguments in the specified JSON format."
    )
    
    # Run both agents in parallel
    print(f"Running parallel debate for: {claim}")
    print("Starting Advocate and Opposer simultaneously...")
    
    results = await asyncio.gather(
        call_llm_async(ADVOCATE_SYSTEM_PROMPT, advocate_msg),
        call_llm_async(OPPOSER_SYSTEM_PROMPT, opposer_msg),
        return_exceptions=True
    )
    
    pro_result, con_result = results
    
    if isinstance(pro_result, Exception):
        print(f"Advocate failed: {pro_result}")
    else:
        print(f"Advocate completed: {len(pro_result)} chars")
    
    if isinstance(con_result, Exception):
        print(f"Opposer failed: {con_result}")
    else:
        print(f"Opposer completed: {len(con_result)} chars")
    
    return pro_result, con_result


if __name__ == "__main__":
    claim = "AI should be used in judicial decisions"
    asyncio.run(run_parallel_debate(claim))