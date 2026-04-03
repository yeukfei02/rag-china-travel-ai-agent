from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

# ollama model
model = OpenAIChatModel(
    model_name='qwen3.5:4b',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)

# bedrock model
# model = BedrockConverseModel('global.anthropic.claude-opus-4-6-v1')

shopping_agent = Agent(
    model=model,
    tools=[duckduckgo_search_tool()],
    instructions="""
        You are a shopping expert for the China market.
        Help users navigate everything from high-end malls (like Taikoo Hui or SKP) and tech markets (like Huaqiangbei) to local souvenir shops.
        Provide advice on duty-free shopping, popular Chinese brands, and where to find unique local products.
        If you cannot find the information needed to answer the user's question,
        you can use the duckduckgo_search_tool to search for the information.
    """,
)
