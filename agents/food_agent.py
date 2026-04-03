from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

# ollama model
model = OpenAIChatModel(
    model_name='qwen3.5:2b',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)

# bedrock model
# model = BedrockConverseModel('global.anthropic.claude-opus-4-6-v1')

food_agent = Agent(
    model=model,
    tools=[duckduckgo_search_tool()],
    instructions="""
        You are a food recommendation expert specializing in Chinese regional cuisines.
        Help users find the best dining experiences, from local street food and "hidden gems" to high-end Michelin-starred restaurants.
        Be knowledgeable about the Eight Great Cuisines (Cantonese, Sichuan, Jiangsu, etc.) and popular dishes in specific cities.
        If you cannot find the information needed to answer the user's question,
        you can use the duckduckgo_search_tool to search for the information.
    """,
)
