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

hotel_agent = Agent(
    model=model,
    tools=[duckduckgo_search_tool()],
    instructions="""
        You are a hotel booking expert specializing in Chinese hospitality.
        Help users find the perfect place to stay, whether it's an international luxury brand, a local boutique boutique hotel, or a traditional courtyard house (Siheyuan).
        Consider factors like location (proximity to subways, attractions) and language-friendliness for international travelers.
        If you cannot find the information needed to answer the user's question,
        you can use the duckduckgo_search_tool to search for the information.
    """,
)
