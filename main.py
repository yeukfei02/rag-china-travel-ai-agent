from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.bedrock import BedrockConverseModel
from agents.flight_agent import flight_agent
from agents.hotel_agent import hotel_agent
from agents.food_agent import food_agent
from agents.shopping_agent import shopping_agent
from agents.itinerary_agent import itinerary_agent
from rag.get_data import get_pinecone_data

load_dotenv()

# ollama model
model = OpenAIChatModel(
    model_name='minimax-m2.7:cloud',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)

# bedrock model
# model = BedrockConverseModel('global.anthropic.claude-opus-4-6-v1')

# orchestrator agent
orchestrator_agent = Agent(
    model=model,
    name="orchestrator",
    system_prompt="""
        You are an expert China travel assistant. Your goal is to help users plan their trips to China by providing accurate, helpful, and localized information.
        You have deep knowledge of Chinese cities, culture, food, and travel logistics such as:
        - Visa policies (including transit visa-free entry).
        - Payment methods (setting up Alipay/WeChat Pay with foreign cards, Digital RMB).
        - Transportation (High-speed rail booking, Didi, local subways).
        - Must-see attractions and hidden gems.
        - Local etiquette and travel tips.

        You can call the following tools to get information for the user:
        - flight_search(query: str): to search for flights to or within China.
        - hotel_search(query: str): to search for hotels in Chinese cities.
        - food_search(query: str): to search for local Chinese cuisine and restaurants.
        - shopping_search(query: str): to search for shopping districts, deals, and tax-free info.
        - itinerary_search(query: str): to create detailed travel plans for different Chinese regions.
        - rag_search(query: str): to search for specific travel tips, visa rules, payment guides, or FAQs from our private knowledge base which contains detailed China-specific data.

        When the user asks a question, determine their specific needs and use the appropriate tools. If the query is about specific Chinese policies or local tips, prioritize using `rag_search`.
    """,
)


@orchestrator_agent.tool
async def flight_search(ctx: RunContext[None], query: str):
    flight_result = await flight_agent.run(query, usage=ctx.usage)
    return flight_result.output


@orchestrator_agent.tool
async def hotel_search(ctx: RunContext[None], query: str):
    hotel_result = await hotel_agent.run(query, usage=ctx.usage)
    return hotel_result.output


@orchestrator_agent.tool
async def food_search(ctx: RunContext[None], query: str):
    food_result = await food_agent.run(query, usage=ctx.usage)
    return food_result.output


@orchestrator_agent.tool
async def shopping_search(ctx: RunContext[None], query: str):
    shopping_result = await shopping_agent.run(query, usage=ctx.usage)
    return shopping_result.output


@orchestrator_agent.tool
async def itinerary_search(ctx: RunContext[None], query: str):
    itinerary_result = await itinerary_agent.run(query, usage=ctx.usage)
    return itinerary_result.output


@orchestrator_agent.tool
async def rag_search(ctx: RunContext[None], query: str):
    results = get_pinecone_data(query)

    result = ""
    if results:
        context = "\n".join([f"- {item['text']}" for item in results])
        result = f"Retrieved knowledge:\n{context}"
    else:
        result = "No relevant knowledge found."

    return result

# query = "Plan a 5 day trip to Guangzhou from Singapore. Reply me in chinese."

# result = orchestrator_agent.run_sync(
#     query,
#     usage_limits=UsageLimits(request_limit=10, total_tokens_limit=1000),
# )
# print(f"result.output = {result.output}")

app = orchestrator_agent.to_web()
