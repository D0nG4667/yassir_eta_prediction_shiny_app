# Redis server has latency issues if it not a high availability backend
# Use memory cache for now.

# from fastapi_cache.backends.redis import RedisBackend

# from redis import asyncio as aioredis

# from utils.config import ENV_PATH

# load_dotenv(ENV_PATH)

# @asynccontextmanager
# async def lifespan(_: FastAPI) -> AsyncIterator[None]:
#     url = os.getenv("REDIS_URL")
#     username = os.getenv("REDIS_USERNAME")
#     password = os.getenv("REDIS_PASSWORD")
#     redis = aioredis.from_url(url=url, username=username,
#                               password=password, encoding="utf8", decode_responses=False)  # fastapi-cache2 0.2.2 needs decode_responses to be False to avoid breaking changes
#     FastAPICache.init(RedisBackend(redis), prefix="yassir-fastapi-cache")
#     yield
