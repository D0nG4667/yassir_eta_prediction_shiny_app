# Redis server has latency issues if it not a high availability backend
# Use memory cache for now.

# import os

# from dotenv import load_dotenv

# import redis
# from cashews import cache

# load_dotenv(ENV_PATH)

# # Install persistent cache using Redis Cache
# url = os.getenv("REDIS_URL")
# username = os.getenv("REDIS_USERNAME")
# password = os.getenv("REDIS_PASSWORD")
# connection = redis.from_url(url=url, username=username,
#                             password=password, encoding="utf8", decode_responses=False)
# backend = R.RedisCache(connection=connection)

# Functions Redis Cache- only works for async functions and awaits
# full_url = os.getenv("REDIS_FULL_URL")
# cache.setup(full_url)


# @cache(ttl="10m", prefix='yassir_history_data') # Redis
