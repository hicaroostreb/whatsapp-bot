import os
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache


class CacheManager:
    @staticmethod
    def initialize_cache():
        cache_dir = os.path.join(os.getcwd(), "bot", "bot-cache")
        os.makedirs(cache_dir, exist_ok=True)
        set_llm_cache(
            SQLiteCache(database_path=os.path.join(cache_dir, "langchain.db"))
        )
