from fastapi_app.python.db import sync_inserts, sync_queries


def init_db():
    if not sync_queries.check_if_weather_data_exists():
        sync_inserts.update_weather_db()
    sync_inserts.create_default_user_account()


if __name__ == '__main__':
    init_db()