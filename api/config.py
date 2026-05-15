from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "development"
    service_name: str = "pagehub-llm-gateway"
    git_commit: str = "unknown"

    gateway_auth_token: str = ""

    xai_api_key: str = ""
    grok_base_url: str = "https://api.x.ai/v1"
    grok_default_model: str = "grok-4"

    request_body_max_bytes: int = 4 * 1024 * 1024
    provider_timeout_seconds: float = 600.0

    log_content: bool = False

    port: int = 4011
    log_level: str = "INFO"


settings = Settings()
