from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "development"
    service_name: str = "pagehub-llm-gateway"
    git_commit: str = "unknown"

    # Caller auth (Anthropic-shaped surface)
    gateway_auth_token: str = ""
    # Admin endpoints auth — separate token so a leaked caller token can't read history.
    admin_auth_token: str = ""

    # xAI Grok backend
    xai_api_key: str = ""
    grok_base_url: str = "https://api.x.ai/v1"
    grok_default_model: str = "grok-4"

    # OpenAI backend (diagnostic second provider — see README "End-to-end
    # verification"). The gateway routes by canonical model id; xAI claims
    # ``grok-*`` and OpenAI claims ``gpt-*`` / ``o1-*`` / ``o3-*`` / ``o4-*``.
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_default_model: str = "gpt-5"

    # Database — empty string disables DB logging entirely (the engine falls back to
    # an in-memory recorder). Required in production.
    database_url: str = ""

    # When False, redact text / tool input / tool result content from the request log.
    # Default True locally so debugging works; production deployments override to False.
    log_content: bool = True

    # Limits / timeouts
    request_body_max_bytes: int = 4 * 1024 * 1024
    provider_timeout_seconds: float = 600.0

    port: int = 4011
    log_level: str = "INFO"


settings = Settings()
