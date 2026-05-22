from pydantic import BaseModel, Field


class ProviderInfo(BaseModel):
    name: str = Field(description='Provider name, e.g. "xai" or "openai".')
    default_model: str = Field(description="Default model this provider falls back to.")
    model_name_prefixes: list[str] = Field(
        description="Canonical model-id prefixes this provider claims, e.g. ['grok'] or ['gpt','o1','o3','o4']."
    )


class HealthResponse(BaseModel):
    status: str = Field(description='Always "ok" for a 200 response.')
    commit: str = Field(description="Git commit (short) the running process was built from.")
    env: str = Field(description="Deployment env: development, staging, production.")
    service: str = Field(description="Service name.")
    default_model: str = Field(
        description=(
            "Default model id of the first registered provider. Preserved for "
            "back-compat with callers (e.g. pagehub-benchmarks) that probe a "
            "single default; ``providers`` carries the full list."
        ),
    )
    providers: list[ProviderInfo] = Field(
        default_factory=list,
        description="All registered backends with their default models and routing prefixes.",
    )


class MetricsResponse(BaseModel):
    requests_total: dict[str, int]
    errors_total: dict[str, int]
    input_tokens_total: int
    output_tokens_total: int
    ttfb_ms_histogram: dict
    total_ms_histogram: dict
