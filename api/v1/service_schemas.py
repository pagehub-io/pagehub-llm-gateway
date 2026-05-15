from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description='Always "ok" for a 200 response.')
    commit: str = Field(description="Git commit (short) the running process was built from.")
    env: str = Field(description="Deployment env: development, staging, production.")
    service: str = Field(description="Service name.")
    default_model: str = Field(description="Default Grok model id surfaced to callers.")


class MetricsResponse(BaseModel):
    requests_total: dict[str, int]
    errors_total: dict[str, int]
    input_tokens_total: int
    output_tokens_total: int
    ttfb_ms_histogram: dict
    total_ms_histogram: dict
