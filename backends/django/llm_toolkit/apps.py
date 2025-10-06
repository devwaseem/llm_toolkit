try:
    from django.apps import AppConfig
except ImportError as exc:
    raise ImportError(
        "Could not import Django. Make sure to install Django before "
        "using llm_toolkit.backends.django"
    ) from exc


class BackendConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "llm_toolkit.backends.django.llm_toolkit"
