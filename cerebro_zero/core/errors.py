class CerebroError(Exception):
    """Base public error."""
class ConfigurationError(CerebroError): pass
class ProviderError(CerebroError): pass
class ExecutionError(CerebroError): pass
class SecurityError(CerebroError): pass
