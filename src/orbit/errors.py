"""Exception types used across ORBIT."""


class OrbitError(Exception):
    """Base class for all expected ORBIT errors."""


class ConfigError(OrbitError):
    """Invalid or unsafe configuration (mappings, shortcuts, parameters)."""


class ActionError(OrbitError):
    """An action failed while executing."""


class ActionUnsupported(ActionError):
    """The action is not available on this platform / configuration."""


class ModelError(OrbitError):
    """A required ML model file is missing or could not be downloaded."""
