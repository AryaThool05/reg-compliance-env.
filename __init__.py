"""
reg-compliance-env — Root package init.

Exports the main environment class and data models for easy import.
"""

try:
    from .server.environment import RegComplianceEnvironment as RegComplianceEnv
    from .models import RegComplianceObservation, RegComplianceAction, RegComplianceState
except ImportError:
    from server.environment import RegComplianceEnvironment as RegComplianceEnv
    from models import RegComplianceObservation, RegComplianceAction, RegComplianceState

# Convenience aliases matching OpenEnv naming conventions
Action = RegComplianceAction
Observation = RegComplianceObservation

__version__ = "1.0.0"
__all__ = [
    "RegComplianceEnv",
    "RegComplianceAction",
    "RegComplianceObservation",
    "RegComplianceState",
    "Action",
    "Observation",
]

