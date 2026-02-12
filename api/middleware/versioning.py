"""API versioning middleware for backward compatibility and version management."""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Optional
import re


class APIVersionMiddleware(BaseHTTPMiddleware):
    """Middleware to handle API versioning through headers or URL paths."""

    def __init__(
        self,
        app,
        default_version: str = "v1",
        supported_versions: Optional[list] = None,
        version_header: str = "X-API-Version",
    ):
        """Initialize API version middleware.
        
        Args:
            app: FastAPI application instance
            default_version: Default API version to use if not specified
            supported_versions: List of supported API versions
            version_header: Header name for API version
        """
        super().__init__(app)
        self.default_version = default_version
        self.supported_versions = supported_versions or ["v1"]
        self.version_header = version_header

    async def dispatch(self, request: Request, call_next: Callable):
        """Process request and extract API version.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
        
        Returns:
            Response from the next middleware/endpoint
        """
        # Try to get version from header
        version = request.headers.get(self.version_header)

        # Try to extract version from URL path (e.g., /v1/endpoint)
        if not version:
            path_version = self._extract_version_from_path(request.url.path)
            if path_version:
                version = path_version

        # Use default version if not specified
        if not version:
            version = self.default_version

        # Validate version
        if version not in self.supported_versions:
            raise HTTPException(
                status_code=400,
                detail=f"API version '{version}' not supported. Supported versions: {', '.join(self.supported_versions)}"
            )

        # Store version in request state for access in endpoints
        request.state.api_version = version

        # Call next middleware/endpoint
        response = await call_next(request)

        # Add version to response headers
        response.headers[self.version_header] = version

        return response

    def _extract_version_from_path(self, path: str) -> Optional[str]:
        """Extract API version from URL path.
        
        Args:
            path: URL path
        
        Returns:
            API version if found, None otherwise
        """
        # Match patterns like /v1/, /v2/, etc.
        match = re.search(r'/v(\d+)/', path)
        if match:
            return f"v{match.group(1)}"
        return None


class DeprecationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle API deprecation warnings."""

    def __init__(
        self,
        app,
        deprecated_versions: Optional[dict] = None,
    ):
        """Initialize deprecation middleware.
        
        Args:
            app: FastAPI application instance
            deprecated_versions: Dict mapping versions to sunset dates
                Example: {"v1": "2025-12-31", "v2": "2026-06-30"}
        """
        super().__init__(app)
        self.deprecated_versions = deprecated_versions or {}

    async def dispatch(self, request: Request, call_next: Callable):
        """Process request and add deprecation headers if needed.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
        
        Returns:
            Response with deprecation headers if applicable
        """
        response = await call_next(request)

        # Check if API version is deprecated
        if hasattr(request.state, 'api_version'):
            version = request.state.api_version
            if version in self.deprecated_versions:
                sunset_date = self.deprecated_versions[version]
                response.headers['Deprecation'] = 'true'
                response.headers['Sunset'] = sunset_date
                response.headers['Link'] = '<https://docs.example.com/migration>; rel="deprecation"'

        return response


def get_api_version(request: Request) -> str:
    """Helper function to get API version from request.
    
    Args:
        request: FastAPI request object
    
    Returns:
        API version string
    """
    return getattr(request.state, 'api_version', 'v1')


# Example usage in server.py:
# 
# from api.middleware.versioning import APIVersionMiddleware, DeprecationMiddleware
# 
# app = FastAPI()
# 
# # Add versioning middleware
# app.add_middleware(
#     APIVersionMiddleware,
#     default_version="v1",
#     supported_versions=["v1", "v2"],
#     version_header="X-API-Version"
# )
# 
# # Add deprecation middleware
# app.add_middleware(
#     DeprecationMiddleware,
#     deprecated_versions={"v1": "2025-12-31"}
# )
# 
# # In endpoints, access version:
# @app.get("/metrics")
# async def get_metrics(request: Request):
#     version = get_api_version(request)
#     if version == "v1":
#         return {"legacy": "data"}
#     else:
#         return {"new": "data"}
