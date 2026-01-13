"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client: TestClient):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client: TestClient):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_version(self, client: TestClient):
        """Health endpoint should include API version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_includes_features(self, client: TestClient):
        """Health endpoint should include available features."""
        response = client.get("/health")
        data = response.json()
        assert "features" in data
        assert data["features"]["mcmc_sampling"] is True
        assert data["features"]["diffusion_sampling"] is True


class TestConfigEndpoint:
    """Tests for the /config endpoint."""

    def test_config_returns_200(self, client: TestClient):
        """Config endpoint should return 200 OK."""
        response = client.get("/config")
        assert response.status_code == 200

    def test_config_includes_lattice_size(self, client: TestClient):
        """Config should include lattice size."""
        response = client.get("/config")
        data = response.json()
        assert "lattice_size" in data
        assert isinstance(data["lattice_size"], int)

    def test_config_includes_critical_temperature(self, client: TestClient):
        """Config should include critical temperature."""
        response = client.get("/config")
        data = response.json()
        assert "T_critical" in data
        # T_c ≈ 2.269 for 2D Ising model
        assert 2.2 < data["T_critical"] < 2.3


class TestOpenAPIDocumentation:
    """Tests for API documentation."""

    def test_openapi_docs_available(self, client: TestClient):
        """OpenAPI docs should be available at /docs."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_available(self, client: TestClient):
        """OpenAPI schema should be available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
