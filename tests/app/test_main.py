"""Tests for src/app/main.py"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from src.app.main import app, lifespan


class TestLifespan:
    """Test suite for the lifespan context manager."""
    
    def test_lifespan_initializes_state(self):
        """Test that lifespan initializes the recommender state."""
        import asyncio
        
        async def run_test():
            with patch("src.app.main.state") as mock_state:
                async with lifespan(app):
                    mock_state.initialize.assert_called_once()
        
        asyncio.run(run_test())


class TestApp:
    """Test suite for the FastAPI application."""
    
    def test_app_creation(self):
        """Test that the FastAPI app is created successfully."""
        assert app.title == "Boardom Buster API"
        assert hasattr(app, "router")
    
    def test_app_has_recommend_router(self):
        """Test that the recommend router is included."""
        routes = [route.path for route in app.routes]
        assert "/recommend" in routes
    
    def test_app_client_creation(self):
        """Test that a test client can be created."""
        with patch("src.app.dependencies.state"):
            client = TestClient(app)
            assert client is not None
