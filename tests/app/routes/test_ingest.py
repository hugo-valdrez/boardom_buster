"""Tests for src/app/routes/ingest.py"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import BackgroundTasks

from src.app.routes.ingest import router, run_ingestion_pipeline, trigger_ingestion


class TestRunIngestionPipeline:
    """Test suite for the run_ingestion_pipeline function."""

    @patch("src.app.routes.ingest.continuous_fetch")
    @patch("src.app.routes.ingest.consolidation")
    @patch("src.app.routes.ingest.state")
    def test_run_ingestion_pipeline(self, mock_state, mock_consolidation, mock_continuous_fetch):
        """Test that the ingestion pipeline runs all steps in order."""
        import asyncio

        async def run_test():
            mock_continuous_fetch.return_value = AsyncMock()

            await run_ingestion_pipeline()

            mock_continuous_fetch.assert_called_once()
            mock_consolidation.assert_called_once()
            mock_state.reload.assert_called_once()

        asyncio.run(run_test())


class TestTriggerIngestion:
    """Test suite for the trigger_ingestion endpoint."""

    def test_trigger_ingestion(self):
        """Test that ingestion can be triggered in the background."""
        import asyncio

        async def run_test():
            background_tasks = Mock(spec=BackgroundTasks)

            result = await trigger_ingestion(background_tasks)

            assert result == {"status": "Ingestion pipeline started in background"}
            background_tasks.add_task.assert_called_once()

        asyncio.run(run_test())
