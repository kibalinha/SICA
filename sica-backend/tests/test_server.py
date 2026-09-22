import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from server import _parse_cors_origins, _sanitize_filename, app


class TestServerUtils:
    def test_parse_cors_origins_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            origins = _parse_cors_origins()
            assert len(origins) == 4
            assert "http://localhost:4173" in origins
            assert "http://localhost:5173" in origins

    def test_parse_cors_origins_wildcard(self) -> None:
        with patch.dict("os.environ", {"SICA_CORS_ORIGINS": "*"}, clear=True):
            origins = _parse_cors_origins()
            assert origins == ["*"]

    def test_parse_cors_origins_custom(self) -> None:
        with patch.dict(
            "os.environ", {"SICA_CORS_ORIGINS": "http://example.com,http://test.com"}, clear=True
        ):
            origins = _parse_cors_origins()
            assert len(origins) == 2
            assert "http://example.com" in origins
            assert "http://test.com" in origins

    def test_sanitize_filename(self) -> None:
        # Test with dangerous characters
        dangerous = "../../../etc/passwd"
        safe = _sanitize_filename(dangerous)
        assert "/" not in safe
        assert "." not in safe
        # A função deve remover os caminhos relativos e extensão
        assert "passwd" in safe or "etc" in safe

        # Test with safe characters
        safe_name = "audio_sample_123"
        result = _sanitize_filename(safe_name)
        assert result == safe_name

        # Test with length limit
        long_name = "a" * 200
        result = _sanitize_filename(long_name)
        assert len(result) <= 100


class TestServerEndpoints:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app)

    def test_root_endpoint(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_health_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "ai_engine" in data
        assert "device" in data
        assert "system" in data
        assert "model" in data
        assert "cache" in data

    @patch("server.process_audio_file")
    @patch("server._get_from_cache")
    def test_analyze_endpoint_success(self, mock_cache: MagicMock, mock_process: MagicMock, client: TestClient) -> None:
        mock_cache.return_value = None  # Desabilitar cache
        mock_process.return_value = {
            "duration_seconds": 4.5,
            "metrics": {
                "total_dba": 65.0,
                "harmonic_music_dba": 60.0,
                "noise_dba": 55.0,
                "snr_db": 5.0,
                "spectral_flatness": 0.1,
                "is_music_detected": True,
            },
            "recommendation": {
                "adjustment_db": 0,
                "status": "Nível equilibrado",
                "confidence": 0.8,
            },
            "ai_diagnostics": {
                "model_name": "Test Model",
                "probabilities": {"music": 50.0, "speech": 30.0, "ambient_noise": 20.0},
                "dominant_scene": "Música",
                "scene_description": "Test",
                "confidence": 0.8,
                "neural_mask_applied": False,
                "device": "cpu",
            },
            "spectrum_analysis": [],
        }

        # Create a fake audio file
        audio_content = b"fake audio data"
        files = {"file": ("test.webm", io.BytesIO(audio_content), "audio/webm")}

        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 200
        assert mock_process.called

    def test_analyze_endpoint_no_file(self, client: TestClient) -> None:
        response = client.post("/api/v1/analyze", data={})
        # FastAPI retorna 422 quando parâmetros obrigatórios estão faltando
        assert response.status_code == 422

    def test_analyze_endpoint_empty_file(self, client: TestClient) -> None:
        files = {"file": ("test.webm", io.BytesIO(b""), "audio/webm")}
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 400

    @patch("server.process_audio_file")
    @patch("server._get_from_cache")
    def test_analyze_endpoint_processing_error(self, mock_cache: MagicMock, mock_process: MagicMock, client: TestClient) -> None:
        mock_cache.return_value = None  # Desabilitar cache
        mock_process.side_effect = ValueError("Test error")

        audio_content = b"fake audio data"
        files = {"file": ("test.webm", io.BytesIO(audio_content), "audio/webm")}

        response = client.post("/api/v1/analyze", files=files)
        # O endpoint deve retornar 422 para erros de processamento (ValueError)
        assert response.status_code == 422
