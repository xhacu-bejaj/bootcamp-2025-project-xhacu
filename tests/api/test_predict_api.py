class TestPredictAPI:
    """Test prediction/processing endpoints"""

    def test_predict_with_mock_provider(self, client):
        """Test POST /v1/predict with mock provider"""
        payload = {
            "purpose": "translate",
            "document_text": "Hello world",
            "provider": "mock",
        }
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "output_text" in data
        assert "model_info" in data
        assert "prompt_id" in data
        assert "prompt_version" in data
        assert "latency_ms" in data
        assert data["model_info"]["model"] == "mock"
        assert "[MOCK OUTPUT]" in data["output_text"]

    def test_predict_with_document_variations(self, client):
        """Test predict with various documents"""
        documents = [
            "Short text",
            "This is a longer piece of text that should be processed",
            "Special characters: @#$%^&*()",
            "Numbers: 123 456 789",
        ]

        for doc in documents:
            payload = {"purpose": "translate", "document_text": doc, "provider": "mock"}
            response = client.post("/v1/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert doc in data["output_text"]

    def test_predict_required_fields(self, client):
        """Test predict with missing required fields"""
        # Missing document_text
        payload = {"purpose": "translate", "provider": "mock"}
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 422  # Validation error

        # Missing purpose
        payload = {"document_text": "test", "provider": "mock"}
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 422  # Validation error


class TestPredictAdvanced:
    """Advanced tests for prediction endpoint"""

    def test_predict_returns_correct_prompt_id(self, client):
        """Test that predict returns the correct prompt ID"""
        payload = {
            "purpose": "translate",
            "document_text": "Test document",
            "provider": "mock",
        }
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prompt_id" in data
        assert data["prompt_id"] is not None

    def test_predict_with_custom_temperature(self, client):
        """Test predict with custom LLM parameters"""
        payload = {
            "purpose": "translate",
            "document_text": "Test document",
            "provider": "mock",
            "params": {"model": "gpt-4o-mini", "temperature": 0.7},
        }
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "model_info" in data

    def test_predict_returns_latency(self, client):
        """Test that predict returns latency information"""
        payload = {
            "purpose": "translate",
            "document_text": "Test document",
            "provider": "mock",
        }
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_predict_with_very_long_document(self, client):
        """Test predict with very long document"""
        long_doc = "This is a test document. " * 500
        payload = {
            "purpose": "translate",
            "document_text": long_doc,
            "provider": "mock",
        }
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert long_doc in data["output_text"]

    def test_predict_model_info_structure(self, client):
        """Test that model_info has correct structure"""
        payload = {"purpose": "translate", "document_text": "Test", "provider": "mock"}
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        model_info = data["model_info"]
        assert "model" in model_info
        assert "temperature" in model_info
        assert isinstance(model_info["temperature"], (int, float))


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_predict_empty_document(self, client):
        """Test predict with empty document"""
        payload = {"purpose": "translate", "document_text": "", "provider": "mock"}
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200


class TestResponseValidation:
    """Tests for response validation"""

    def test_predict_response_has_required_fields(self, client):
        """Test that predict response has all required fields"""
        payload = {"purpose": "translate", "document_text": "Test", "provider": "mock"}
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        required_fields = [
            "output_text",
            "model_info",
            "prompt_id",
            "prompt_version",
            "latency_ms",
        ]
        for field in required_fields:
            assert field in data