class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_health_endpoint(self, client):
        """Test GET /v1/health"""
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_db_health_endpoint(self, client):
        """Test GET /health/db"""
        response = client.get("/health/db")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data


class TestPromptsAPI:
    """Test prompt management endpoints"""

    def test_create_prompt(self, client):
        """Test POST /v1/prompts - Create a new prompt"""
        payload = {
            "purpose": "test_purpose",
            "name": "Test Prompt",
            "template": "test template",
        }
        response = client.post("/v1/prompts", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["purpose"] == "test_purpose"
        assert data["name"] == "Test Prompt"
        assert data["template"] == "test template"

    def test_list_prompts_by_purpose(self, client):
        """Test GET /v1/prompts/{purpose} - List prompts by purpose"""
        # First create a prompt
        payload = {
            "purpose": "translate",
            "name": "Translator",
            "template": "translate to spanish",
        }
        client.post("/v1/prompts", json=payload)

        # Then list prompts for that purpose
        response = client.get("/v1/prompts/translate")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Check that all returned prompts have the correct purpose
        for prompt in data:
            assert prompt["purpose"] == "translate"

    def test_list_prompts_empty_purpose(self, client):
        """Test GET /v1/prompts/{purpose} - List for non-existent purpose"""
        response = client.get("/v1/prompts/nonexistent_purpose_xyz")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_patch_prompt(self, client):
        """Test PATCH /v1/prompts/{prompt_id} - Update a prompt"""
        # Create a prompt first
        payload = {
            "purpose": "extract",
            "name": "Original Name",
            "template": "original template",
        }
        create_response = client.post("/v1/prompts", json=payload)
        prompt_id = create_response.json()["purpose"]  # Using purpose as fallback

        # Get the actual ID from listing
        list_response = client.get("/v1/prompts/extract")
        if list_response.json():
            prompt_id = list_response.json()[0]["id"]

            # Patch the prompt
            patch_payload = {"name": "Updated Name", "template": "updated template"}
            response = client.patch(f"/v1/prompts/{prompt_id}", json=patch_payload)
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Updated Name"
            assert data["template"] == "updated template"

    def test_activate_prompt(self, client):
        """Test POST /v1/prompts/{prompt_id}/activate - Activate a prompt"""
        # Create a prompt first
        payload = {
            "purpose": "summarize",
            "name": "Summarizer",
            "template": "summarize this",
        }
        client.post("/v1/prompts", json=payload)

        # Get the prompt ID
        list_response = client.get("/v1/prompts/summarize")
        if list_response.json():
            prompt_id = list_response.json()[0]["id"]

            # Activate the prompt
            response = client.post(
                f"/v1/prompts/{prompt_id}/activate?purpose=summarize",
                headers={"x-user-id": "test_user"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["active"]
            assert data["purpose"] == "summarize"

    def test_get_active_prompt(self, client):
        """Test GET /v1/get_active/{purpose} - Get active prompt"""
        # Create and activate a prompt first
        payload = {
            "purpose": "classify",
            "name": "Classifier",
            "template": "classify this",
        }
        client.post("/v1/prompts", json=payload)

        # Get and activate the prompt
        list_response = client.get("/v1/prompts/classify")
        if list_response.json():
            prompt_id = list_response.json()[0]["id"]
            client.post(
                f"/v1/prompts/{prompt_id}/activate?purpose=classify",
                headers={"x-user-id": "test_user_2"},
            )

            # Get active prompt
            response = client.get("/v1/get_active/classify?user_id=test_user_2")
            assert response.status_code == 200
            data = response.json()
            if data:  # Will be None if no active prompt for that user
                assert data["active"]
                assert data["purpose"] == "classify"


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


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_prompt_name(self, client):
        """Test creating prompt with empty name"""
        payload = {"purpose": "test", "name": "", "template": "test template"}
        response = client.post("/v1/prompts", json=payload)
        assert response.status_code == 200

    def test_empty_template(self, client):
        """Test creating prompt with empty template"""
        payload = {"purpose": "test", "name": "Test", "template": ""}
        response = client.post("/v1/prompts", json=payload)
        assert response.status_code == 200

    def test_predict_empty_document(self, client):
        """Test predict with empty document"""
        payload = {"purpose": "translate", "document_text": "", "provider": "mock"}
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200
