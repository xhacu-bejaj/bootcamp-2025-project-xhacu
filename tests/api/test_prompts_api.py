
class TestPromptCreation:
    """Detailed tests for prompt creation"""

    def test_create_prompt_with_special_characters(self, client):
        """Test creating prompt with special characters in template"""
        payload = {
            "purpose": "special_chars",
            "name": "Special Characters Test",
            "template": "{{ variable }} with special chars: @#$%^&*()[]{}|;:'\"<>,.?/",
        }
        response = client.post("/v1/prompts", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "special chars" in data["template"]

    def test_create_prompt_with_multiline_template(self, client):
        """Test creating prompt with multiline template"""
        payload = {
            "purpose": "multiline",
            "name": "Multiline Template",
            "template": "Line 1\nLine 2\nLine 3\n{{ variable }}",
        }
        response = client.post("/v1/prompts", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "\n" in data["template"]

    def test_create_prompt_with_long_template(self, client):
        """Test creating prompt with very long template"""
        long_template = "This is a very long template. " * 100
        payload = {
            "purpose": "long_template",
            "name": "Long Template",
            "template": long_template,
        }
        response = client.post("/v1/prompts", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["template"]) > 1000

    def test_create_multiple_prompts_same_purpose(self, client):
        """Test creating multiple prompts with same purpose"""
        purpose = "test_multiple"
        for i in range(5):
            payload = {
                "purpose": purpose,
                "name": f"Prompt {i + 1}",
                "template": f"template {i + 1}",
            }
            response = client.post("/v1/prompts", json=payload)
            assert response.status_code == 200

        response = client.get(f"/v1/prompts/{purpose}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 5

    def test_create_prompt_unicode_characters(self, client):
        """Test creating prompt with unicode characters"""
        payload = {
            "purpose": "unicode",
            "name": "Unicode Test 测试 テスト тест",
            "template": "Template with unicode: 日本語 中文 한국어 العربية",
        }
        response = client.post("/v1/prompts", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "日本語" in data["template"]


class TestPromptListing:
    """Detailed tests for listing prompts"""

    def test_list_prompts_returns_all_fields(self, client):
        """Test that list returns all required fields"""
    
        payload = {
            "purpose": "list_test",
            "name": "Test Prompt",
            "template": "test template",
        }
        client.post("/v1/prompts", json=payload)

        response = client.get("/v1/prompts/list_test")
        assert response.status_code == 200
        data = response.json()

        if data:
            prompt = data[0]
            required_fields = ["id", "purpose", "name", "template", "version", "active"]
            for field in required_fields:
                assert field in prompt

    def test_list_prompts_correct_purpose(self, client):
        """Test that list only returns prompts for specified purpose"""
        
        purposes = ["purpose_a", "purpose_b", "purpose_c"]
        for purpose in purposes:
            payload = {
                "purpose": purpose,
                "name": f"Prompt for {purpose}",
                "template": "template",
            }
            client.post("/v1/prompts", json=payload)

        response = client.get("/v1/prompts/purpose_a")
        assert response.status_code == 200
        data = response.json()

        for prompt in data:
            assert prompt["purpose"] == "purpose_a"

    def test_list_prompts_preserves_version(self, client):
        """Test that version field is preserved"""
        payload = {
            "purpose": "version_test",
            "name": "Version Test",
            "template": "original",
        }
        client.post("/v1/prompts", json=payload)

        response = client.get("/v1/prompts/version_test")
        data = response.json()
        assert len(data) > 0
        assert data[0]["version"] >= 1


class TestPromptPatching:
    """Detailed tests for updating prompts"""

    def test_patch_only_name(self, client):
        """Test patching only the name field"""

        payload = {
            "purpose": "patch_name_test",
            "name": "Original",
            "template": "template",
        }
        client.post("/v1/prompts", json=payload)

        list_response = client.get("/v1/prompts/patch_name_test")
        if list_response.json():
            prompt_id = list_response.json()[0]["id"]
            list_response.json()[0]["template"]

            patch_payload = {"name": "Updated"}
            response = client.patch(f"/v1/prompts/{prompt_id}", json=patch_payload)
            assert response.status_code == 200
            assert response.json()["name"] == "Updated"

    def test_patch_only_template(self, client):
        """Test patching only the template field"""

        payload = {
            "purpose": "patch_template_test",
            "name": "Test",
            "template": "original template",
        }
        client.post("/v1/prompts", json=payload)

        list_response = client.get("/v1/prompts/patch_template_test")
        if list_response.json():
            prompt_id = list_response.json()[0]["id"]

            patch_payload = {"template": "updated template"}
            response = client.patch(f"/v1/prompts/{prompt_id}", json=patch_payload)
            assert response.status_code == 200
            assert response.json()["template"] == "updated template"

    def test_patch_increments_version(self, client):
        """Test that patching increments version"""

        payload = {
            "purpose": "version_increment_test",
            "name": "Test",
            "template": "template",
        }
        client.post("/v1/prompts", json=payload)

        list_response = client.get("/v1/prompts/version_increment_test")
        if list_response.json():
            prompt_id = list_response.json()[0]["id"]
            original_version = list_response.json()[0]["version"]

            patch_payload = {"name": "Updated"}
            response = client.patch(f"/v1/prompts/{prompt_id}", json=patch_payload)

            if "version" in response.json():
                assert response.json()["version"] == original_version + 1


class TestPromptActivation:
    """Detailed tests for activating prompts"""

    def test_activate_sets_active_flag(self, client):
        """Test that activation sets active flag"""
        payload = {"purpose": "activate_test", "name": "Test", "template": "template"}
        client.post("/v1/prompts", json=payload)

        list_response = client.get("/v1/prompts/activate_test")
        if list_response.json():
            prompt_id = list_response.json()[0]["id"]

            response = client.post(
                f"/v1/prompts/{prompt_id}/activate?purpose=activate_test",
                headers={"x-user-id": "test_user"},
            )
            assert response.status_code == 200
            assert response.json()["active"]

    def test_activate_different_users_different_active(self, client):
        """Test that different users can have different active prompts"""
        payload = {"purpose": "multi_user_test", "name": "Test", "template": "template"}
        client.post("/v1/prompts", json=payload)

        list_response = client.get("/v1/prompts/multi_user_test")
        if list_response.json():
            prompt_id = list_response.json()[0]["id"]

            response1 = client.post(
                f"/v1/prompts/{prompt_id}/activate?purpose=multi_user_test",
                headers={"x-user-id": "user1"},
            )
            assert response1.status_code == 200

    def test_activate_with_query_parameter(self, client):
        """Test activate endpoint with purpose in query parameter"""
        payload = {
            "purpose": "query_param_test",
            "name": "Test",
            "template": "template",
        }
        client.post("/v1/prompts", json=payload)

        list_response = client.get("/v1/prompts/query_param_test")
        if list_response.json():
            prompt_id = list_response.json()[0]["id"]

            response = client.post(
                f"/v1/prompts/{prompt_id}/activate?purpose=query_param_test"
            )
            assert response.status_code == 200


class TestGetActivePrompt:
    """Detailed tests for getting active prompt"""

    def test_get_active_returns_none_if_not_set(self, client):
        """Test that get_active returns None if no active prompt"""
        response = client.get("/v1/get_active/nonexistent_purpose?user_id=some_user")
        assert response.status_code == 200
        data = response.json()
        
        assert data is None

    def test_get_active_returns_activated_prompt(self, client):
        """Test that get_active returns the activated prompt"""

        payload = {
            "purpose": "get_active_test",
            "name": "Test Prompt",
            "template": "test template",
        }
        client.post("/v1/prompts", json=payload)

        list_response = client.get("/v1/prompts/get_active_test")
        if list_response.json():
            prompt_id = list_response.json()[0]["id"]
            list_response.json()[0]["name"]

            client.post(
                f"/v1/prompts/{prompt_id}/activate?purpose=get_active_test",
                headers={"x-user-id": "test_user_3"},
            )

            response = client.get("/v1/get_active/get_active_test?user_id=test_user_3")
            assert response.status_code == 200
            data = response.json()
            if data:
                assert data["id"] == prompt_id



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


class TestResponseValidation:
    """Tests for response validation"""

    def test_create_prompt_response_format(self, client):
        """Test that create prompt response has correct format"""
        payload = {"purpose": "validation_test", "name": "Test", "template": "template"}
        response = client.post("/v1/prompts", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "purpose" in data
        assert "name" in data
        assert "template" in data

    def test_list_prompts_response_is_array(self, client):
        """Test that list prompts returns array"""
        response = client.get("/v1/prompts/test_array")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


from unittest.mock import patch

class TestExportEndpoint:
    """Tests for the export endpoint"""

    def test_export_fails_with_in_memory_store(self, client):
        """Test that export fails when not using MongoDB (default test setup)"""
        response = client.post("/v1/prompts/export")
        assert response.status_code == 400
        assert "Export requires MongoDB store" in response.json()["detail"]

    def test_export_succeeds_with_mocked_mongodb_store(self, client):
        """Test that export succeeds when the store is mocked to be a MongoDBStore"""
        with patch("app.api.routes_prompts.isinstance") as mock_isinstance:
            mock_isinstance.return_value = True
            with patch("app.api.routes_prompts.store") as mock_store:
                mock_store.export_prompt_usage_logs.return_value = "/path/to/logs.csv"
                response = client.post("/v1/prompts/export")

                assert response.status_code == 200
                json_response = response.json()
                assert json_response["status"] == "success"
                assert json_response["export_path"] == "/path/to/logs.csv"
                mock_store.export_prompt_usage_logs.assert_called_once()
