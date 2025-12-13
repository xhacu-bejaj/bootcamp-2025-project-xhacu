"""
End-to-end tests for complete document processing workflows.

These tests verify that the entire system works together correctly,
from API endpoints through business logic to storage.
"""
import pytest


@pytest.mark.e2e
class TestDocumentProcessingWorkflow:
    """Test complete document processing workflows."""

    def test_complete_translation_workflow(self, e2e_client, sample_document_text, sample_prompts):
        """Test complete workflow: create prompt → activate → translate → verify."""
        # Step 1: Create a translation prompt
        create_response = e2e_client.post(
            "/v1/prompts",
            json={
                "purpose": "translate",
                "name": sample_prompts["translate"]["name"],
                "template": sample_prompts["translate"]["template"]
            }
        )
        assert create_response.status_code == 201
        prompt_data = create_response.json()
        prompt_id = prompt_data["id"]
        assert prompt_data["purpose"] == "translate"
        assert prompt_data["active"] is False
        
        # Step 2: Activate the prompt
        activate_response = e2e_client.post(
            f"/v1/prompts/{prompt_id}/activate",
            headers={"X-User-Id": "e2e_test_user"}
        )
        assert activate_response.status_code == 200
        activated = activate_response.json()
        assert activated["active"] is True
        
        # Step 3: Process a document
        predict_response = e2e_client.post(
            "/v1/predict",
            json={
                "purpose": "translate",
                "document_text": "Bonjour le monde",
                "provider": "mock"
            },
            headers={"X-User-Id": "e2e_test_user"}
        )
        assert predict_response.status_code == 200
        result = predict_response.json()
        assert "output_text" in result
        assert result["prompt_id"] == prompt_id
        assert result["model_info"]["model"] == "mock"
        assert result["latency_ms"] >= 0
        
        # Step 4: Verify the active prompt can be retrieved
        active_response = e2e_client.get(
            "/v1/prompts/active",
            params={"purpose": "translate"},
            headers={"X-User-Id": "e2e_test_user"}
        )
        assert active_response.status_code == 200
        active_prompt = active_response.json()
        assert active_prompt["id"] == prompt_id

    def test_summarization_workflow_with_multiple_prompts(self, e2e_client, sample_document_text, sample_prompts):
        """Test creating multiple prompts and switching between them."""
        # Create two different summarization prompts
        prompt1 = e2e_client.post(
            "/v1/prompts",
            json={
                "purpose": "summarize",
                "name": "Brief Summarizer",
                "template": "Provide a brief summary:\n\n{{ document_text }}"
            }
        ).json()
        
        prompt2 = e2e_client.post(
            "/v1/prompts",
            json={
                "purpose": "summarize",
                "name": "Detailed Summarizer",
                "template": "Provide a detailed summary:\n\n{{ document_text }}"
            }
        ).json()
        
        # Activate first prompt
        e2e_client.post(
            f"/v1/prompts/{prompt1['id']}/activate",
            headers={"X-User-Id": "user1"}
        )
        
        # Process document with first prompt
        result1 = e2e_client.post(
            "/v1/predict",
            json={
                "purpose": "summarize",
                "document_text": sample_document_text,
                "provider": "mock"
            },
            headers={"X-User-Id": "user1"}
        ).json()
        assert result1["prompt_id"] == prompt1["id"]
        
        # Switch to second prompt
        e2e_client.post(
            f"/v1/prompts/{prompt2['id']}/activate",
            headers={"X-User-Id": "user1"}
        )
        
        # Process document with second prompt
        result2 = e2e_client.post(
            "/v1/predict",
            json={
                "purpose": "summarize",
                "document_text": sample_document_text,
                "provider": "mock"
            },
            headers={"X-User-Id": "user1"}
        ).json()
        assert result2["prompt_id"] == prompt2["id"]
        assert result2["prompt_id"] != result1["prompt_id"]

    def test_multi_user_workflow(self, e2e_client):
        """Test that different users can have different active prompts."""
        # Create a prompt
        prompt_response = e2e_client.post(
            "/v1/prompts",
            json={
                "purpose": "extract",
                "name": "Entity Extractor",
                "template": "Extract entities:\n\n{{ document_text }}"
            }
        )
        prompt_id = prompt_response.json()["id"]
        
        # User 1 activates the prompt
        e2e_client.post(
            f"/v1/prompts/{prompt_id}/activate",
            headers={"X-User-Id": "user1"}
        )
        
        # User 2 activates the prompt
        e2e_client.post(
            f"/v1/prompts/{prompt_id}/activate",
            headers={"X-User-Id": "user2"}
        )
        
        # Both users can use the prompt
        for user_id in ["user1", "user2"]:
            result = e2e_client.post(
                "/v1/predict",
                json={
                    "purpose": "extract",
                    "document_text": "John works at Microsoft in Seattle.",
                    "provider": "mock"
                },
                headers={"X-User-Id": user_id}
            )
            assert result.status_code == 200
            assert result.json()["prompt_id"] == prompt_id

    def test_prompt_update_workflow(self, e2e_client):
        """Test updating a prompt and using the new version."""
        # Create prompt
        create_response = e2e_client.post(
            "/v1/prompts",
            json={
                "purpose": "classify",
                "name": "Classifier V1",
                "template": "Classify: {{ document_text }}"
            }
        )
        prompt = create_response.json()
        prompt_id = prompt["id"]
        assert prompt["version"] == 1
        
        # Update the prompt
        update_response = e2e_client.patch(
            f"/v1/prompts/{prompt_id}",
            json={
                "name": "Classifier V2",
                "template": "Classify this text: {{ document_text }}"
            }
        )
        updated = update_response.json()
        assert updated["version"] == 2
        assert updated["name"] == "Classifier V2"
        assert updated["id"] == prompt_id  # Same ID
        
        # Activate and use updated prompt
        e2e_client.post(f"/v1/prompts/{prompt_id}/activate")
        result = e2e_client.post(
            "/v1/predict",
            json={
                "purpose": "classify",
                "document_text": "This is a test",
                "provider": "mock"
            }
        )
        assert result.status_code == 200
        assert result.json()["prompt_version"] == 2

    def test_error_handling_no_active_prompt(self, e2e_client):
        """Test error when trying to process without an active prompt."""
        response = e2e_client.post(
            "/v1/predict",
            json={
                "purpose": "nonexistent_purpose",
                "document_text": "Test document",
                "provider": "mock"
            }
        )
        # Should return error (404 or 400)
        assert response.status_code in [400, 404]
        error = response.json()
        assert "detail" in error

    def test_prompt_listing_workflow(self, e2e_client):
        """Test listing prompts by purpose."""
        # Create multiple prompts with same purpose
        purposes = ["analyze", "analyze", "compare"]
        for i, purpose in enumerate(purposes):
            e2e_client.post(
                "/v1/prompts",
                json={
                    "purpose": purpose,
                    "name": f"Prompt {i}",
                    "template": f"Template {i}: {{{{ document_text }}}}"
                }
            )
        
        # List prompts for 'analyze'
        response = e2e_client.get("/v1/prompts", params={"purpose": "analyze"})
        assert response.status_code == 200
        prompts = response.json()
        analyze_prompts = [p for p in prompts if p["purpose"] == "analyze"]
        assert len(analyze_prompts) >= 2
        
        # List prompts for 'compare'
        response = e2e_client.get("/v1/prompts", params={"purpose": "compare"})
        assert response.status_code == 200
        prompts = response.json()
        compare_prompts = [p for p in prompts if p["purpose"] == "compare"]
        assert len(compare_prompts) >= 1


@pytest.mark.e2e
class TestChunkWorkflow:
    """Test complete chunk storage and retrieval workflows."""

    @pytest.mark.asyncio
    async def test_document_chunking_and_retrieval_workflow(self, e2e_client, clean_chunk_store):
        """Test inserting chunks and retrieving them."""
        # Insert multiple chunks
        chunks_data = [
            {"text": "First chunk about Python programming", "metadata": {"source": "doc1", "idx": 0}},
            {"text": "Second chunk about machine learning", "metadata": {"source": "doc1", "idx": 1}},
            {"text": "Third chunk about data science", "metadata": {"source": "doc2", "idx": 0}},
        ]
        
        inserted_ids = []
        for chunk_data in chunks_data:
            response = e2e_client.post("/v1/chunks/insert", json=chunk_data)
            assert response.status_code == 201
            inserted_ids.append(response.json()["id"])
        
        # Retrieve similar chunks
        query_response = e2e_client.post(
            "/v1/chunks/query",
            json={"query_text": "Python programming", "n_chunks": 2}
        )
        assert query_response.status_code == 200
        results = query_response.json()
        assert len(results) <= 2
        assert any("Python" in r["text"] for r in results)

    @pytest.mark.asyncio
    async def test_chunk_count_workflow(self, e2e_client, clean_chunk_store):
        """Test counting chunks after insertion."""
        # Insert chunks
        for i in range(5):
            response = e2e_client.post(
                "/v1/chunks/insert",
                json={"text": f"Test chunk number {i}"}
            )
            assert response.status_code == 201
        
        # Get count
        count_response = e2e_client.get("/v1/chunks/count")
        assert count_response.status_code == 200
        count = count_response.json()
        assert count >= 5


@pytest.mark.e2e
@pytest.mark.slow
class TestFullSystemWorkflow:
    """Test complete system workflows involving multiple components."""

    def test_complete_rag_workflow(self, e2e_client, sample_document_text, sample_prompts):
        """Test a complete RAG-like workflow: chunk → store → retrieve → process."""
        # Step 1: Create and activate a prompt
        prompt_response = e2e_client.post(
            "/v1/prompts",
            json={
                "purpose": "qa",
                "name": "QA System",
                "template": "Answer based on context:\n\n{{ document_text }}"
            }
        )
        prompt_id = prompt_response.json()["id"]
        e2e_client.post(f"/v1/prompts/{prompt_id}/activate")
        
        # Step 2: Insert document chunks (simulating RAG storage)
        chunks = [
            "AI is transforming technology.",
            "Machine learning enables data-driven decisions.",
            "Deep learning uses neural networks."
        ]
        
        for chunk_text in chunks:
            chunk_response = e2e_client.post(
                "/v1/chunks/insert",
                json={"text": chunk_text}
            )
            assert chunk_response.status_code == 201
        
        # Step 3: Query for relevant chunks
        query_response = e2e_client.post(
            "/v1/chunks/query",
            json={"query_text": "What is machine learning?", "n_chunks": 2}
        )
        assert query_response.status_code == 200
        relevant_chunks = query_response.json()
        assert len(relevant_chunks) > 0
        
        # Step 4: Process with LLM (using retrieved context)
        context = " ".join([c["text"] for c in relevant_chunks])
        predict_response = e2e_client.post(
            "/v1/predict",
            json={
                "purpose": "qa",
                "document_text": context,
                "provider": "mock"
            }
        )
        assert predict_response.status_code == 200
        result = predict_response.json()
        assert "output_text" in result
        assert result["prompt_id"] == prompt_id
