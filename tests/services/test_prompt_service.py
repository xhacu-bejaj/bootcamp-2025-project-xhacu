from app.models.domain import Prompt
from app.services.prompt_store import InMemoryStore, FileSnapshotStore
import os
import json
import tempfile
import pytest


def test_prompt_creation():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="Hello, {name}!",
        version=1,
    )
    assert prompt.id == "prompt1"
    assert prompt.purpose == "test purpose"
    assert prompt.name == "Test Prompt"
    assert prompt.template == "Hello, {name}!"
    assert prompt.version == 1


def test_prompt_update_version_increment():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="Hello, {name}!",
        version=1,
    )
    new_template = "Hi, {name}! Welcome to {place}."
    prompt.update(new_template)
    assert prompt.template == new_template
    assert prompt.version == 2


def test_prompt_render_replace_one_variable():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="Hello, {{name}}!",
        version=1,
    )
    rendered = prompt.render({"name": "Alice"})
    assert rendered == "Hello, Alice!"


def test_prompt_render_no_variables():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="Hello, {name}!",
        version=1,
    )
    rendered = prompt.render({"name": "Alice"})
    assert rendered == "Hello, {name}!"


def test_prompt_render_multiple_variables():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="Hello, {{name}}! Welcome to {{place}}.",
        version=1,
    )
    rendered = prompt.render({"name": "Bob", "place": "Wonderland"})
    assert rendered == "Hello, Bob! Welcome to Wonderland."


def test_prompt_render_missing_variable():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="Hello, {{name}}! Welcome to {{place}}.",
        version=1,
    )
    rendered = prompt.render({"name": "Charlie"})
    assert rendered == "Hello, Charlie! Welcome to ."


def test_prompt_render_extra_variable():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="Hello, {{name}}!",
        version=1,
    )
    rendered = prompt.render({"name": "Diana", "place": "Utopia"})
    assert rendered == "Hello, Diana!"


def test_prompt_render_for_loop():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="{% for item in items %}Item: {{item}}\n{% endfor %}",
        version=1,
    )
    rendered = prompt.render({"items": ["apple", "banana", "cherry"]})
    expected_output = "Item: apple\nItem: banana\nItem: cherry\n"
    assert rendered == expected_output


def test_prompt_render_conditional():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="{% if is_member %}Welcome back, member!{% else %}Please sign up.{% endif %}",
        version=1,
    )
    rendered_member = prompt.render({"is_member": True})
    rendered_non_member = prompt.render({"is_member": False})
    assert rendered_member == "Welcome back, member!"
    assert rendered_non_member == "Please sign up."


def test_prompt_render_dictionary_variable():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="User: {{user.name}}, Age: {{user.age}}",
        version=1,
    )
    rendered = prompt.render({"user": {"name": "Eve", "age": 30}})
    assert rendered == "User: Eve, Age: 30"


@pytest.mark.asyncio
async def test_store_create_prompt():
    store = InMemoryStore()
    prompt = await store.create(
        purpose="summarization", name="Summarizer", template="Summarize: {{text}}"
    )

    assert prompt.id is not None
    assert prompt.purpose == "summarization"
    assert prompt.name == "Summarizer"
    assert prompt.template == "Summarize: {{text}}"
    assert prompt.version == 1
    assert prompt.active is False


@pytest.mark.asyncio
async def test_store_list_prompts_by_purpose():
    store = InMemoryStore()

    await store.create("summarization", "Summarizer 1", "Summarize: {{text}}")
    await store.create("summarization", "Summarizer 2", "Quick summary: {{text}}")
    await store.create("translation", "Translator", "Translate to English: {{text}}")

    # List by purpose
    summary_prompts = await store.list("summarization")
    assert len(summary_prompts) == 2
    assert all(p.purpose == "summarization" for p in summary_prompts)

    trans_prompts = await store.list("translation")
    assert len(trans_prompts) == 1
    assert trans_prompts[0].purpose == "translation"


@pytest.mark.asyncio
async def test_store_get_prompt_by_id():
    store = InMemoryStore()

    created = await store.create("translation", "Translator", "Translate: {{data}}")
    retrieved = await store.get(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.name == "Translator"

    assert await store.get("non-existent-id") is None


@pytest.mark.asyncio
async def test_store_patch_prompt():
    store = InMemoryStore()

    prompt = await store.create("qa", "QA Bot", "Question: {{q}}")
    assert prompt.version == 1

    updated = await store.patch(prompt.id, name="QA Assistant", template="Q: {{q}}, A: {{a}}")
    assert updated is not None
    assert updated.name == "QA Assistant"
    assert updated.template == "Q: {{q}}, A: {{a}}"
    assert updated.version == 2

    retrieved = await store.get(prompt.id)
    assert retrieved.template == "Q: {{q}}, A: {{a}}"
    assert retrieved.version == 2


@pytest.mark.asyncio
async def test_store_patch_only_name():
    store = InMemoryStore()

    prompt = await store.create("tagging", "Tagger v1", "Tag: {{text}}")
    updated = await store.patch(prompt.id, name="Tagger v2")

    assert updated is not None
    assert updated.name == "Tagger v2"
    assert updated.template == "Tag: {{text}}"
    assert updated.version == 2


@pytest.mark.asyncio
async def test_store_patch_only_template():
    store = InMemoryStore()

    prompt = await store.create("classification", "Classifier", "Classify: {{text}}")
    updated = await store.patch(prompt.id, template="New classification: {{text}}")

    assert updated is not None
    assert updated.name == "Classifier"
    assert updated.template == "New classification: {{text}}"
    assert updated.version == 2


@pytest.mark.asyncio
async def test_store_set_and_get_active_prompt():
    store = InMemoryStore()

    p1 = await store.create("summarization", "Summary v1", "Summarize v1: {{text}}")
    p2 = await store.create("summarization", "Summary v2", "Summarize v2: {{text}}")

    user_id = "user123"
    purpose = "summarization"

    active = await store.set_active(user_id, purpose, p1.id)
    assert active is not None
    assert active.id == p1.id
    assert active.active is True

    retrieved_active = await store.get_active(user_id, purpose)
    assert retrieved_active is not None
    assert retrieved_active.id == p1.id

    new_active = await store.set_active(user_id, purpose, p2.id)
    assert new_active.id == p2.id
    assert new_active.active is True

    p1_check = await store.get(p1.id)
    assert p1_check.active is False

    final_active = await store.get_active(user_id, purpose)
    assert final_active.id == p2.id

    other_active = await store.get_active("other_user", purpose)
    assert other_active is None


@pytest.mark.asyncio
async def test_store_set_active_nonexistent_prompt():
    store = InMemoryStore()

    result = await store.set_active("user1", "purpose1", "fake-id-123")
    assert result is None


@pytest.mark.asyncio
async def test_store_multiple_users_multiple_purposes():
    store = InMemoryStore()

    p1 = await store.create("summarization", "Summary", "Summarize: {{text}}")
    p2 = await store.create("translation", "Translator", "Translate: {{text}}")

    await store.set_active("user1", "summarization", p1.id)
    await store.set_active("user2", "translation", p2.id)
    await store.set_active("user1", "translation", p2.id)

    assert (await store.get_active("user1", "summarization")).id == p1.id
    assert (await store.get_active("user2", "translation")).id == p2.id
    assert (await store.get_active("user1", "translation")).id == p2.id
    assert await store.get_active("user2", "summarization") is None


@pytest.mark.asyncio
async def test_store_list_empty():
    store = InMemoryStore()

    result = await store.list("nonexistent_purpose")
    assert result == []


@pytest.mark.asyncio
async def test_store_list_all_prompts():
    store = InMemoryStore()

    await store.create("analysis", "Analyzer", "Analyze: {{data}}")
    await store.create("analysis", "Analyzer v2", "Analyze v2: {{data}}")
    await store.create("extraction", "Extractor", "Extract: {{text}}")

    all_analysis = await store.list("analysis")
    assert len(all_analysis) == 2
    assert all(p.purpose == "analysis" for p in all_analysis)


@pytest.mark.asyncio
async def test_store_patch_nonexistent_prompt():
    store = InMemoryStore()

    result = await store.patch("fake-id", name="New Name", template="New Template")
    assert result is None


@pytest.mark.asyncio
async def test_store_get_active_no_active_set():
    store = InMemoryStore()

    await store.create("testing", "Tester", "Test: {{data}}")

    active = await store.get_active("user_x", "testing")
    assert active is None


def test_prompt_render_with_filters():
    prompt = Prompt(
        id="prompt1",
        purpose="filter",
        name="Filter Test",
        template="Text: {{text | upper}}",
        version=1,
    )

    rendered = prompt.render({"text": "hello world"})
    assert rendered == "Text: HELLO WORLD"


# FileSnapshotStore Tests


@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file for testing."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.asyncio
async def test_file_snapshot_store_initialization_creates_empty_file(temp_json_file):
    """Test that FileSnapshotStore initializes with an empty state."""
    store = FileSnapshotStore(temp_json_file)

    assert os.path.exists(temp_json_file)
    assert len(await store.list("any_purpose")) == 0


@pytest.mark.asyncio
async def test_file_snapshot_store_persists_created_prompt(temp_json_file):
    """Test that created prompts are saved to JSON file."""
    store = FileSnapshotStore(temp_json_file)
    prompt = await store.create("summarize", "Summarizer", "Summarize: {{text}}")

    # Verify file was written
    assert os.path.exists(temp_json_file)

    # Verify file content
    with open(temp_json_file, "r") as f:
        data = json.load(f)

    assert len(data["prompts"]) == 1
    assert data["prompts"][0]["id"] == prompt.id
    assert data["prompts"][0]["name"] == "Summarizer"
    assert data["prompts"][0]["template"] == "Summarize: {{text}}"


@pytest.mark.asyncio
async def test_file_snapshot_store_loads_existing_data(temp_json_file):
    """Test that FileSnapshotStore loads existing prompts from file."""
    # Create and save data
    store1 = FileSnapshotStore(temp_json_file)
    p1 = await store1.create("translate", "Translator", "Translate: {{text}}")
    p2 = await store1.create("summarize", "Summarizer", "Summarize: {{text}}")

    # Create new instance and verify it loads the data
    store2 = FileSnapshotStore(temp_json_file)
    prompts = await store2.list("translate")

    assert len(prompts) == 1
    assert prompts[0].id == p1.id
    assert prompts[0].name == "Translator"

    all_summary = await store2.list("summarize")
    assert len(all_summary) == 1
    assert all_summary[0].id == p2.id


@pytest.mark.asyncio
async def test_file_snapshot_store_persists_patch(temp_json_file):
    """Test that patched prompts are saved to file."""
    store1 = FileSnapshotStore(temp_json_file)
    prompt = await store1.create("qa", "QA Bot", "Question: {{q}}")

    # Patch the prompt
    updated = await store1.patch(prompt.id, name="QA Assistant", template="Q: {{q}}, A: {{a}}")
    assert updated is not None
    assert updated.version == 2

    # Load in new instance
    store2 = FileSnapshotStore(temp_json_file)
    retrieved = await store2.get(prompt.id)

    assert retrieved is not None
    assert retrieved.name == "QA Assistant"
    assert retrieved.template == "Q: {{q}}, A: {{a}}"
    assert retrieved.version == 2


@pytest.mark.asyncio
async def test_file_snapshot_store_persists_active_prompts(temp_json_file):
    """Test that active prompts are saved and loaded correctly."""
    store1 = FileSnapshotStore(temp_json_file)
    p1 = await store1.create("summarize", "Summary v1", "Summarize: {{text}}")
    p2 = await store1.create("summarize", "Summary v2", "Summarize v2: {{text}}")

    # Set active
    await store1.set_active("user1", "summarize", p1.id)

    # Load in new instance
    store2 = FileSnapshotStore(temp_json_file)
    active = await store2.get_active("user1", "summarize")

    assert active is not None
    assert active.id == p1.id
    assert active.active is True

    # Verify old prompt is marked inactive
    p2_retrieved = await store2.get(p2.id)
    assert p2_retrieved.active is False


@pytest.mark.asyncio
async def test_file_snapshot_store_updates_active_correctly(temp_json_file):
    """Test that changing active prompt updates file correctly."""
    store1 = FileSnapshotStore(temp_json_file)
    p1 = await store1.create("translate", "Trans v1", "Translate v1: {{text}}")
    p2 = await store1.create("translate", "Trans v2", "Translate v2: {{text}}")

    # Set p1 as active
    await store1.set_active("user1", "translate", p1.id)

    # Change to p2
    await store1.set_active("user1", "translate", p2.id)

    # Load in new instance
    store2 = FileSnapshotStore(temp_json_file)

    # p2 should be active
    active = await store2.get_active("user1", "translate")
    assert active.id == p2.id

    # p1 should not be active
    p1_retrieved = await store2.get(p1.id)
    assert p1_retrieved.active is False


@pytest.mark.asyncio
async def test_file_snapshot_store_multiple_users_and_purposes(temp_json_file):
    """Test multiple users with different purposes persist correctly."""
    store1 = FileSnapshotStore(temp_json_file)
    p1 = await store1.create("summarize", "Summarizer", "Summarize: {{text}}")
    p2 = await store1.create("translate", "Translator", "Translate: {{text}}")

    await store1.set_active("user1", "summarize", p1.id)
    await store1.set_active("user2", "translate", p2.id)
    await store1.set_active("user1", "translate", p2.id)

    # Load in new instance
    store2 = FileSnapshotStore(temp_json_file)

    assert (await store2.get_active("user1", "summarize")).id == p1.id
    assert (await store2.get_active("user2", "translate")).id == p2.id
    assert (await store2.get_active("user1", "translate")).id == p2.id
    assert await store2.get_active("user2", "summarize") is None


@pytest.mark.asyncio
async def test_file_snapshot_store_handles_corrupted_json(temp_json_file):
    """Test that corrupted JSON files don't crash initialization."""
    # Write corrupted JSON
    with open(temp_json_file, "w") as f:
        f.write("{ invalid json content }")

    # Should not crash, should start fresh
    store = FileSnapshotStore(temp_json_file)
    assert len(await store.list("any_purpose")) == 0


@pytest.mark.asyncio
async def test_file_snapshot_store_creates_directory_if_missing():
    """Test that FileSnapshotStore creates parent directories if needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = os.path.join(tmpdir, "nested", "dir", "data.json")

        store = FileSnapshotStore(nested_path)
        await store.create("test", "Test", "Test: {{x}}")

        # Verify directory and file were created
        assert os.path.exists(nested_path)
        assert os.path.isdir(os.path.dirname(nested_path))


@pytest.mark.asyncio
async def test_file_snapshot_store_json_structure(temp_json_file):
    """Test that JSON file has correct structure."""
    store = FileSnapshotStore(temp_json_file)
    p1 = await store.create("summarize", "Summarizer", "Summarize: {{text}}")
    await store.set_active("user1", "summarize", p1.id)

    with open(temp_json_file, "r") as f:
        data = json.load(f)

    # Verify top-level keys
    assert "prompts" in data
    assert "active_prompts" in data

    # Verify prompts structure
    assert isinstance(data["prompts"], list)
    assert len(data["prompts"]) == 1
    prompt_data = data["prompts"][0]
    assert "id" in prompt_data
    assert "purpose" in prompt_data
    assert "name" in prompt_data
    assert "template" in prompt_data
    assert "version" in prompt_data
    assert "active" in prompt_data

    # Verify active_prompts structure
    assert isinstance(data["active_prompts"], list)
    assert len(data["active_prompts"]) == 1
    active_data = data["active_prompts"][0]
    assert "user_id" in active_data
    assert "purpose" in active_data
    assert "prompt_id" in active_data


@pytest.mark.asyncio
async def test_file_snapshot_store_get_by_id(temp_json_file):
    """Test that get() method works correctly after reload."""
    store1 = FileSnapshotStore(temp_json_file)
    created = await store1.create("test", "Test Prompt", "Test: {{data}}")

    # Load in new instance and retrieve by ID
    store2 = FileSnapshotStore(temp_json_file)
    retrieved = await store2.get(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.name == "Test Prompt"
    assert retrieved.template == "Test: {{data}}"


@pytest.mark.asyncio
async def test_file_snapshot_store_list_by_purpose(temp_json_file):
    """Test that list() method filters correctly after reload."""
    store1 = FileSnapshotStore(temp_json_file)
    await store1.create("summarize", "S1", "Summarize 1: {{text}}")
    await store1.create("summarize", "S2", "Summarize 2: {{text}}")
    await store1.create("translate", "T1", "Translate: {{text}}")

    # Load in new instance
    store2 = FileSnapshotStore(temp_json_file)

    summary_prompts = await store2.list("summarize")
    assert len(summary_prompts) == 2
    assert all(p.purpose == "summarize" for p in summary_prompts)

    translate_prompts = await store2.list("translate")
    assert len(translate_prompts) == 1
    assert translate_prompts[0].purpose == "translate"


@pytest.mark.asyncio
async def test_file_snapshot_store_patch_only_name(temp_json_file):
    """Test patching only name persists correctly."""
    store1 = FileSnapshotStore(temp_json_file)
    prompt = await store1.create("test", "Original", "Template: {{x}}")

    updated = await store1.patch(prompt.id, name="Updated Name")
    assert updated is not None

    # Load in new instance
    store2 = FileSnapshotStore(temp_json_file)
    retrieved = await store2.get(prompt.id)

    assert retrieved.name == "Updated Name"
    assert retrieved.template == "Template: {{x}}"
    assert retrieved.version == 2


@pytest.mark.asyncio
async def test_file_snapshot_store_patch_only_template(temp_json_file):
    """Test patching only template persists correctly."""
    store1 = FileSnapshotStore(temp_json_file)
    prompt = await store1.create("test", "Name", "Old: {{x}}")

    updated = await store1.patch(prompt.id, template="New: {{y}}")
    assert updated is not None

    # Load in new instance
    store2 = FileSnapshotStore(temp_json_file)
    retrieved = await store2.get(prompt.id)

    assert retrieved.name == "Name"
    assert retrieved.template == "New: {{y}}"
    assert retrieved.version == 2
