from app.models.domain import Prompt
from app.services.prompt_store import InMemoryStore

def test_prompt_creation():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="Hello, {name}!",
        version=1
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
        version=1
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
        version=1
    )
    rendered = prompt.render({"name": "Alice"})
    assert rendered == "Hello, Alice!"

def test_prompt_render_no_variables():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="Hello, {name}!",
        version=1
    )
    rendered = prompt.render({"name": "Alice"})
    assert rendered == "Hello, {name}!"

def test_prompt_render_multiple_variables():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="Hello, {{name}}! Welcome to {{place}}.",
        version=1
    )
    rendered = prompt.render({"name": "Bob", "place": "Wonderland"})
    assert rendered == "Hello, Bob! Welcome to Wonderland."

def test_prompt_render_missing_variable():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="Hello, {{name}}! Welcome to {{place}}.",
        version=1
    )
    rendered = prompt.render({"name": "Charlie"})
    assert rendered == "Hello, Charlie! Welcome to ."

def test_prompt_render_extra_variable():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="Hello, {{name}}!",
        version=1
    )
    rendered = prompt.render({"name": "Diana", "place": "Utopia"})
    assert rendered == "Hello, Diana!"

def test_prompt_render_for_loop():
    prompt = Prompt(
        id="prompt1",
        purpose="test purpose",
        name="Test Prompt",
        template="{% for item in items %}Item: {{item}}\n{% endfor %}",
        version=1
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
        version=1
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
        version=1
    )
    rendered = prompt.render({"user": {"name": "Eve", "age": 30}})
    assert rendered == "User: Eve, Age: 30"

def test_store_create_prompt():
    store = InMemoryStore()
    prompt = store.create(
        purpose="summarization",
        name="Summarizer",
        template="Summarize: {{text}}"
    )
    
    assert prompt.id is not None
    assert prompt.purpose == "summarization"
    assert prompt.name == "Summarizer"
    assert prompt.template == "Summarize: {{text}}"
    assert prompt.version == 1
    assert prompt.active is False


def test_store_list_prompts_by_purpose():
    store = InMemoryStore()
    
    prompt1 = store.create("summarization", "Summarizer 1", "Summarize: {{text}}")
    prompt2 = store.create("summarization", "Summarizer 2", "Quick summary: {{text}}")
    prompt3 = store.create("translation", "Translator", "Translate to English: {{text}}")
    
    # List by purpose
    summary_prompts = store.list("summarization")
    assert len(summary_prompts) == 2
    assert all(p.purpose == "summarization" for p in summary_prompts)
    
    trans_prompts = store.list("translation")
    assert len(trans_prompts) == 1
    assert trans_prompts[0].purpose == "translation"


def test_store_get_prompt_by_id():
    store = InMemoryStore()
    
    created = store.create("translation", "Translator", "Translate: {{data}}")
    retrieved = store.get(created.id)
    
    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.name == "Translator"
    
    assert store.get("non-existent-id") is None


def test_store_patch_prompt():
    store = InMemoryStore()
    
    prompt = store.create("qa", "QA Bot", "Question: {{q}}")
    assert prompt.version == 1
    
    updated = store.patch(prompt.id, name="QA Assistant", template="Q: {{q}}, A: {{a}}")
    assert updated is not None
    assert updated.name == "QA Assistant"
    assert updated.template == "Q: {{q}}, A: {{a}}"
    assert updated.version == 2
    
    retrieved = store.get(prompt.id)
    assert retrieved.template == "Q: {{q}}, A: {{a}}"
    assert retrieved.version == 2

def test_store_patch_only_name():
    store = InMemoryStore()
    
    prompt = store.create("tagging", "Tagger v1", "Tag: {{text}}")
    updated = store.patch(prompt.id, name="Tagger v2")
    
    assert updated is not None
    assert updated.name == "Tagger v2"
    assert updated.template == "Tag: {{text}}"
    assert updated.version == 2


def test_store_patch_only_template():
    store = InMemoryStore()
    
    prompt = store.create("classification", "Classifier", "Classify: {{text}}")
    updated = store.patch(prompt.id, template="New classification: {{text}}")
    
    assert updated is not None
    assert updated.name == "Classifier"
    assert updated.template == "New classification: {{text}}"
    assert updated.version == 2


def test_store_set_and_get_active_prompt():
    store = InMemoryStore()
    
    p1 = store.create("summarization", "Summary v1", "Summarize v1: {{text}}")
    p2 = store.create("summarization", "Summary v2", "Summarize v2: {{text}}")
    
    user_id = "user123"
    purpose = "summarization"
    
    active = store.set_active(user_id, purpose, p1.id)
    assert active is not None
    assert active.id == p1.id
    assert active.active is True
    
    retrieved_active = store.get_active(user_id, purpose)
    assert retrieved_active is not None
    assert retrieved_active.id == p1.id
    
    new_active = store.set_active(user_id, purpose, p2.id)
    assert new_active.id == p2.id
    assert new_active.active is True
    
    p1_check = store.get(p1.id)
    assert p1_check.active is False
    
    final_active = store.get_active(user_id, purpose)
    assert final_active.id == p2.id
    
    other_active = store.get_active("other_user", purpose)
    assert other_active is None


def test_store_set_active_nonexistent_prompt():
    store = InMemoryStore()
    
    result = store.set_active("user1", "purpose1", "fake-id-123")
    assert result is None


def test_store_multiple_users_multiple_purposes():
    store = InMemoryStore()
    
    p1 = store.create("summarization", "Summary", "Summarize: {{text}}")
    p2 = store.create("translation", "Translator", "Translate: {{text}}")
    
    store.set_active("user1", "summarization", p1.id)
    store.set_active("user2", "translation", p2.id)
    store.set_active("user1", "translation", p2.id)
    
    assert store.get_active("user1", "summarization").id == p1.id
    assert store.get_active("user2", "translation").id == p2.id
    assert store.get_active("user1", "translation").id == p2.id
    assert store.get_active("user2", "summarization") is None


def test_store_list_empty():
    store = InMemoryStore()
    
    result = store.list("nonexistent_purpose")
    assert result == []


def test_store_list_all_prompts():
    store = InMemoryStore()
    
    p1 = store.create("analysis", "Analyzer", "Analyze: {{data}}")
    p2 = store.create("analysis", "Analyzer v2", "Analyze v2: {{data}}")
    p3 = store.create("extraction", "Extractor", "Extract: {{text}}")
    
    all_analysis = store.list("analysis")
    assert len(all_analysis) == 2
    assert all(p.purpose == "analysis" for p in all_analysis)


def test_store_patch_nonexistent_prompt():
    store = InMemoryStore()
    
    result = store.patch("fake-id", name="New Name", template="New Template")
    assert result is None


def test_store_get_active_no_active_set():
    store = InMemoryStore()
    
    p1 = store.create("testing", "Tester", "Test: {{data}}")
    
    active = store.get_active("user_x", "testing")
    assert active is None


def test_prompt_render_with_filters():
    prompt = Prompt(
        id="prompt1",
        purpose="filter",
        name="Filter Test",
        template="Text: {{text | upper}}",
        version=1
    )
    
    rendered = prompt.render({"text": "hello world"})
    assert rendered == "Text: HELLO WORLD"