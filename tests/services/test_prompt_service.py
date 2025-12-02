from app.models.domain import Prompt

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