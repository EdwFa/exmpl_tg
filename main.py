import streamlit as st
import yaml
import json
import os
from openai import OpenAI
import tiktoken

# === ПУТИ И КОНФИГУРАЦИЯ ===

TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY") or "your-together-api-key-here"
PROMPTS_DIR = "prompts"

CHAT_MODELS = {
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "Qwen3 Next 80B",
    "meta-llama/Llama-3-70b-chat-hf": "Llama 3 70B Chat",
    "meta-llama/Llama-3-8b-chat-hf": "Llama 3 8B Chat",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B Instruct",
    "google/gemma-2-9b-it": "Gemma 2 9B Instruct",
}

MODEL_CONTEXT_LIMITS = {
    "Qwen/Qwen3-Next-80B-A3B-Instruct": 32768,
    "meta-llama/Llama-3-70b-chat-hf": 8192,
    "meta-llama/Llama-3-8b-chat-hf": 8192,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
    "google/gemma-2-9b-it": 8192,
}

DEFAULT_PROMPT_NAME = "default"

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        return len(text) // 4

def trim_messages(messages, system_prompt, model_id, max_gen_tokens=1024):
    context_limit = MODEL_CONTEXT_LIMITS.get(model_id, 4096)
    available_tokens = context_limit - max_gen_tokens - 200
    total_tokens = count_tokens(system_prompt, model_id)
    trimmed = [{"role": "system", "content": system_prompt}]
    for msg in reversed(messages[1:]):
        msg_tokens = count_tokens(msg["content"], model_id)
        if total_tokens + msg_tokens > available_tokens:
            break
        trimmed.insert(1, msg)
        total_tokens += msg_tokens
    if len(trimmed) == 1 and len(messages) > 1:
        trimmed.append(messages[-1])
    return trimmed

# === ЗАГРУЗКА ПРОМПТОВ ИЗ ФАЙЛОВ ===

def load_prompt_from_path(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f)
            elif filepath.endswith(".json"):
                data = json.load(f)
            else:
                return None
        if isinstance(data, dict) and "system_prompt" in data:
            return data["system_prompt"].strip()
    except Exception as e:
        st.error(f"Ошибка загрузки {filepath}: {e}")
    return None

def get_available_prompt_profiles():
    profiles = []
    if os.path.exists(PROMPTS_DIR):
        for f in sorted(os.listdir(PROMPTS_DIR)):
            if f.endswith((".yaml", ".yml", ".json")):
                name = os.path.splitext(f)[0]
                profiles.append(name)
    return profiles if profiles else [DEFAULT_PROMPT_NAME]

# === ИНТЕРФЕЙС ===

st.set_page_config(page_title="💬 Чат с профилями промптов", layout="wide")
st.title("💬 ИИ-ассистент с профилями (Together.ai)")

# Инициализация состояния
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_system_prompt" not in st.session_state:
    st.session_state.current_system_prompt = ""
if "model_id" not in st.session_state:
    st.session_state.model_id = list(CHAT_MODELS.keys())[0]
if "selected_profile" not in st.session_state:
    st.session_state.selected_profile = DEFAULT_PROMPT_NAME
if "uploaded_prompt" not in st.session_state:
    st.session_state.uploaded_prompt = None

# Получаем список профилей
available_profiles = get_available_prompt_profiles()

# Боковая панель
with st.sidebar:
    st.header("⚙️ Промпты и модели")

    # Выбор профиля
    selected = st.selectbox(
        "📁 Выберите профиль промпта",
        options=available_profiles,
        index=available_profiles.index(st.session_state.selected_profile)
        if st.session_state.selected_profile in available_profiles
        else 0
    )
    st.session_state.selected_profile = selected

    # Загрузка файла (переопределяет профиль)
    prompt_file = st.file_uploader("📤 Или загрузите свой файл (YAML/JSON)", type=["yaml", "yml", "json"])
    if prompt_file:
        st.session_state.uploaded_prompt = prompt_file
    else:
        st.session_state.uploaded_prompt = None

    # Определяем актуальный системный промпт
    if st.session_state.uploaded_prompt:
        # Загруженный файл имеет приоритет
        file = st.session_state.uploaded_prompt
        try:
            content = file.read().decode("utf-8")
            if file.name.endswith((".yaml", ".yml")):
                data = yaml.safe_load(content)
            else:
                data = json.loads(content)
            system_prompt = data.get("system_prompt", "").strip() if isinstance(data, dict) else ""
        except:
            system_prompt = ""
        if not system_prompt:
            st.warning("Файл не содержит system_prompt. Используется профиль.")
            st.session_state.uploaded_prompt = None
    else:
        # Используем профиль
        filepath = os.path.join(PROMPTS_DIR, f"{st.session_state.selected_profile}.yaml")
        if not os.path.exists(filepath):
            filepath = os.path.join(PROMPTS_DIR, f"{st.session_state.selected_profile}.json")
        system_prompt = load_prompt_from_path(filepath) or ""

    # Сохраняем текущий промпт
    if system_prompt != st.session_state.current_system_prompt:
        st.session_state.current_system_prompt = system_prompt
        # Обновляем системное сообщение
        if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
            st.session_state.messages[0]["content"] = system_prompt
        else:
            st.session_state.messages.insert(0, {"role": "system", "content": system_prompt})

    st.text_area("Текущий системный промпт", value=st.session_state.current_system_prompt, height=150, disabled=True)

    # Настройки модели
    api_key = st.text_input("🔑 API-ключ Together.ai", type="password", value=TOGETHER_API_KEY)
    st.session_state.api_key = api_key

    model_id = st.selectbox(
        "🧠 Модель",
        options=list(CHAT_MODELS.keys()),
        format_func=lambda x: CHAT_MODELS[x],
        index=0
    )
    st.session_state.model_id = model_id

    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.3, 0.05)
    max_tokens = st.slider("📏 Макс. токенов в ответе", 128, 4096, 1024, 128)
    top_p = st.slider("🎯 Top-p", 0.1, 1.0, 0.9, 0.05)

    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = [{"role": "system", "content": st.session_state.current_system_prompt}]
        st.rerun()

# Проверка ключа
if not st.session_state.api_key:
    st.warning("Введите API-ключ Together.ai.")
    st.stop()

if not st.session_state.current_system_prompt:
    st.warning("Не удалось загрузить системный промпт. Проверьте файлы в папке prompts/.")

# Отображение истории (без системного сообщения)
for msg in st.session_state.messages[1:]:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# Ввод пользователя
if prompt := st.chat_input("Ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_to_send = trim_messages(
        st.session_state.messages,
        st.session_state.current_system_prompt,
        st.session_state.model_id,
        max_tokens
    )

    with st.chat_message("assistant"):
        with st.spinner("Генерация ответа..."):
            try:
                client = OpenAI(
                    api_key=st.session_state.api_key,
                    base_url="https://api.together.xyz/v1"
                )
                response = client.chat.completions.create(
                    model=st.session_state.model_id,
                    messages=messages_to_send,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False
                )
                reply = response.choices[0].message.content.strip()
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Ошибка: {e}")