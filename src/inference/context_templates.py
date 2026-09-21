from dataclasses import dataclass
from textwrap import dedent

@dataclass
class BaseConfig:
    system_prompt: str
    prompt: str
    retrieved_data: str = ""

    @classmethod
    def generate_context(cls, text: str, context: str) -> list[dict[str, str]]:
        # System prompt
        system_prompt = cls.system_prompt

        # User prompt
        prompt = cls.retrieved_data.format(context=context) + cls.prompt.format(text=text)

        # Structured messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return messages

    @classmethod
    def generate_base_context(cls, text: str) -> list[dict[str, str]]:
        # System prompt
        system_prompt = cls.system_prompt

        # User prompt
        prompt = cls.prompt.format(text=text)

        # Structured messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return messages


@dataclass
class ContextConfig(BaseConfig):
    system_prompt: str = dedent("""\
        Eres un asistente especializado en la Universidad Nacional de Educación a Distancia (UNED). 
        Se te proporcionará una Pregunta y un Contexto (documentación relevante) que puede ayudarte a contestar a la pregunta. 

        Por favor, sigue estas reglas al formular tu respuesta:
        1. Si la pregunta contiene una premisa falsa o incorrecta, responde "Pregunta inválida" y explica por qué.
        2. Si no puedes responder con certeza, responde "No sé la respuesta". En este caso, proporciona un correo electrónico de contacto para que el usuario pueda obtener más información.
    """)
    prompt: str = dedent("""\
        Tu tarea es contestar a la pregunta formulada a continuación en una sola frase.

        ### Pregunta:
        {text}

        ### Respuesta:
    """)
    retrieved_data: str = dedent("""\
        ### Contexto proporcionado: 
        {context}
    """)

@dataclass
class BaseContextConfig(BaseConfig):
    system_prompt: str = dedent("""\
        Eres un asistente especializado en la Universidad Nacional de Educación a Distancia (UNED).
        Se te proporcionará una Pregunta que debes responder. 

        Por favor, sigue estas reglas al formular tu respuesta:
        1. Si la pregunta contiene una premisa falsa o incorrecta, responde "Pregunta inválida" y explica por qué..
        2. Si no puedes responder con certeza, responde "No sé la respuesta". En este caso, proporciona un correo electrónico de contacto para que el usuario pueda obtener más información.
    """)
    prompt: str = dedent("""\
        Tu tarea es contestar a la pregunta formulada a continuación en una sola frase.

        ### Pregunta:
        {text}

        ### Respuesta:
    """)