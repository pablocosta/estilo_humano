
PROMPT_IN_V2 = """A tarefa de reescrita textual ao estilo autoral tem como objetivo reescrever textos ao estilo-alvo mantendo seu significado original, mas com palavras diferentes. Certifique-se de preservar as características semânticas originais do texto, garantindo clareza e coerência na reescrita. O texto original a seguir está em português, e o texto reescrito também deve estar em português. Você deve apresentar uma única alternativa de reescrita e ao final do raciocionio colocar a resposta entre aspas.

Texto original:
{input_text}
"""

PROMPT_IN_V1_EN = """The task of textual rewriting in the authorial style aims to rewrite texts in the target style while maintaining their original meaning, but with different words. Make sure to preserve the original semantic characteristics of the text, ensuring clarity and coherence when rewriting. The original text below is in Portuguese, and the rewritten text must also be in Portuguese. You must present a single rewrite alternative. Find the solution.

Target author:
{target_author}

Original text:
{input_text}
"""

PROMPT_IN_V1 = """A tarefa de reescrita textual ao estilo autoral tem como objetivo reescrever textos ao estilo-alvo mantendo seu significado original, mas com palavras diferentes. Certifique-se de preservar as características semânticas originais do texto, garantindo clareza e coerência na reescrita. O texto original a seguir está em português, e o texto reescrito também deve estar em português. Você deve apresentar uma única alternativa de reescrita.

Autor-alvo:
{autor_alvo}

Texto original:
{input_text}

"""

PROMPT_OUT = """Texto reescrito:
{rewritten_text}"""
