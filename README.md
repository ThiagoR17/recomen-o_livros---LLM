## Este repositório contém todo o código para completar o curso gratuito da freeCodeCamp, "Build a Semantic Book Recommender with LLMs – Full Course". O tutorial é dividido em **cinco componentes**:

* **Limpeza de dados de texto** (código no *notebook* `data-exploration.ipynb`)
* **Busca Semântica (vetorial)** e como construir um banco de dados vetorial (código no *notebook* `vector-search.ipynb`). Isso permite aos usuários encontrar os livros mais similares a uma consulta em linguagem natural (ex.: "um livro sobre uma pessoa buscando vingança").
* **Classificação de texto usando classificação *zero-shot* em LLMs** (código no *notebook* `text-classification.ipynb`). Isso nos permite classificar os livros como "ficção" ou "não-ficção", criando um filtro (*facet*) que os usuários podem aplicar.
* **Análise de sentimento usando LLMs e extração das emoções do texto** (código no *notebook* `sentiment-analysis.ipynb`). Isso permitirá aos usuários ordenar os livros pelo seu tom, como quão suspensivos, alegres ou tristes eles são.
* **Criação de uma aplicação web usando Gradio** para que os usuários obtenham recomendações de livros (código no arquivo `gradio-dashboard.py`).

---

Este projeto foi inicialmente criado em **Python 3.13**. Para rodar o projeto, as seguintes **dependências são requeridas**:

* `kagglehub`
* `pandas`
* `matplotlib`
* `seaborn`
* `python-dotenv`
* `langchain-community`
* `langchain-opencv`
* `langchain-chroma`
* `transformers`
* `gradio`
* `notebook`
* `ipywidgets`

Para criar seu banco de dados vetorial, você precisará criar um arquivo **`.env`** no seu diretório raiz contendo sua **chave API da OpenAI ou outro modelo de IA**. As instruções sobre como fazer isso fazem parte do tutorial.

Os dados para este projeto podem ser baixados do **Kaggle**. As instruções sobre como fazer isso também estão no repositório.
