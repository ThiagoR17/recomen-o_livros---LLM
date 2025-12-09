import pandas as pd
import numpy as np
from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()


books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)


raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=4000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, HuggingFaceEmbeddings())


def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 10,
    final_top_k: int = 16,
    
) ->pd.DataFrame:
    
    reco = db_books.similarity_search(query, k=initial_top_k)
    books_list = [
    int(rec.page_content.strip().split()[0].replace('"', ''))
    for rec in reco
]
    books_reco = books[books["isbn13"].isin(books_list)].head(final_top_k)
    
    
    if category != "All":
        books_reco = books_reco[books_reco["simple_categories"] == category].head(final_top_k)
    else:
        books_reco = books_reco.head(final_top_k)

    if tone == "Happy":
        books_reco.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_reco.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_reco.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        books_reco.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_reco.sort_values(by="sadness", ascending=False, inplace=True)
    
    return books_reco    

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        
        
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
   dashboard.launch(theme = gr.themes.Glass())
    