import requests
import xmltodict
from multiprocessing import Pool
from sentence_transformers import SentenceTransformer
import torch


class ArxivPaper:
    def __init__(self, title, abstract, link, category, published):
        self.title = title
        self.abstract = abstract
        self.link = link
        self.category = category
        self.published = published

    def get_embedding(self, model):
        print("generating embedding")
        self.embedding = model.encode(self.abstract)
        print("embedding generated")

    def embedding_to_pinecone(self):
        pass


class ArxivScraper:
    def __init__(self):
        pass

    def get_papers(self, total, chunk_size):
        url = "http://export.arxiv.org/api/query"

        papers = []
        start_range = list(range(0, total, chunk_size))
        while len(start_range) > 0:
            start = start_range.pop(0)
            print(f">>> {start} of {total}", end="\r")
            params = {
                "search_query": "all",
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": start,
                "max_results": chunk_size
            }
            response = requests.get(url, params=params)
            response = xmltodict.parse(response.text)["feed"]

            try:
                response = response["entry"]
            except KeyError:
                print("KeyError >:(")
                start_range.insert(0, start)
                continue

            for paper in response:
                if type(paper["category"]) == list:
                    category_name = paper["category"][0]["@term"]
                else:
                    category_name = paper["category"]["@term"]

                papers.append(ArxivPaper(paper["title"], paper["summary"],
                                         paper["id"], category_name,
                                         paper["published"]))

        return papers

def process_paper(paper, model):
    paper.get_embedding(model)
    return paper

def main():
    # Scrape arxiv
    scraper = ArxivScraper()
    papers = scraper.get_papers(total=100, chunk_size=10)

    print(torch.multiprocessing.get_all_sharing_strategies())

    model_1 = SentenceTransformer('all-mpnet-base-v2', device="cuda")
    model_2 = SentenceTransformer('all-mpnet-base-v2', device="cuda")
    
    models = [model_1, model_2]

    torch.multiprocessing.spawn(process_paper, args=(zip(papers, models)), nprocs=2, join=True)

if __name__ == "__main__": main()
