# Idea: chat with an export of WOS

# read the csv

# generate a paper instance for each row

# generate an embedding for each paper

# upload the embeddings to chroma

# generate a similary measure between the search and query embedding

# optional: add a penalty for papers that are dissimilar to a group of papers
# that are similar. To do this, select a few papers a priori that are wanted
# and some that are unwanted. Train a model to predict whether a paper is
# wanted or unwanted. Classify all papers. If the paper is unwanted, add a
# penalty.

# Return the top N papers that are most similar to the search query

# Then do something else on the papers like a summary?

import selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromiumService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.utils import ChromeType
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from util import paper_pal_template

import os
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import PyPDF2
import re
import pickle
import nltk
import annoy
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

from util import wos_category_names

class WebOfSciencePaper:
    """
    Represent an academic article with title, abstract and doi. Provices methods
    to download and scrape the pdf.
    """
    def __init__(self, title, abstract, doi):
        self.title = title
        self.abstract = abstract
        self.process_abstract()
        self.doi = doi
        self.abstract_embedding = None
        self.text_embedding = None
        self.index = None # This index relates to the title
        # Each (nested) sentence has an index and we can use this to find the
        # title of the paper after querying the vector store
        self.abstract_indexes = []

    def get_abstract_embedding(self, model):
        """Get the embedding of the abstract."""
        self.abstract_embedding = model.encode(self.abstract).tolist()

    def get_text_embedding(self, model):
        """Gets one embedding for every sentence in the text."""
        self.text_embedding = []
        for sentence in self.text:
            self.text_embedding.append(model.encode(sentence).tolist())

    def _init_driver(self):
        # I have to reconstruct the driver for every paper, because of some
        # obscure and haunted bug. :(
        options = webdriver.ChromeOptions()
        profile = {"plugins.plugins_list": [{"enabled": False,
                                             "name": "Chrome PDF Viewer"}],
                   "download.default_directory": "/home/sebastiaan/fun/GlossPT",
                   "download.extensions_to_open": "",
                   "plugins.always_open_pdf_externally": True}

        options.add_experimental_option("prefs", profile)
        options.add_argument('--headless')
        options.add_argument("--remote-debugging-port=9222") # do not delete
        options.add_argument("--window-size=800,800")
        options.add_argument('--disable-blink-features=AutomationControlled')
        driver = webdriver.Chrome(
            options=options,
            service=ChromiumService(
                ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
            )
        )
        return driver

    def _get_soup(self, driver):
        """Get html from sci-hub. Enter the search term and click the search
           button. Return the html."""
        base_url = "https://sci-hub.ru/"
        driver.get(base_url)
        # Wait until button becomes visible
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        btn = WebDriverWait(driver, 10)\
                .until(EC.presence_of_element_located((By.ID, "request")))
        # Search for the paper
        search_target = self.doi if self.doi!="" else self.title
        search_inputs = driver.find_elements(By.ID, "request")

        # I have since found out that this is for a DDOS protection
        # In this case, the pdf might be available, but we cannot access it
        # So the title should not be added to unavailable_papers.txt

        if len(search_inputs) == 0:
            return None

        search_inputs[0].send_keys(search_target)
        # Find and click the search button
        buttons = driver.find_elements(By.TAG_NAME, "button")
        buttons[0].click()
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        return soup

    def _scrape_pdf(self):
        """Scrape the pdf from sci-hub using the DOI or title if it is not
           available."""
        # Construct driver, enter search term, go to next page, return html
        base_url = "https://sci-hub.ru/"
        driver = self._init_driver()
        soup = self._get_soup(driver)

        # There is no search bar on the page
        # This callback ensures that the driver can be closed
        # if soup == 1:
        #     driver.quit()
        #     self.pdf_available = False
        #     raise PaperNotFoundException

        # If there is no button, sci-hub does not have this paper and did not
        # return a page with a pdf
        result = soup.find_all('button')
        if len(result) == 0:
            driver.quit()
            # self.pdf_available = False 
            raise PaperNotFoundException

        # If there is a button, sci-hub has the paper and we download the pdf
        if "onclick" in result[0].attrs.keys():
            link = result[0]["onclick"]
            link = link.replace("location.href='", "").replace("'", "")
            if "sci-hub" not in link:
                base_url = base_url[:-1]
                link = base_url + link
            else:
                link = link.replace("//", "")
                link = "https://" + link
            requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
            response = requests.get(link, verify=False) # Yes, I know
            driver.quit()
            if response.status_code == 200:
                with open(f"pdf_files/{self.title}.pdf", "wb") as file:
                    file.write(response.content)
                    driver.quit()
                    # self.pdf_available = True
                    return
            # This is in the case of a 404 error, even if we found the link
            else:
                driver.quit()
                # self.pdf_available = False
                raise PaperNotFoundException

        driver.quit()
        # self.pdf_available = False
        raise PaperNotFoundException

    def get_pdf(self):
        """Try to find the pdf in the pdf_files folder. If it is not there,
           scrape sci-hub and raise an exception if it is not available."""
        # If the pdf has already been downloaded, get that
        pdf_files = os.listdir("pdf_files")
        pdf_files = [i.replace(".pdf", "") for i in pdf_files]
        if self.title in pdf_files:
            # self.pdf_available = True
            pass
        # If not, scrape sci-hub
        else:
            self._scrape_pdf()

    def process_pdf(self):
        """Read the pdf and extract the text."""
        # reader = PyPDF2.PdfReader(BytesIO(self.pdf))
        reader = PyPDF2.PdfReader(open(f"pdf_files/{self.title}.pdf", "rb"))
        pdf_text = [reader.pages[i].extract_text() for i in range(len(reader.pages))]
        pdf_text = "".join(pdf_text)
        pdf_text = re.sub("\s+", " ", pdf_text)
        pdf_text = pdf_text.replace("\n", "") \
                           .replace("\t", "") \
                           .replace("- ", "") \
                           .replace("  ", " ") \
                           .encode("ascii", "ignore") \
                           .decode("ascii")

        # If the text is empty, raise a PaperNotFoundException
        if len(pdf_text) < 100:
            # self.pdf_available = False
            raise PaperNotFoundException

        # Delete the references section
        if "references" in pdf_text.lower():
            references_index = pdf_text.lower().rfind("references")
            pdf_text = pdf_text[:references_index]

        sentences = nltk.tokenize.sent_tokenize(pdf_text)

        # Merge short sentences with the previous sentence
        for idx in range(1, len(sentences)):
            if len(sentences[idx]) < 20:
                sentences[idx-1] += sentences[idx]
                sentences[idx] = ""
        sentences = [i for i in sentences if i != ""]
        sentences = [i for i in sentences if "keywords" not in i.lower()]
        sentences = [i for i in sentences if "arxiv" not in i.lower()]
        # Remove all sentences with a link in them
        sentences = [i for i in sentences if not bool(re.search(r'https?://\S+|www\.\S+', i))]
        # Delete short sentences
        sentences = [i for i in sentences if len(i) > 5]
        self.text = sentences

    def process_abstract(self):
        self.abstract = nltk.tokenize.sent_tokenize(self.abstract)
        # Make self.abstract into an overlapping nested list
        self.abstract = [self.abstract[i:i+3] for i in range(len(self.abstract)-2)]


class PaperNotFoundException(Exception):
    def __init__(self):
        pass

class DDOSCheckException(Exception):
    def __init__(self):
        pass

class WebOfScienceScraper:
    """
    Scrapes a list of tab delimited files that has been exported from Web of
    Science.
    """
    def __init__(self):
        self.data = None # [[title, abstract, doi]]

    def read_tdf(self, file_paths: List[str], top_n: int):
        # TODO: implement path as last of multiple files
        data = []
        for path in file_paths:
            with open(path, "r") as file:
                data.extend(file.readlines())

        data = data[:top_n]
        data = [i.replace("\ufeff", "").replace("\n", "") for i in data]
        data = [i.split("\t") for i in data]

        # Remove all columns except title, abstract, and doi
        idx_to_keep = []
        for k in wos_category_names.keys():
            try:
                idx = data[0].index(k)
                idx_to_keep.append(idx)
            except ValueError:
                continue
        data = [[i[j] for j in idx_to_keep] for i in data]

        # Delete obs with no abstract
        data = [i for i in data if len(i[1]) > 0]

        # Remove ' and / from the titles
        # This is necessary for an error with duckdb and chroma
        for i in range(len(data)):
            data[i][0] = data[i][0].replace("'", "").replace("/", "")

        # Titles can't be longer than 255 characters
        for i in range(len(data)):
            if len(data[i][0]) > 200:
                data[i][0] = data[i][0][:200]

        # Make a dict out of each row of data
        data[0] = [wos_category_names[i] for i in data[0]]
        data = [dict(zip(data[0], i)) for i in data[1:]]

        self.data = data[1:]

    def _delete_duplicates(self, papers: List[WebOfSciencePaper]):
        titles = [i.title for i in papers]
        titles, idx = np.unique(titles, return_index=True)
        return [papers[i] for i in idx]

    def generate_papers(self) -> List[WebOfSciencePaper]:
        # Construct the papers
        papers = [WebOfSciencePaper(title=i["title"], abstract=i["abstract"],
                                    doi=i["DOI"]) for i in self.data]
        # Create an index that maps back to the title
        idx = list(range(len(self.data)))
        idx_to_title = dict(zip(idx, [i["title"] for i in self.data]))
        # Save this function
        with open("data/idx_to_title.pt", "wb") as file:
            pickle.dump(idx_to_title, file)
        # Add the index to the papers
        for index, paper in zip(idx, papers):
            paper.index = index
        # Delete duplicates
        papers = self._delete_duplicates(papers)
        return papers


class PaperCollection:
    def __init__(self, papers: List[WebOfSciencePaper], n_trees):
        self.papers = papers
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.process_papers()
        self.insert_papers(n_trees=n_trees)

    def process_papers(self):
        unavailable_titles = open("data/unavailable_titles.txt", "r").readlines()
        unavailable_titles = [i.replace("\n", "") for i in unavailable_titles]
        embedding_files = [i.replace(".pt", "") for i in os.listdir("data/embeddings/")]
        print("Embedding files:", embedding_files)
        # Make papers into a nested list of chunks
        processed_papers = []

        for paper in tqdm(self.papers, desc="Processing PDFs", leave=False):
            paper.unavailable = True if paper.title in unavailable_titles else False
            paper.embedding_created = True if paper.title in embedding_files else False
            if paper.unavailable == False:
                if paper.embedding_created == False:
                    print(">>> Loading paper")
                    try:
                        paper.get_pdf()
                        paper.process_pdf()
                    except PaperNotFoundException:
                        print("PaperNotFoundException")
                        print(paper.title)
                        with open("data/unavailable_titles.txt", "a") as file:
                            file.write(paper.title + "\n")
            processed_papers.append(paper)

        for paper in tqdm(processed_papers, desc="Creating embeddings", leave=False):
            if paper.unavailable == False:
                if paper.embedding_created == False:
                    print(">>> Embedding created")
                    paper.get_text_embedding(self.model)
                    with open(f"data/embeddings/{paper.title}.pt", "wb") as file:
                        pickle.dump(paper.text_embedding, file) # list of embeddings
                else:
                    print(">>> Embedding already created")
                    with open(f"data/embeddings/{paper.title}.pt", "rb") as file:
                        paper.text_embedding = pickle.load(file) # list of embeddings

        self.papers = processed_papers

    def insert_papers(self, n_trees):
        self.annoy = annoy.AnnoyIndex(768, 'angular')
        idx = 0
        for paper in tqdm(self.papers, desc="Inserting papers", leave=False):
            for embedding in paper.text_embedding:
                self.annoy.add_item(idx, embedding); idx += 1;
        import time; start = time.time()
        print("Building text collection index...", end="\r")
        self.annoy.build(n_trees)
        print("Number of vectors in text collection index:", self.annoy.get_n_items())
        print("Building index took", round(time.time() - start, 2),
              "seconds based on", n_trees, "trees")
        print(f"Answering your questions based on the {len(self.papers)} papers "
               "that are closest to your selected topics.")

    def query(self, query: str, n_results=10):
        query_embedding = self.model.encode(query).tolist()
        # This index iterates over all sentences of the papers appended
        top_idx = self.annoy.get_nns_by_vector(query_embedding, n_results)
        # Take into account context
        top_idx = [[i-2, i-1, i, i+1, i+2] for i in top_idx]
        top_idx = [i for j in top_idx for i in j]
        # Problem: if you do it like this they lose their ordering
        sentences = [i.text for i in self.papers]
        sentences_unnested = [i for j in sentences for i in j]
        top_sentences = [sentences_unnested[i] for i in top_idx]
        # Delete duplicates
        top_sentences = list(dict.fromkeys(top_sentences))
        return top_sentences


class AbstractCollection:
    def __init__(self, papers: List[WebOfSciencePaper], reload: bool, n_trees: int):
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.papers = papers
        # Load index
        self.annoy = annoy.AnnoyIndex(768, 'angular')
        if not reload: self.annoy.load("data/abstracts.ann")
        # Load mapping from index to title
        self.idx_to_title = pickle.load(open("data/idx_to_title.pt", "rb"))
        if reload: self._insert_abstracts(papers, n_trees=n_trees)
        print("Number of vectors in abstract index:", self.annoy.get_n_items())
        # Save annoy index
        self.annoy.save("data/abstracts.ann")

    def _insert_abstracts(self, papers: List[WebOfSciencePaper], n_trees):
        self.titles = []
        for paper in tqdm(papers, desc="Generating abstract embeddings", leave=False):
            paper.get_abstract_embedding(self.model)
            self.titles.extend([paper.title] * len(paper.abstract_embedding))

        abstract_idx = 0
        for paper in papers:
            for embedding in paper.abstract_embedding:
                self.annoy.add_item(abstract_idx, embedding); abstract_idx += 1;

        import time; start = time.time()
        self.annoy.build(n_trees)
        print("Building abstract index took", time.time() - start,
              "seconds based on", n_trees)

    def query(self, query: str, n_results=10) -> List[WebOfSciencePaper]:
        """Returns the top N most similar papers to the query."""
        query_embedding = self.model.encode(query).tolist()
        top_idx = self.annoy.get_nns_by_vector(query_embedding, n_results)

        idx = 0
        top_papers = []
        for paper in self.papers:
            for _ in paper.abstract:
                if idx in top_idx and paper not in top_papers:
                    top_papers.append(paper)
                idx += 1

        return top_papers


class PaperPal():
    """
    This the chatbot class.
    """
    def __init__(self, topics: str, file_paths: List[str]):
        # Scrape the data file
        self.scraper = WebOfScienceScraper()
        self.scraper.read_tdf(file_paths=file_paths, top_n=5000)
        self.papers = self.scraper.generate_papers()

        # Create embeddings and store them in annoy
        # Set reload=True to generate the abstract embeddings and construct the
        # annoy index. Thus, only set it equal to true the first time that you run
        # the code for a certain WoS export.
        self.abstract_collection = AbstractCollection(self.papers, reload=False, n_trees=20000)

        # Limit the number of results to n_papers most similar the the query such
        # that we don't have to scrape too many pdfs
        # TODO: implement positive and negative queries and examples to select
        # top papers.
        # TODO: add the paper index as the source => relate this to the first author and publication date
        # TODO: don't make the sentences of the article text overlapping, but
        # get the previous and next sentences of the text
        # TODO: make it so that actually 100 papers are returned, and not 100
        # are tested
        top_papers = self.abstract_collection.query(topics, n_results=300)

        # Get the pdf and full text of the top papers
        self.paper_collection = PaperCollection(top_papers, n_trees=10000)

    def chat(self):
        # response = self._summarize_sentences(user_input, top_sentences)
        # print(response)
        prompt = PromptTemplate(input_variables=["input", "text", "history"],
                                template=paper_pal_template)
        memory = ConversationBufferWindowMemory(k=5, input_key="input",
                                                memory_key="history")
        chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt, memory=memory,
                         verbose=True)
        print("Hello! I am PaperPal. I can help you explore the literature on a "
              "topic of your choice. Type your question below or 'quit' to exit.")
        while True:
            user_input = input(">>> ")
            if user_input == "quit":
                break
            else:
                top_sentences = self.paper_collection.query(user_input, n_results=20)
                top_sentences = " \n".join(top_sentences)
                output = chain.predict(text=top_sentences, input=user_input)
                print(output)


def main():
    paper_pal = PaperPal(topics="Recency, frequency, monetary",
                         file_paths=["data/savedrecs_1_RFM.txt", "data/savedrecs_2_RFM.txt"])
    paper_pal.chat()


if __name__ == "__main__":
    main()
 