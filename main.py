from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromiumService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.utils import ChromeType

from bs4 import BeautifulSoup
import requests
import PyPDF2
from io import BytesIO
import re
import time
from multiprocessing import Pool

from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


class Scraper:
    def __init__(self, n_papers=50):
        self.n_papers = n_papers
        self.paper_links = []
        self.driver = self.get_driver()
        self.html = self.get_papers("https://www.paperswithcode.com")
        # self.get_paper_links()

    def get_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument("--remote-debugging-port=9222") # do not delete
        driver = webdriver.Chrome(
            options=options,
            service=ChromiumService(
                ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
            )
        )
        return driver
    
    def get_papers(self, url):
        self.driver.get(url)
        # Scroll to the bottom of the page until n_papers is reached
        papers = []
        while len(papers) < self.n_papers:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            papers = soup.find_all('a', href=True)
            papers = [i["href"] for i in papers]
            papers = [i for i in papers if i.startswith("/paper/")]
            papers = [i for i in papers if "#" not in i]
            papers = list(dict.fromkeys(papers))

        print("+++", len(papers))
        self.paper_links = [i.split("/")[-1] for i in papers][:self.n_papers]
        return self.driver.page_source
        
    def __len__(self):
        return len(self.paper_links)
        
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.paper_links):
            value = self.paper_links[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration


class Paper:
    def __init__(self, link):
        self.link = link
        paper_info = requests.get(f"https://paperswithcode.com/api/v1/papers/{self.link}")
        paper_info = paper_info.json()
        self.id = paper_info["id"]
        self.arxiv_id = paper_info["arxiv_id"]
        self.url_pdf = paper_info["url_pdf"]
        self.title = paper_info["title"]
        self.abstract = paper_info["abstract"]
        self.authors = paper_info["authors"]
        self.published = paper_info["published"]
        
    def get_full_text(self):
        pdf_file = requests.get(self.url_pdf)
        reader = PyPDF2.PdfReader(BytesIO(pdf_file.content))
        pdf_text = [reader.pages[i].extract_text() for i in range(len(reader.pages))]
        pdf_text = "".join(pdf_text)
        pdf_text = pdf_text.replace("\n", "") \
                           .replace("\t", "") \
                           .encode("ascii", "ignore") \
                           .decode("ascii")
        
        pdf_text = re.sub("\s+", " ", pdf_text)
        # assert len(pdf_text.split(" ")) < 10000
        n_words = len(pdf_text.split(" "))
        print(f">>> There are {n_words} words in the paper {self.title}")
        self.pdf_text = pdf_text
        
    def summarize(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=0
        )
        texts = text_splitter.split_text(self.pdf_text)
        docs = [Document(page_content=t) for t in texts]
        llm = OpenAI(temperature=0, batch_size=5)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        self.summary = chain.run(docs)
        
def process_paper(link):
    paper = Paper(link)
    paper.get_full_text()
    # paper.summarize()
    return paper


def main():
    # Scrape the site
    scraper = Scraper(n_papers=50)
    print(">>> len scraper:", len(scraper.paper_links))
    # Process the papers
    with Pool(10) as pool: papers = pool.map(process_paper, scraper.paper_links)
    

if __name__ == "__main__": main()