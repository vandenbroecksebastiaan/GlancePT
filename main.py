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
import numpy as np
import nltk
import os
from tqdm import tqdm

from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

from sentence_transformers import SentenceTransformer, util

from email_client import EmailClient


class Scraper:
    def __init__(self, n_papers=50):
        self.n_papers = n_papers
        self.paper_links = []
        self.driver = self.get_driver()
        self.html = self.get_papers("https://www.paperswithcode.com")

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
        pdf_text = re.sub("\s+", " ", pdf_text)
        pdf_text = pdf_text.replace("\n", "") \
                           .replace("\t", "") \
                           .encode("ascii", "ignore") \
                           .decode("ascii")
        
        # Make sure that "references" is only mentioned once and delete the
        # references section
        if pdf_text.lower().count("references") < 2: print("MULTPLE REFERENCES")
        references_index = pdf_text.lower().rfind("references")
        pdf_text = pdf_text[:references_index]

        # We do this to improve the quality of sentence splitting
        pdf_text = re.sub(r'\[[^\]]*\]', '', pdf_text)
        pdf_text = re.sub(r'e\.g\.', 'for example', pdf_text)
        pdf_text = re.sub(r'i\.e\.', 'for example', pdf_text)
        pdf_text = re.sub(r'i\.i\.d\.', 'independent and identically distributed', pdf_text)
        pdf_text = re.sub(r'etc\.', 'etc', pdf_text)
        pdf_text = re.sub(r'rst', '', pdf_text)
        pdf_text = re.sub(r'- ', '', pdf_text)
        pdf_text = re.sub(r'We ', 'The authors ', pdf_text)
        pdf_text = re.sub(r'we ', 'the authors ', pdf_text)
        pdf_text = re.sub(r'I ', 'the author ', pdf_text)
        
        # Remove non-english words, we do this to remove names of authors and
        # places, but it may also remove words that are not in the vocab
        # and are important for the paper

        sentences = nltk.tokenize.sent_tokenize(pdf_text)
        
        # Merge short sentences with the previous sentence
        for i in range(len(sentences)):
            if len(sentences[i]) < 20:
                sentences[i-1] += sentences[i]
                sentences[i] = ""
        sentences = [i for i in sentences if i != ""]
        sentences = [i for i in sentences if "keywords" not in i.lower()]
        sentences = [i for i in sentences if "arxiv" not in i.lower()]
        
        # Delete short sentences
        sentences = [i for i in sentences if len(i) > 5]
        # Delete the first sentence, since it contains the title and names
        # of the authors and is not very informative
        sentences = sentences[1:]
        
        # Make sentences into a non-overlapping sliding window of 3 elements each
        sentences = ["".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
        
        n_words = len("".join(pdf_text).split(" "))
        n_chars = len("".join(pdf_text))
        print(f">>> There are about {n_words} words and {n_chars} chars in " \
              f"the paper {self.title}")

        self.pdf_text = pdf_text
        self.sentences = sentences
        
    def get_embeddings(self, n_sentences=5):
        model = SentenceTransformer('all-mpnet-base-v2', device="cuda")
        embeddings = model.encode(self.sentences)
        self.embeddings = np.vstack(embeddings)

        query = self.title + self.abstract \
                + "What is the most important sentence? What are the new findings?" 
        query_embedding = model.encode(query)
        
        sentence_dot_scores = []
        for idx, embedding in enumerate(self.embeddings):
            dot_score = util.dot_score(query_embedding, embedding)
            sentence_dot_scores.append((self.sentences[idx], dot_score))
        
        # Sort sentences by dot scores
        sentence_dot_scores = sorted(sentence_dot_scores, key=lambda x: x[1],
                                     reverse=True)
        
        for i, j in sentence_dot_scores[:n_sentences]: print(i)
        
        self.most_important_sentences = [i[0] for i in sentence_dot_scores[:n_sentences]]

    def get_summary(self):
        prompt = """
You have been tasked with providing an overview of the following text.
Start the overview by the passive form of what the text is about.
For example, "A new method called ... was proposed ...".

"""
        for i in self.most_important_sentences: prompt += i + "\n"
        self.summary = self._gpt_completion_call(prompt)

    """
    def get_clusters(self, n_clusters=1):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=1337, n_init="auto") \
                     .fit(self.embeddings)
        self.clusters = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_
    
    def get_closest_sentences(self, top=10):
        # Get the closest sentence to each cluster center
        closest_sentences = []
        for i in range(len(self.cluster_centers)):
            distances = np.linalg.norm(self.embeddings - self.cluster_centers[i], axis=1)
            print(np.argsort(distances)[:top])
            closest_sentences.append(np.argsort(distances)[:top])
            
        for i in closest_sentences:
            print("-"*100)
            for j in i:
                print(self.sentences[j])
        
    def summarize(self):
        text_splitter = CharacterTextSplitter(
            chunk_size=2000, chunk_overlap=0, separator=" "
        )
        texts = text_splitter.split_text(self.pdf_text)
        print("+++", len(texts))
        for i in texts: print("---", len(i.split(" ")))
        docs = [Document(page_content=t) for t in texts]
        llm = OpenAI(temperature=0, batch_size=5, model="curie")
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        self.summary = chain.run(docs)
        print(">>> Summary:", self.summary)
    """
        
    def _gpt_completion_call(self, prompt):
        completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "user", "content": prompt}
          ]
        )
        return completion["choices"][0]["message"]["content"]
        

def main():
    # Scrape the site
    scraper = Scraper(n_papers=2)
    print(">>> len scraper:", len(scraper.paper_links))

    # Process the papers
    paper_titles = []
    paper_summaries = []
    for link in tqdm(scraper, desc="Processing papers", total=len(scraper)):
        paper = Paper(link)
        paper.get_full_text()
        paper.get_embeddings()
        paper.get_summary()
        paper_titles.append(paper.title); paper_summaries.append(paper.summary);
    
    # Send the mail
    email_sender = EmailClient()
    email_sender.make_email(paper_titles, paper_summaries)
    email_sender.send_email("van.den.broeck.sebastiaan@gmail.com")

if __name__ == "__main__": main()