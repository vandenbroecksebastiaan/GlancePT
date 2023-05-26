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
import numpy as np
import torch
import nltk
from typing import List
import os
from tqdm import tqdm
import argparse
import umap
import matplotlib.pyplot as plt
from multiprocessing import Pool
import textwrap

import openai
from sentence_transformers import SentenceTransformer, util
openai.api_key = os.environ["OPENAI_API_KEY"]


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
            time.sleep(0.1)
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.link = link
        paper_info = requests.get(f"https://paperswithcode.com/api/v1/papers/{self.link}")
        paper_info = paper_info.json()

        if "detail" in paper_info.keys():
            raise PaperNotFoundException(self.link)

        self.id = paper_info["id"]
        self.arxiv_id = paper_info["arxiv_id"]
        self.url_pdf = paper_info["url_pdf"]
        self.title = paper_info["title"]
        self.abstract = paper_info["abstract"]
        self.authors = paper_info["authors"]
        self.published = paper_info["published"]
        
        self.process_abstract()
        self.get_full_text()
        
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

        # Remove sentences from the text that are in the abstract
        for line in self.abstract:
            if line.lower() in pdf_text.lower():
                start_index = pdf_text.lower().find(line.lower())
                end_index = start_index + len(line)
                pdf_text = pdf_text[:start_index] + pdf_text[end_index:]
        
        # Make sure that "references" is only mentioned once and delete the
        # references section
        if "references" in pdf_text.lower():
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
        # Delete the first sentence, since it contains the title and names
        # of the authors and is not very informative
        sentences = sentences[1:]
        # Make sentences into a non-overlapping sliding window of 3 elements each
        # sentences = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
        
        n_words = len("".join(pdf_text).split(" "))
        n_chars = len("".join(pdf_text))
        print(f">>> There are about {n_words} words and {n_chars} chars in " \
              f"the paper {self.title}")

        self.pdf_text = pdf_text
        self.sentences = sentences
        
        
    def process_abstract(self):
        self.abstract = re.sub(r'\[[^\]]*\]', '', self.abstract)
        self.abstract = re.sub(r'e\.g\.', 'for example', self.abstract)
        self.abstract = re.sub(r'i\.e\.', 'for example', self.abstract)
        self.abstract = re.sub(r'i\.i\.d\.', 'independent and identically distributed', self.abstract)
        self.abstract = re.sub(r'etc\.', 'etc', self.abstract)
        self.abstract = re.sub(r'rst', '', self.abstract)
        self.abstract = re.sub(r'- ', '', self.abstract)
        # TODO: move this to one of the last steps
        # self.abstract = re.sub(r' We ', ' The authors ', self.abstract)
        # self.abstract = re.sub(r' we ', ' the authors ', self.abstract)
        # self.abstract = re.sub(r' I ', ' the author ', self.abstract)
        self.abstract = nltk.tokenize.sent_tokenize(self.abstract)
        # Remove all sentences that have a link in them
        self.abstract = [i for i in self.abstract if not bool(re.search(r'https?://\S+|www\.\S+', i))]
        
    def get_embeddings(self, n_sentences=5):
        # model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
        model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
        text_embeddings = model.encode(self.sentences)
        self.text_embeddings = np.vstack(text_embeddings)
        abstract_embeddings = model.encode(self.abstract)
        self.abstract_embeddings = np.vstack(abstract_embeddings)
        del model; torch.cuda.empty_cache();

        self.most_important_sentences = []
        for abstract_embedding in self.abstract_embeddings:
            sentence_dot_scores = []
            for idx, text_embedding in enumerate(self.text_embeddings):
                if idx==len(self.text_embeddings)-1: next_idx = idx
                else: next_idx = idx + 1
                dot_score = util.cos_sim(abstract_embedding, text_embedding)
                sentence_dot_scores.append((self.sentences[idx]+self.sentences[next_idx], dot_score))
                sentence_dot_scores = sorted(sentence_dot_scores, key=lambda x: x[1],
                                             reverse=True)
            
            self.most_important_sentences.extend([i[0] for i in sentence_dot_scores[:n_sentences]])

    def get_summary(self):
        prompt = """
You have been tasked with providing an overview of the following text.
Provide a general overview and also try to answer the following questions:
What are the new findings? What problem does the proposed method solve? \
What can the new method be used for in the future? \
What are the limitations?
Try to end the paper with a take-home message or topic sentence.
Start the summary with the main topic of the text. \
For example, "FastComposer is a new method for text-to-image generation without fine-tuning ...".
Use the third person when referring to the authors of the paper.

"""
        for i in self.most_important_sentences: prompt += i + "\n"
        self.summary = self._gpt_completion_call(prompt)

    def get_clusters(self, n_clusters=1):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=1337, n_init="auto") \
                     .fit(self.text_embeddings)
        self.clusters = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_
    
    def get_closest_sentences(self, top=10):
        # Get the closest sentence to each cluster center
        closest_sentences = []
        for i in range(len(self.cluster_centers)):
            distances = np.linalg.norm(self.text_embeddings - self.cluster_centers[i], axis=1)
            closest_sentences.append(np.argsort(distances)[:top])
                
    def visualize_embedding(self, n_neighbors=10):
        red_embeddings = umap.UMAP(n_neighbors=n_neighbors) \
                             .fit_transform(self.text_embeddings)
        plt.figure(figsize=(10,10))
        plt.scatter(red_embeddings[:,0], red_embeddings[:,1],
                    c=np.arange(red_embeddings.shape[0]), cmap="viridis")
        plt.colorbar()
        plt.savefig(f"visualizations/{self.title}.png", dpi=300, bbox_inches='tight')
        
    def _gpt_completion_call(self, prompt):
        for idx in range(10):
            try: 
                completion = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=[{"role": "user", "content": prompt}]
                )
                return completion["choices"][0]["message"]["content"]
            except openai.error.RateLimitError:
                print(f"Error in GPT-3 call: Rate limit exceeded. Trying again... {idx}")
                
class PaperNotFoundException(Exception):
    pass

def process_paper(link):
    try:
        paper = Paper(link)
        paper.get_embeddings(n_sentences=3)
        paper.get_summary()
        # paper.visualize_embedding(n_neighbors=5)
        return paper
    except PaperNotFoundException as e:
        print(f"Oh no an exception >:( with paper {str(e)}")
        return None
              
def make_abstract_visualization(abstracts: List, paper_titles: List):
    abstract_lengths = [len(i) for i in abstracts]
    abstracts = [i for j in abstracts for i in j]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    embeddings = model.encode(abstracts)
    red_embeddings = umap.UMAP(n_neighbors=10).fit_transform(embeddings)

    # Generate a colormap based on paper titles
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    while len(colors) < len(set(paper_titles)):
        colors += colors

    title_to_color = {}
    for idx, title in enumerate(list(set(paper_titles))):
        title_to_color[title] = colors[idx]
              
    colors = [[title_to_color[i]]*length for i, length in zip(paper_titles, abstract_lengths)]
    colors = [i for j in colors for i in j]
    labels = [[title]*length for title, length in zip(paper_titles, abstract_lengths)]
    labels = [i for j in labels for i in j]
    labels = [i.split(":")[0] for i in labels]
              
    # Only keep the first label per paper, since we want to avoid duplicate
    # elements in the legend
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)
        else:
            unique_labels.append('')
    labels = unique_labels

    # Add a linebreak to the labels if they are longer than 30 characters
    wrapped_labels = [textwrap.fill(label, 30) for label in labels]

    fig, ax = plt.subplots(figsize=(7, 7))
    # The loop is necessary to get the legend to work
    for (x, y), color, label in zip(red_embeddings, colors, wrapped_labels):
        plt.scatter(x, y, c=color, label=label, alpha=0.8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title("")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(f"visualizations/abstracts.png", dpi=300, bbox_inches='tight')

def main():
    # Get vargs
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_papers", type=int, default=4)
    parser.add_argument("--email", type=str, required=True)
    parser.add_argument("--n_processes", type=int, default=8)
    args = parser.parse_args()

    n_papers = args.n_papers
    email = args.email
    n_processes = args.n_processes

    # Scrape the site and process the papers
    scraper = Scraper(n_papers=n_papers)
    with Pool(n_processes) as pool: 
        papers = pool.map(process_paper, list(scraper))
        
    # These are the papers for which the API did not work
    papers = [i for i in papers if i is not None]
    paper_titles = [i.title for i in papers]
    paper_summaries = [i.summary for i in papers]
    abstracts = [i.abstract for i in papers]
    
    print("here")
    make_abstract_visualization(abstracts, paper_titles)
    
    # Send the mail
    email_sender = EmailClient()
    email_sender.make_email(paper_titles, paper_summaries, email)
    email_sender.send_email()

if __name__ == "__main__": main()
