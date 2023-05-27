# GlancePT

## What is it?

This project is essentially an intelligent web scraping tool. It leverages
OpenAI's GPT-3.5, Selenium, PyPDF2 and Papers With Code's API to generate
summaries of the most popular papers on Papers With Code. An email is then sent
to notify users of this information.

The abstract often gives a good overview of what a paper is about, but it is
not written to be a summary of the paper. Instead, an abstract discusses
objectives, methods or approaches and results. Often, the abstract is written
in a somewhat boring way as well, since this is simply the style of most
academic articles.

By combining web scraping, natural language processing and email integration,
this project offers a solution for anybody that wants to be better informed
about the latest trends in machine learning and AI. 

## How does it work?

1. Scrape the site of Papers With Code to collect the most popular papers and
get the article text and details with their API.

    Papers With Code is scraped using Selenium and Beautiful Soup in order to
    get the most popular papers. Next, one can retrieve more information about
    the articles themselves using their API.
    Why don't you use the Papers With Code API directly to get the most popular
    papers? I could not get this working with their API. It does allow to sort
    by recency, but not by the popularity that you see on its home page.
   
2. Process the papers. 

    The processing of the pdf files could probably be improved a little bit. I
    just open the pdf file and extract its text with PyPDF2. Afterwards, there
    are still some special characters from the binary nature of the pdf file so
    those are removed. Then, the references and hyperlinks are taken out as
    well.

3. Generate embeddings.

    I use the Sentence Transformer library to do this. Embeddings are generated
    for each sentence in the abstract and article text.

3. Get the most informative sentences from the article text using the embeddings.

    I assume that the abstract talks about the most important aspects of the
    work, but drops important details. Remember, the goal of the abstract is to
    get you to read the paper. In other words, there are still some important
    details in the article text that we need to get in order to make a
    high-quality summary. For each sentence in the abstract, I then get the most
    informative sentences from the text.

4. Ask GPT-3.5 to write a summary.

    It's a pretty basic prompt, but GPT-3.5 is able to generate a higher
    quality summary, since it has more information than just the abstract.

5. Finally, an email with the summaries is sent.


## How can you get it working?

Create a virtual environment and install the required packages.
Have your OpenAI API key in your environment:

```bash
export OPENAI_API_KEY=YOUR_API_KEY_HERE
```

You can set up a Cron job to run this script daily or whenever you would like.
Speed is not the main concern here, but since the embeddings are generated
locally you do need to have some RAM. I believe 6 gigs should be plenty. Having
a GPU in your system would be ideal, but is not necessary. Another thing to
take into account is that I use a multiprocessing pool to speed up this
process a little bit. Which means that the embedding model is going to be
loaded a certain number of times. If you don't have a lot of (V)RAM available,
you may choose to lower the number of processes in the pool.

## Why don't you directly feed the whole text to GPT-3.5 or use LangChain?

I recently took a look at the Usage section of my account on OpenAI and things
are not looking good. To restrict token usage, I only want to use the most
informative sentences from the article.

## Example: this is what such a mail could look like

<img width="1212" alt="Screenshot 2023-05-27 at 14 07 09" src="https://github.com/vandenbroecksebastiaan/GlancePT/assets/101555259/4c7d0f6a-1bff-44e9-9258-4d50c3000137">

