import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import nltk
nltk.download('punkt')


# Load environment variables for OpenAI key, etc.
load_dotenv()

# -- Web Scraping Function --
def fetch_articles_from_urls(url_list):
    try:
        from newspaper import Article
    except ImportError:
        st.error("Please install 'newspaper3k' (add to requirements.txt).")
        return []
    articles = []
    for url in url_list:
        try:
            article = Article(url)
            article.download()
            article.parse()
            content = article.title + "\n" + article.text
            articles.append({"url": url, "content": content})
        except Exception as e:
            st.warning(f"Failed to fetch {url}: {e}")
    return articles

# -- Market Research Workflow --
def run_market_research(articles):
    if not articles:
        return None
    raw_content = "\n".join([f"URL: {a['url']}\n{a['content']}\n{'='*40}\n" for a in articles])

    # Prompts
    content_extractor_prompt = """You are a content extraction agent. Extract the most important factual information, statistics, and statements from the following source material. Ignore fluff, advertisements, and unrelated commentary.
Source Material:
{raw_content}
Extracted Key Content:
"""

    sentiment_prompt = """You are a sentiment analysis agent. Given the following extracted content, assess the overall sentiment (positive, negative, neutral) and explain the reasoning. Mention if sentiment varies by topic.
Extracted Content:
{extracted_content}
Sentiment Analysis:
"""

    trend_prompt = """You are a trend analysis agent. Identify any recurring themes, trends, new products, or noteworthy changes mentioned in the extracted content. Highlight anything that appears multiple times or is emphasized across different sources.
Extracted Content:
{extracted_content}
Identified Trends and Topics:
"""

    report_writer_prompt = """You are an executive report writer. Based on the extracted content, sentiment analysis, and trends, write a concise summary (max 300 words) suitable for a business executive. Focus on actionable insights, market opportunities, or risks.
Extracted Content:
{extracted_content}
Sentiment Analysis:
{sentiment_analysis}
Trends and Topics:
{trends}
Executive Summary:
"""

    # Chains
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, streaming=True)

    content_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["raw_content"], template=content_extractor_prompt)
    )
    sentiment_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["extracted_content"], template=sentiment_prompt)
    )
    trend_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["extracted_content"], template=trend_prompt)
    )
    report_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["extracted_content", "sentiment_analysis", "trends"],
            template=report_writer_prompt
        )
    )

    # Multi-agent workflow
    extracted_content = content_chain.run(raw_content=raw_content)
    sentiment_analysis = sentiment_chain.run(extracted_content=extracted_content)
    trends = trend_chain.run(extracted_content=extracted_content)
    summary = report_chain.run(
        extracted_content=extracted_content,
        sentiment_analysis=sentiment_analysis,
        trends=trends
    )
    return {
        "extracted_content": extracted_content,
        "sentiment_analysis": sentiment_analysis,
        "trends": trends,
        "summary": summary
    }

# --- Streamlit UI ---
st.title("📰 Automated Multi-Agent Market Research")

st.markdown("""
This app scrapes news articles from the URLs you provide, extracts key facts, analyzes sentiment and trends, and gives you an executive summary—all powered by OpenAI and LangChain!
""")

url_input = st.text_area("Paste one or more URLs (one per line):", height=100)
sample = st.button("Insert Example URLs")
if sample:
    url_input = "https://www.nytimes.com/2024/06/20/business/ai-market-news.html\nhttps://www.forbes.com/sites/forbesbusinesscouncil/2024/06/19/emerging-trends-in-ai-business/"

if st.button("Run Market Research"):
    urls = [u.strip() for u in url_input.splitlines() if u.strip()]
    if not urls:
        st.error("Please enter at least one URL.")
    else:
        with st.spinner("Fetching articles and running multi-agent workflow..."):
            articles = fetch_articles_from_urls(urls)
            if not articles:
                st.error("No articles were fetched.")
            else:
                st.success(f"Fetched {len(articles)} articles.")
                results = run_market_research(articles)
                if results:
                    with st.expander("🔑 Key Extracted Content"):
                        st.write(results["extracted_content"])
                    with st.expander("😊 Sentiment Analysis"):
                        st.write(results["sentiment_analysis"])
                    with st.expander("📈 Trends & Topics"):
                        st.write(results["trends"])
                    st.subheader("💡 Executive Summary")
                    st.write(results["summary"])
                else:
                    st.error("Error in processing market research workflow.")

st.markdown("---")
st.caption("Built with LangChain, OpenAI, and Streamlit.")

