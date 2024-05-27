import requests
from bs4 import BeautifulSoup
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_webpage(url):
    """Fetches the webpage content from the given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching the webpage: {e}")
        return None

def parse_faqs(html_content, website):
    """Parses the FAQ questions and answers from the HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    faq_elements = soup.find_all("div", class_="rich-text-body")
    
    faqs = []
    for i in range(0, len(faq_elements) - 1, 2):
        question = faq_elements[i].get_text(strip=True)
        answer = faq_elements[i + 1].get_text(strip=True)
        faqs.append({"question": question, "answer": answer})
    faqs = [{website: faqs}]
    return faqs

def save_to_json(data, filename):
    """Saves the data to a JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"FAQs saved to {filename}")
    except IOError as e:
        logging.error(f"Error saving to JSON file: {e}")

def main(url, output_file, website):
    """Main function to fetch, parse, and save FAQs."""
    html_content = fetch_webpage(url)
    if html_content:
        faqs = parse_faqs(html_content, website)
        save_to_json(faqs, output_file)

if __name__ == "__main__":
    URL = "https://www.databricks.com/product/faq"
    OUTPUT_FILE = "faqs.json"
    WEBSITE = "databricks"
    
    main(URL, OUTPUT_FILE, WEBSITE)
