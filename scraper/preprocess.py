import json

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def clean_text(text):
    # Add more cleaning steps as needed
    text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u00a0', ' ').replace('\u2122', 'TM').replace('\u2013', '-')
    return text

def preprocess_data(data):
    for website, faqs in data.items():
        for faq in faqs:
            faq['question'] = clean_text(faq['question'])
            faq['answer'] = clean_text(faq['answer'])
    return data

def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    input_filename = 'faqs.json'
    output_filename = 'cleaned_faqs.json'

    data = load_data(input_filename)
    cleaned_data = {}

    for website_data in data:
        for website, faqs in website_data.items():
            cleaned_data[website] = preprocess_data({website: faqs})[website]

    save_data(cleaned_data, output_filename)

    print(f"Preprocessed data saved to {output_filename}")
