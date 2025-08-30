from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os
import math
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Harari to English character mapping
HARARI_TO_ENGLISH = {
    'ሀ': 'he', 'ሁ': 'hu', 'ሂ': 'hi', 'ሃ': 'ha', 'ሄ': 'hē', 'ህ': 'h', 'ሆ': 'ho',
    'ለ': 'le', 'ሉ': 'lu', 'ሊ': 'li', 'ላ': 'la', 'ሌ': 'lē', 'ል': 'l', 'ሎ': 'lo',
    'ሐ': 'Hxe', 'ሑ': 'Hxu', 'ሒ': 'Hxi', 'ሓ': 'Hxa', 'ሔ': 'Hxē', 'ሕ': 'Hx', 'ሖ': 'Hxo',
    'መ': 'me', 'ሙ': 'mu', 'ሚ': 'mi', 'ማ': 'ma', 'ሜ': 'mē', 'ም': 'm', 'ሞ': 'mo',
    'ሠ': 'se', 'ሡ': 'su', 'ሢ': 'si', 'ሣ': 'sa', 'ሤ': 'sē', 'ሥ': 's', 'ሦ': 'so',
    'ረ': 're', 'ሩ': 'ru', 'ሪ': 'ri', 'ራ': 'ra', 'ሬ': 'rē', 'ር': 'r', 'ሮ': 'ro',
    'ሰ': 'Sse', 'ሱ': 'Ssu', 'ሲ': 'Ssi', 'ሳ': 'Ssa', 'ሴ': 'Ssē', 'ስ': 'Ss', 'ሶ': 'Sso',
    'ሸ': 'she', 'ሹ': 'shu', 'ሺ': 'shi', 'ሻ': 'sha', 'ሼ': 'shē', 'ሽ': 'sh', 'ሾ': 'sho',
    'ቀ': 'qe', 'ቁ': 'qu', 'ቂ': 'qi', 'ቃ': 'qa', 'ቄ': 'qē', 'ቅ': 'q', 'ቆ': 'qo',
    'በ': 'be', 'ቡ': 'bu', 'ቢ': 'bi', 'ባ': 'ba', 'ቤ': 'bē', 'ብ': 'b', 'ቦ': 'bo',
    'ተ': 'te', 'ቱ': 'tu', 'ቲ': 'ti', 'ታ': 'ta', 'ቴ': 'tē', 'ት': 't', 'ቶ': 'to',
    'ቸ': 'che', 'ቹ': 'chu', 'ቺ': 'chi', 'ቻ': 'cha', 'ቼ': 'chē', 'ች': 'ch', 'ቾ': 'cho',
    'ኀ': 'hha', 'ኁ': 'hhu', 'ኂ': 'hhi', 'ኃ': 'hha', 'ኄ': 'hhē', 'ኅ': 'hh', 'ኆ': 'hho',
    'ነ': 'ne', 'ኑ': 'nu', 'ኒ': 'ni', 'ና': 'na', 'ኔ': 'nē', 'ን': 'n', 'ኖ': 'no',
    'ኘ': 'nye', 'ኙ': 'nyu', 'ኚ': 'nyi', 'ኛ': 'nya', 'ኜ': 'nyē', 'ኝ': 'ny', 'ኞ': 'nyo',
    'አ': 'xe', 'ኡ': 'xu', 'ኢ': 'xi', 'ኣ': 'xa', 'ኤ': 'xē', 'እ': 'x', 'ኦ': 'xo',
    'ከ': 'ke', 'ኩ': 'ku', 'ኪ': 'ki', 'ካ': 'ka', 'ኬ': 'kē', 'ክ': 'k', 'ኮ': 'ko',
    'ኸ': 'xhe', 'ኹ': 'xhu', 'ኺ': 'xhi', 'ኻ': 'xha', 'ኼ': 'xhē', 'ኽ': 'xh', 'ኾ': 'xho',
    'ወ': 'we', 'ዉ': 'wu', 'ዊ': 'wi', 'ዋ': 'wa', 'ዌ': 'wē', 'ው': 'w', 'ዎ': 'wo',
    'ዐ': 'xxe', 'ዑ': 'xxu', 'ዒ': 'xxi', 'ዓ': 'xxa', 'ዔ': 'xxē', 'ዕ': 'xxi', 'ዖ': 'xxo',
    'ዘ': 'ze', 'ዙ': 'zu', 'ዚ': 'zi', 'ዛ': 'za', 'ዜ': 'zē', 'ዝ': 'z', 'ዞ': 'zo',
    'ዠ': 'zhe', 'ዡ': 'zhu', 'ዢ': 'zhi', 'ዣ': 'zha', 'ዤ': 'zhē', 'ዥ': 'zh', 'ዦ': 'zho',
    'የ': 'ye', 'ዩ': 'yu', 'ዪ': 'yi', 'ያ': 'ya', 'ዬ': 'yē', 'ይ': 'y', 'ዮ': 'yo',
    'ደ': 'de', 'ዱ': 'du', 'ዲ': 'di', 'ዳ': 'da', 'ዴ': 'dē', 'ድ': 'd', 'ዶ': 'do',
    'ጀ': 'je', 'ጁ': 'ju', 'ጂ': 'ji', 'ጃ': 'ja', 'ጄ': 'jē', 'ጅ': 'j', 'ጆ': 'jo',
    'ገ': 'ge', 'ጉ': 'gu', 'ጊ': 'gi', 'ጋ': 'ga', 'ጌ': 'gē', 'ግ': 'g', 'ጎ': 'go',
    'ጠ': 'Tte', 'ጡ': 'Ttu', 'ጢ': 'Tti', 'ጣ': 'Tta', 'ጤ': 'Ttē', 'ጥ': 'Tt', 'ጦ': 'Tto',
    'ጨ': 'Ce', 'ጩ': 'Cu', 'ጪ': 'Ci', 'ጫ': 'Ca', 'ጬ': 'Cē', 'ጭ': 'C', 'ጮ': 'Co',
    'ጰ': 'Ppe', 'ጱ': 'Ppu', 'ጲ': 'Ppi', 'ጳ': 'Ppa', 'ጴ': 'Ppē', 'ጵ': 'Pp', 'ጶ': 'Ppo',
    'ጸ': 'Tse', 'ጹ': 'Tsu', 'ጺ': 'Tsi', 'ጻ': 'Tsa', 'ጼ': 'Tsē', 'ጽ': 'Ts', 'ጾ': 'Tso',
    'ፀ': 'Tsse', 'ፁ': 'Tssu', 'ፂ': 'Tssi', 'ፃ': 'Tssa', 'ፄ': 'Tssē', 'ፅ': 'Tss', 'ፆ': 'Tsso',
    'ፈ': 'fe', 'ፉ': 'fu', 'ፊ': 'fi', 'ፋ': 'fa', 'ፌ': 'fē', 'ፍ': 'f', 'ፎ': 'fo',
    'ፐ': 'pe', 'ፑ': 'pu', 'ፒ': 'pi', 'ፓ': 'pa', 'ፔ': 'pē', 'ፕ': 'p', 'ፖ': 'po',
}

def transliterate_harari_to_english(text):
    result = ''
    for char in text:
        result += HARARI_TO_ENGLISH.get(char, char)
    return result.capitalize()

def english_to_harari(text):
    ENGLISH_TO_HARARI = {}
    for harari, english in HARARI_TO_ENGLISH.items():
        ENGLISH_TO_HARARI[english.lower()] = harari

    text = text.lower()
    result = ''
    i = 0
    while i < len(text):
        found = False
        for length in [3, 2, 1]:
            if i + length <= len(text):
                substr = text[i:i+length]
                if substr in ENGLISH_TO_HARARI:
                    result += ENGLISH_TO_HARARI[substr]
                    i += length
                    found = True
                    break
        if not found:
            result += text[i]
            i += 1
    return result

def harari_stem(word):
    vowels = "aeiouē"

    # Define suffix and prefix lists
    suffixes = ["zēw","zolē","zē","zo","ni","wa","lē","um","bēm","tany","w"]
    prefixes = ["ye"]

    # Remove prefixes
    for prefix in prefixes:
        if word.startswith(prefix):
            word = word[len(prefix):]
            break

    # Remove suffixes
    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break

    # Remove vowels
    word = ''.join([char for char in word if char not in vowels])

    return word

def process_corpus(corpus_dir="./CorpusTxt", title_weight=5):
    """Process corpus with title weighting and tracking"""
    word_index = defaultdict(lambda: defaultdict(int))
    doc_names = []
    max_freqs = {}
    doc_word_counts = {}
    doc_titles = {}
    doc_contents = {}

    files = sorted(
        [f for f in os.listdir(corpus_dir) if f.endswith('.txt')],
        key=lambda x: int(x[3:-4]) if x.startswith('Doc') else x
    )

    for filename in files:
        doc_name = os.path.splitext(filename)[0]
        doc_names.append(doc_name)
        local_counts = defaultdict(int)

        with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as f:
            content = f.read().splitlines()

        # Store document content
        doc_titles[doc_name] = content[0] if content else ""
        doc_contents[doc_name] = content[1:] if len(content) > 1 else []

        # Process title and content
        if content:
            # Title processing with weighting
            for word in content[0].lower().split():
                stemmed_word = harari_stem(transliterate_harari_to_english(word).lower())
                local_counts[stemmed_word] += title_weight

            # Content processing
            for line in content[1:]:
                for word in line.lower().split():
                    stemmed_word = harari_stem(transliterate_harari_to_english(word).lower())
                    local_counts[stemmed_word] += 1

        # Track max frequency and store counts
        if local_counts:
            max_freq = max(local_counts.values())
        else:
            max_freq = 1
        max_freqs[doc_name] = max_freq
        doc_word_counts[doc_name] = local_counts

        # Update global index
        for word, count in local_counts.items():
            word_index[word][doc_name] = count

    return word_index, doc_names, max_freqs, doc_word_counts, doc_titles, doc_contents

# Initialize search engine components at startup
class SearchEngine:
    def __init__(self):
        self.doc_vectors = None
        self.doc_norms = None
        self.doc_titles = None
        self.doc_contents = None
        self.idf = None
        self.doc_names = None
        self.initialize_engine()

    def initialize_engine(self):
        word_index, doc_names, max_freqs, doc_word_counts, doc_titles, doc_contents = process_corpus()
        total_docs = len(doc_names)
        idf = {}

        for word in word_index:
            df = len(word_index[word])
            idf[word] = math.log(total_docs / df) if df else 0

        doc_vectors = {}
        doc_norms = {}
        for doc in doc_names:
            vector = {}
            total = 0.0
            for word, count in doc_word_counts[doc].items():
                tf = count / max_freqs[doc]
                vector[word] = tf * idf[word]
                total += vector[word] ** 2
            doc_vectors[doc] = vector
            doc_norms[doc] = math.sqrt(total) if total > 0 else 1.0

        self.doc_vectors = doc_vectors
        self.doc_norms = doc_norms
        self.doc_titles = doc_titles
        self.doc_contents = doc_contents
        self.idf = idf
        self.doc_names = doc_names

        # Save TF-IDF vectors to a file
        self.save_tfidf_to_file('tfidf_vectors.txt')

    def save_tfidf_to_file(self, filename):
        # Create a mapping of English to Harari words
        word_mapping = {}
        for word in self.idf.keys():
            # Convert back to Harari script
            harari_word = english_to_harari(word)
            word_mapping[word] = harari_word

        # Sort words based on Harari script
        sorted_words = sorted(self.idf.keys(),
                            key=lambda x: word_mapping[x])

        with open(filename, 'w', encoding='utf-8') as f:
            for word in sorted_words:
                harari_word = word_mapping[word]
                idf_score = self.idf[word]
                tf_scores = []
                for doc, vector in self.doc_vectors.items():
                    if word in vector:
                        tf_scores.append(f"{doc}:{vector[word]:.4f}")
                # Write both Harari and English versions
                f.write(f"{harari_word} ({word})\t{idf_score:.4f}\t[{', '.join(tf_scores)}]\n")

search_engine = SearchEngine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def handle_search():
    data = request.get_json()
    query = data.get('query', '')
    is_exact_match = False
    original_phrase = ''

    # Check if query is a quoted phrase
    if len(query) >= 2 and query.startswith('"') and query.endswith('"'):
        is_exact_match = True
        original_phrase = query[1:-1].strip()
        # Process the phrase for searching
        english_query = transliterate_harari_to_english(original_phrase)
        stemmed_query = harari_stem(english_query.lower())
    else:
        english_query = transliterate_harari_to_english(query)
        stemmed_query = harari_stem(english_query.lower())

    # Get initial results using cosine similarity
    results = calculate_cosine_similarity(
        stemmed_query,
        search_engine.doc_vectors,
        search_engine.doc_norms,
        search_engine.idf,
        search_engine.doc_names
    )

    # Filter for exact matches if it's a quoted phrase
    if is_exact_match:
        exact_matches = []
        for doc, score in results:
            # Check document content for exact phrase match
            doc_content = ' '.join(search_engine.doc_contents.get(doc, []))
            if original_phrase in doc_content:
                exact_matches.append((doc, score))
        results = exact_matches

    # Format results
    formatted_results = []
    for doc, score in results:
        if score < 0.001:
            continue

        formatted_results.append({
            'doc': doc,
            'score': round(score, 4),
            'title': search_engine.doc_titles.get(doc, "Untitled"),
            'preview': ' '.join(search_engine.doc_contents.get(doc, [])[:3])[:150] + '...'
        })

    return jsonify({'results': formatted_results})

def calculate_cosine_similarity(query, doc_vectors, doc_norms, idf, doc_names):
    """Calculate cosine similarity between query and all documents"""
    # Process query
    query_terms = query.lower().split()
    if not query_terms:
        return []

    # Calculate query vector
    query_counts = defaultdict(int)
    for term in query_terms:
        query_counts[term] += 1
    max_count = max(query_counts.values(), default=1)

    query_vector = {}
    query_norm = 0.0
    for term, count in query_counts.items():
        tf = count / max_count
        tf_idf = tf * idf.get(term, 0)
        query_vector[term] = tf_idf
        query_norm += tf_idf ** 2
    query_norm = math.sqrt(query_norm) if query_norm > 0 else 1.0

    # Calculate similarities
    results = []
    for doc in doc_names:
        dot_product = 0.0
        for term, q_value in query_vector.items():
            dot_product += q_value * doc_vectors[doc].get(term, 0.0)

        similarity = dot_product / (query_norm * doc_norms[doc])
        results.append((doc, similarity))

    return sorted(results, key=lambda x: x[1], reverse=True)

@app.route('/api/document/<doc_id>')
def get_document(doc_id):
    try:
        # Get the document content from your corpus
        doc_title = search_engine.doc_titles.get(doc_id, "Untitled")
        doc_content = search_engine.doc_contents.get(doc_id, [])

        # Format the content with proper line breaks
        formatted_content = '<p>' + '</p><p>'.join(doc_content) + '</p>'

        return jsonify({
            'title': doc_title,
            'content': formatted_content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
