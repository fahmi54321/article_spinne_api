
# kita mengimpor library yang dibutuhkan:
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import wget

"""
•	numpy & pandas → untuk pengolahan data.
•	textwrap → digunakan untuk memformat teks agar tidak melebar keluar layar saat dicetak.
•	nltk → Natural Language Toolkit, digunakan untuk tokenizing (memecah teks menjadi kata) dan proses NLP lainnya.
•	TreebankWordDetokenizer → digunakan untuk menggabungkan kembali token menjadi kalimat utuh (detokenisasi).
"""


# Kita akan mulai dengan mengunduh dataset yang digunakan, yaitu BBC News Dataset
url = "https://raw.githubusercontent.com/fahmi54321/nlp_tfidf/refs/heads/main/bbc_text_cls.csv"
wget.download(url, 'bbc_text_cls.csv')

# Agar proses tokenisasi berjalan, kita perlu mengunduh paket punkt dari NLTK:
nltk.download('punkt')

# Kita baca file CSV hasil unduhan ke dalam DataFrame menggunakan Pandas:
df = pd.read_csv('bbc_text_cls.csv')

# Langkah berikutnya adalah menetapkan label pilihan kita. Untuk tujuan pembelajaran ini, kita akan memilih label business.
label = 'business'

"""
Selanjutnya, kita akan mengambil hanya teks dari baris-baris yang memiliki label sesuai pilihan kita. 
Seperti biasa, kita dapat membaca kodenya dari bagian dalam terlebih dahulu, 
bagian dalam memilih baris dengan label sesuai pilihan kita. Setelah itu, kita ambil kolom yang sesuai, yaitu text.
"""
texts = df[df['labels'] == label]['text']


"""
Langkah berikutnya adalah membuat model kita. Seperti yang sudah dijelaskan sebelumnya, proses ini bisa dibagi menjadi dua bagian:

1.	Menghitung jumlah kemungkinan keluaran (outcome).
2.	Menormalisasi jumlah tersebut agar menjadi probabilitas.

"""

# Bangun probabilitas kata
probs = {}
for doc in texts:
    lines = doc.split("\n")
    for line in lines:
        tokens = word_tokenize(line)
        for i in range(len(tokens) - 2):
            t_0 = tokens[i]
            t_1 = tokens[i + 1]
            t_2 = tokens[i + 2]
            key = (t_0, t_2)
            if key not in probs:
                probs[key] = {}
            probs[key][t_1] = probs[key].get(t_1, 0) + 1

# Normalisasi probabilitas
for key, d in probs.items():
    total = sum(d.values())
    for k, v in d.items():
        d[k] = v / total

"""
Seperti yang diingat, kita perlu memecah (tokenize) setiap paragraf agar bisa melakukan spinning pada artikel. Tetapi, setelah proses spinning selesai, kita harus menggabungkan kembali daftar token menjadi teks utuh.

Bagian yang agak sulit adalah: kadang token harus digabung dengan spasi, tapi kadang tidak.

•	Jika kita punya dua kata berdampingan → perlu dipisahkan dengan spasi.
•	Jika kita punya kata dan tanda baca → tidak perlu diberi spasi di antaranya.

Untuk menangani hal ini, kita akan menggunakan sebuah kelas bernama TreebankWordDetokenizer. Langkah pertama adalah membuat instance dari kelas ini:

"""
detokenizer = TreebankWordDetokenizer()

"""
Langkah berikutnya adalah mendefinisikan fungsi sample_word, yang akan mengambil (sampling) sebuah kata secara acak dari sebuah distribusi probabilitas yang direpresentasikan sebagai dictionary.
Seperti yang sudah dibahas sebelumnya, konsep ini sama seperti yang pernah kita lihat, jadi tidak akan dijelaskan secara detail lagi.
"""
def sample_word(d):
    p0 = np.random.random()
    cumulative = 0
    for t, p in d.items():
        cumulative += p
        if p0 < cumulative:
            return t
    return list(d.keys())[0]  # fallback

# Langkah berikutnya adalah mengimplementasikan fungsi spin_line.
def spin_line(line, replace_log):
    tokens = word_tokenize(line)
    i = 0
    output = [tokens[0]]
    while i < (len(tokens) - 2):
        t_0 = tokens[i]
        t_1 = tokens[i + 1]
        t_2 = tokens[i + 2]
        key = (t_0, t_2)
        p_dist = probs.get(key, {})
        if len(p_dist) > 1 and np.random.random() < 0.3:
            middle = sample_word(p_dist)
            output.append(t_1)
            output.append("<" + middle + ">")
            output.append(t_2)
            replace_log.append({"old_word": t_1, "new_word": middle})
            i += 2
        else:
            output.append(t_1)
            i += 1
    if i == len(tokens) - 2:
        output.append(tokens[-1])
    return detokenizer.detokenize(output)


"""
Langkah berikutnya adalah membuat sebuah fungsi bernama spin_document. Ide utamanya adalah fungsi ini merupakan fungsi tingkat tinggi (higher-level function). Kita akan memecah masalah besar menjadi bagian yang lebih kecil, yaitu memproses spin setiap paragraf satu per satu.

Jadi, tugas utama fungsi ini adalah:
1.	Memanggil fungsi lain untuk melakukan spin pada setiap paragraf.
2.	Menggabungkan kembali paragraf-paragraf tersebut sehingga hasil akhirnya memiliki format yang sama seperti input.

"""

def spin_document(doc):
    lines = doc.split("\n")
    output = []
    replace_log = []
    for line in lines:
        if line.strip():
            new_line = spin_line(line, replace_log)
        else:
            new_line = line
        output.append(new_line)
    return "\n".join(output), replace_log

# Flask app
app = Flask(__name__)

@app.route("/spin", methods=["POST"])
def spin():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Text input required"}), 400

    text = data["text"]
    spun_text, replaceable_words = spin_document(text)

    return jsonify({
        "spun_text": spun_text,
        "replaceable_words": replaceable_words
    })

if __name__ == "__main__":
    app.run(debug=True)
