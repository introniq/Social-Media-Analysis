from transformers import pipeline
import json, re, pandas as pd
from statistics import mean

# Load dataset
print("📌 Loading dataset...")
with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"✅ Loaded {len(data)} posts")

# Sentiment analysis
print("📌 Initializing sentiment analysis pipeline...")
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
print("✅ Sentiment pipeline ready")

# Zero-shot post classifier
print("📌 Initializing zero-shot post type classification...")
caption_type_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
print("✅ Post type classifier ready")

# Auto category inference
print("📌 Generating auto candidate labels...")
captions = [p["caption"] for p in data if p.get("caption")]
sample_text = " ".join(captions[:50])

prompt = f"""
You are an AI social media strategist.
Below are several Instagram captions. 
Analyze them and infer the 5–7 broad content categories they represent.

Each category should describe *the purpose or nature of the posts* — examples include: 
Informative, Promotional, Inspirational, Humorous, Product Showcase, Event Update, Educational, Lifestyle, Announcement, etc.

Do not describe specific hashtags — return only general category names, comma-separated, no extra text.

Captions:
{sample_text[:2000]}

Output:
"""

generator = pipeline("text2text-generation", model="google/flan-t5-base")
response = generator(prompt, max_new_tokens=64, temperature=0.4)[0]["generated_text"]
CANDIDATE_LABELS = [c.strip().title() for c in response.split(",") if c.strip()]
print("✅ Auto-generated candidate labels:", CANDIDATE_LABELS)
def extract_hashtags(caption):
    return re.findall(r"#(\w+)", caption.lower()) if caption else []

# Sentiment correction: map raw model confidence to smoother scale
def corrected_sentiment(label, score):
    if label == "POSITIVE":
        return 0.8 + 0.2 * score
    elif label == "NEGATIVE":
        return -0.8 * score
    else:
        return 0

rows = []
print("📌 Processing posts...")
for idx, post in enumerate(data, start=1):
    caption = post.get("caption", "")
    likes = post.get("like_count", 0)
    c_list = post.get("comments", [])

    # Comment sentiment
    if c_list:
        comment_texts = [c["text"] for c in c_list if c["text"].strip()]
        if comment_texts:
            sentiments = sentiment(comment_texts, truncation=True)
            sentiment_scores = [corrected_sentiment(r["label"], r["score"]) for r in sentiments]
            avg_sent = mean(sentiment_scores)
        else:
            avg_sent = 0
    else:
        avg_sent = 0

    hashtags = extract_hashtags(caption)

    # Post type classification
    caption_type = "Unknown"
    if caption.strip():
        type_result = caption_type_classifier(caption, candidate_labels=CANDIDATE_LABELS)
        caption_type = type_result["labels"][0]

    rows.append({
        "shortcode": post["shortcode"],
        "likes": likes,
        "comment_count": post.get("comment_count", len(c_list)),
        "avg_sentiment": avg_sent,
        "hashtags": hashtags,
        "caption": caption,
        "caption_type": caption_type
    })

    if idx % 10 == 0 or idx == len(data):
        print(f"Processed {idx}/{len(data)} posts")

df = pd.DataFrame(rows)
print(f"✅ DataFrame created with {len(df)} rows")

# Normalize likes by lowest nonzero avg
tags = df.explode("hashtags")
tag_stats = tags.groupby("hashtags").agg(
    avg_likes=("likes", "mean"),
    avg_sentiment=("avg_sentiment", "mean"),
    post_count=("hashtags", "count")
).reset_index()

tag_stats = tag_stats[tag_stats["hashtags"].notnull()]
min_like = max(tag_stats["avg_likes"].min(), 1)
tag_stats["avg_likes_norm"] = (tag_stats["avg_likes"] / min_like).round(2)
tag_stats["avg_sentiment"] = tag_stats["avg_sentiment"].round(2)
tag_stats["avg_likes"] = tag_stats["avg_likes"].round(2)
tag_stats["post_count"] = tag_stats["post_count"].round(2)

tag_stats_sorted = tag_stats.sort_values(by="avg_likes_norm", ascending=False)
tag_stats_sorted.to_csv("top_hashtags.csv", index=False)
print("✅ Normalized hashtag stats saved to top_hashtags.csv")

# Engagement and bias-adjusted post type stats
df["engagement_score"] = df["likes"] + (df["comment_count"] * 5)
global_avg_engagement = df["engagement_score"].mean()
global_avg_sent = df["avg_sentiment"].mean()

post_stats = df.groupby("caption_type").agg(
    sum_engagement=("engagement_score", "sum"),
    sum_sent=("avg_sentiment", "sum"),
    count=("caption_type", "count")
).reset_index()

POST_WEIGHT = 5
post_stats["adj_avg_engagement"] = (
    post_stats["sum_engagement"] + POST_WEIGHT * global_avg_engagement
) / (post_stats["count"] + POST_WEIGHT)

post_stats["adj_avg_sentiment"] = (
    post_stats["sum_sent"] + POST_WEIGHT * global_avg_sent
) / (post_stats["count"] + POST_WEIGHT)

# Normalize engagement
min_eng = max(post_stats["adj_avg_engagement"].min(), 1)
post_stats["sum_engagement_norm"] = (post_stats["sum_engagement"] / min_eng).round(2)
post_stats["adj_avg_engagement_norm"] = (post_stats["adj_avg_engagement"] / min_eng).round(2)

# Round everything
for col in ["sum_engagement", "adj_avg_engagement", "sum_sent", "adj_avg_sentiment"]:
    post_stats[col] = post_stats[col].round(2)

post_stats_sorted = post_stats.sort_values(by="adj_avg_engagement_norm", ascending=False)
post_stats_sorted.to_csv("caption_type_performance.csv", index=False)
print("✅ Normalized caption type performance saved to caption_type_performance.csv")

# Per-post comment sentiment
sentiment_analyzer = sentiment
post_sentiments = []
for post in data:
    comments = post.get("comments", [])
    if comments:
        comment_texts = [c["text"] for c in comments if c["text"].strip()]
        sentiments = sentiment_analyzer(comment_texts, truncation=True)
        sent_scores = [corrected_sentiment(r["label"], r["score"]) for r in sentiments]
        avg_sentiment = round(mean(sent_scores), 2) if sent_scores else 0
    else:
        avg_sentiment = 0
    post_sentiments.append({"shortcode": post["shortcode"], "avg_comment_sentiment": avg_sentiment})

df_sentiments = pd.DataFrame(post_sentiments)
df_sentiments.to_csv("per_post_sentiments.csv", index=False)
print("✅ Per-post comment sentiments saved to per_post_sentiments.csv")

print("✅ Analysis complete (values normalized and rounded to 2 decimals)")
