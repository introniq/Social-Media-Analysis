from instagrapi import Client
from concurrent.futures import ThreadPoolExecutor
import json, os, sys

USERNAME = "lemme_be_kshtj"
PASSWORD = "57757557"  # only needed for first run; afterward it uses session.json
TARGET = os.getenv("TARGET") or "starsportsindia"
SESSION_FILE = "session.json"
POST_LIMIT = 10

def login_client():
    cl = Client()

    # Try using saved session first
    if os.path.exists(SESSION_FILE):
        try:
            cl.load_settings(SESSION_FILE)
            cl.get_timeline_feed()  # verify session
            print("✅ Logged in via saved session")
            return cl
        except Exception as e:
            print(f"⚠️ Session invalid, retrying login: {e}")

    # Fresh login
    try:
        cl.login(USERNAME, PASSWORD)
        cl.dump_settings(SESSION_FILE)
        print("✅ New session saved")
        return cl
    except Exception as e:
        print(f"❌ Login failed: {e}")
        sys.exit(1)


def fetch_single_post(cl, post):
    """Fetch one post and its comments."""
    try:
        comments = cl.media_comments(post.pk, amount=200)
    except Exception as e:
        print(f"⚠️ Could not fetch comments for post {post.pk}: {e}")
        comments = []

    return {
        "id": post.pk,
        "shortcode": post.code,
        "caption": post.caption_text,
        "like_count": post.like_count,
        "comment_count": len(comments),
        "comments": [{"user": c.user.username, "text": c.text} for c in comments]
    }


def fetch_posts(cl):
    """Fetch recent posts from the target user."""
    try:
        user_id = cl.user_id_from_username(TARGET)
    except Exception as e:
        print(f"❌ Could not find user '{TARGET}': {e}")
        sys.exit(1)

    posts = cl.user_medias_v1(user_id, amount=POST_LIMIT)
    print(f"📸 Found {len(posts)} posts for {TARGET}")

    data = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        for i, result in enumerate(pool.map(lambda p: fetch_single_post(cl, p), posts), start=1):
            data.append(result)
            print(f"✅ {i}/{len(posts)} fetched")

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("💾 Saved to output.json")


if __name__ == "__main__":
    cl = login_client()
    fetch_posts(cl)
