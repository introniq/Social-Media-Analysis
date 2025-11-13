import React, { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [username, setUsername] = useState("");
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState(null);
  const [error, setError] = useState("");

  const handleRun = async () => {
    if (!username.trim()) return alert("Enter a valid username");
    setLoading(true);
    setError("");
    setOutput(null);

    try {
      const res = await axios.post("http://127.0.0.1:5000/run", { username });
      setOutput(res.data);
    } catch (err) {
      setError("Error running analysis. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <nav className="navbar">
        <div className="navbar-logo">Instagram Analytics Dashboard</div>
      </nav>

      <main className="main-container">
        <div className="input-section">
          <h2>Analyze any public Instagram account</h2>
          <p>Enter a username below to fetch and analyze posts, captions, and hashtags.</p>

          <div className="input-group">
            <input
              type="text"
              placeholder="e.g. starsportsindia"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
            <button onClick={handleRun} disabled={loading}>
              {loading ? "Analyzing..." : "Run Analysis"}
            </button>
          </div>

          {error && <p className="error">{error}</p>}
        </div>

        {output && (
          <div className="results">
            <div className="summary-card">
              <h3>Analysis Summary</h3>
              <p><strong>Account:</strong> {output.username}</p>
              <p><strong>Posts Analyzed:</strong> {output.counts?.posts || 0}</p>
            </div>

            {/* ---------- Caption Type Performance ---------- */}
            {output.caption_performance?.length > 0 && (
              <div className="card">
                <h3>Caption Type Performance</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Caption Type</th>
                      <th>Adjusted Engagement</th>
                      <th>Adjusted Sentiment</th>
                      <th>Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {output.caption_performance.map((row, i) => (
                      <tr key={i}>
                        <td>{row.caption_type}</td>
                        <td>{row.adj_avg_engagement_norm}</td>
                        <td>{row.adj_avg_sentiment}</td>
                        <td>{row.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* ---------- Top Hashtags ---------- */}
            {output.top_hashtags?.length > 0 && (
              <div className="card">
                <h3>Hashtag Performance</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Hashtag</th>
                      <th>Avg Likes</th>
                      <th>Avg Sentiment</th>
                      <th>Post Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {output.top_hashtags.map((tag, i) => (
                      <tr key={i}>
                        <td>#{tag.hashtags}</td>
                        <td>{tag.avg_likes_norm}</td>
                        <td>{tag.avg_sentiment}</td>
                        <td>{tag.post_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* ---------- Per-Post Sentiments ---------- */}
            {output.per_post_sentiments?.length > 0 && (
              <div className="card">
                <h3>Post-Level Sentiment Analysis</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Shortcode</th>
                      <th>Average Comment Sentiment</th>
                    </tr>
                  </thead>
                  <tbody>
                    {output.per_post_sentiments.map((p, i) => (
                      <tr key={i}>
                        <td>{p.shortcode}</td>
                        <td>{p.avg_comment_sentiment}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
