import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.urlencoded({ extended: true }));

// Health check (useful for Render)
app.get('/health', (req, res) => res.status(200).send('OK'));

// Home page
app.get('/', (req, res) => {
  const indexPath = path.join(__dirname, 'public', 'index.html');
  res.sendFile(indexPath, (err) => {
    if (err) {
      console.error('Error sending index.html:', err.message);
      res.status(404).send('index.html not found');
    }
  });
});

// Analyze endpoint
app.post('/analyze', async (req, res) => {
  const sentence = (req.body.sentence || '').trim();

  if (!sentence) {
    return res.redirect('/');
  }

  const wordCount = sentence.split(/\s+/).filter(Boolean).length;

  let weirdness = 0;
  let perplexityInfo = '—';
  let errorMessage = null;

  try {
    if (!process.env.HF_TOKEN) {
      throw new Error('HF_TOKEN is not set in environment variables');
    }
const hfResponse = await fetch(
  'https://router.huggingface.co/hf-inference/models/HuggingFaceTB/SmolLM3-3B',
  {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.HF_TOKEN}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      inputs: sentence,
      parameters: {
        max_new_tokens: 1,
        details: true,  // Keep for now; if missing, code will fallback gracefully
        return_full_text: false
      }
    })
  }
);

    if (!hfResponse.ok) {
      const errorText = await hfResponse.text();
      throw new Error(`Hugging Face API error: ${hfResponse.status} - ${errorText}`);
    }

    const data = await hfResponse.json();

    if (!data?.details?.prefill?.length) {
      throw new Error('No token logprobs returned from model');
    }

    const totalLogProb = data.details.prefill
      .map(token => token.logprob || 0)
      .reduce((a, b) => a + b, 0);

    const tokenCount = data.details.prefill.length;
    const avgLogProb = totalLogProb / tokenCount;
    const ppl = Math.exp(-avgLogProb);

    weirdness = Math.min(100, Math.max(0, Math.round(
      (Math.log(ppl + 1) ** 1.7) * 11
    )));

    perplexityInfo = `Perplexity ≈ ${ppl.toFixed(1)}`;
  } catch (err) {
    console.error('HF / weirdness calculation failed:', {
      message: err.message,
      stack: err.stack ? err.stack.substring(0, 400) : 'no stack'
    });

    perplexityInfo = 'Error calculating weirdness';
    errorMessage = 'Could not reach Hugging Face. Check logs or try again later.';
  }

  // Render result page
  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <title>Weirdness Checker 2026</title>
      <link rel="stylesheet" href="/style.css" />
    </head>
    <body>
      <div class="container">
        <h1>Weirdness Checker (2026)</h1>
        <p>Enter any sentence — get word count + weirdness score (higher = more bizarre)</p>

        <form method="POST" action="/analyze">
          <textarea
            name="sentence"
            placeholder="e.g. flying broccoli eating ten thousand frogs is greater than 50 burritos"
            required
          >${escapeHtml(sentence)}</textarea>
          <button type="submit">Analyze Weirdness →</button>
        </form>

        <div class="result">
          <h3>Your sentence</h3>
          <p class="sentence">${escapeHtml(sentence)}</p>
          <ul>
            <li>Words: <strong>${wordCount}</strong></li>
            <li>Weirdness: <strong>${weirdness}/100</strong></li>
            <li>${perplexityInfo}</li>
          </ul>
          ${errorMessage ? `<p class="error">${errorMessage}</p>` : ''}
          <div class="bar">
            <div class="fill" style="width: ${weirdness}%"></div>
          </div>
        </div>

        <p class="back"><a href="/">← Try another sentence</a></p>
      </div>
    </body>
    </html>
  `);
});

// HTML escape function
function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Start server
app.listen(port, '0.0.0.0', () => {
  console.log(`Server running on port ${port} → http://0.0.0.0:${port}`);
});
