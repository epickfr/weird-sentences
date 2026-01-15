import express from 'express';
import { HfInference } from '@huggingface/inference';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

// Hugging Face client (uses HF_TOKEN from env)
const hf = new HfInference(process.env.HF_TOKEN);

// Serve static files (html + css)
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.urlencoded({ extended: true }));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/analyze', async (req, res) => {
  const sentence = (req.body.sentence || '').trim();

  if (!sentence) {
    return res.renderResult(res, sentence, null);
  }

  const wordCount = sentence.split(/\s+/).filter(Boolean).length;

  let weirdness = 0;
  let perplexityInfo = '—';

  try {
    // Use a small fast model for perplexity (2026-friendly choice)
    const response = await hf.textGeneration({
      model: 'distilgpt2', // or 'gpt2' / 'openai-community/gpt2' if you prefer
      inputs: sentence,
      parameters: {
        max_new_tokens: 1,      // we only need logprobs of the actual text
        details: true,
        return_full_text: false
      }
    });

    // HuggingFace Inference API with details=true gives us token-level logprobs
    const totalLogProb = response.details.prefill
      .map(token => token.logprob || 0)
      .reduce((a, b) => a + b, 0);

    const tokenCount = response.details.prefill.length;
    const avgLogProb = totalLogProb / tokenCount;
    const ppl = Math.exp(-avgLogProb);

    // Simple non-linear mapping → 0–100 weirdness
    weirdness = Math.min(100, Math.max(0, Math.round(
      (Math.log(ppl + 1) ** 1.7) * 11
    )));

    perplexityInfo = `Perplexity ≈ ${ppl.toFixed(1)}`;
  } catch (err) {
    console.error(err);
    perplexityInfo = 'Error calculating weirdness';
  }

  res.renderResult(res, sentence, { wordCount, weirdness, perplexityInfo });
});

// Helper to render the page with result
app.use((req, res, next) => {
  res.renderResult = (res, sentence, result) => {
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
            >${sentence ? escapeHtml(sentence) : ''}</textarea>
            <button type="submit">Analyze Weirdness →</button>
          </form>

          ${result ? `
            <div class="result">
              <h3>Your sentence</h3>
              <p class="sentence">${escapeHtml(sentence)}</p>

              <ul>
                <li>Words: <strong>${result.wordCount}</strong></li>
                <li>Weirdness: <strong>${result.weirdness}/100</strong></li>
                <li>${result.perplexityInfo}</li>
              </ul>

              <div class="bar">
                <div class="fill" style="width: ${result.weirdness}%"></div>
              </div>
            </div>
          ` : ''}
        </div>
      </body>
      </html>
    `);
  };
  next();
});

// Basic HTML escape
function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

app.listen(port, () => {
  console.log(`Server running → http://localhost:${port}`);
});
