#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SITE = ROOT / "site"


def main():
    nhanes_summary = OUT / "nhanes" / "summary.json"
    summary = json.loads((nhanes_summary if nhanes_summary.exists() else OUT / "summary.json").read_text())
    SITE.mkdir(exist_ok=True)
    if summary.get("source") == "nhanes":
        return build_nhanes(summary)
    cards = "\n".join(
        f"""
        <article class="feature">
          <div class="feature-head">
            <h3>Unit {card['unit']:02d}</h3>
            <p>{card['latent'].replace('_', ' ')}</p>
          </div>
          <dl class="stats">
            <dt>latent r</dt><dd>{card['latent_correlation']:+.3f}</dd>
            <dt>active</dt><dd>{card['active_rate']:.1%}</dd>
          </dl>
          <table>
            <thead><tr><th>Top field</th><th>r</th></tr></thead>
            <tbody>
              {''.join(f"<tr><td>{f['field'].replace('_', ' ')}</td><td>{f['correlation']:+.2f}</td></tr>" for f in card['top_fields'])}
            </tbody>
          </table>
        </article>
        """
        for card in summary["cards"]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phenome SAE Biobank Pilot</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <p class="eyebrow">Biobank phenome sparse features</p>
    <h1>What does an SAE learn from LDL, diagnoses, labs, meds, and traits?</h1>
    <p class="lede">A privacy-safe synthetic pilot with known latent health factors. This validates the reporting loop before moving to UK Biobank, All of Us, FinnGen-style endpoints, or MIMIC-IV.</p>
  </header>
  <main>
    <section class="metrics">
      <div><span>final loss</span><b>{summary['final_loss']:.4f}</b></div>
      <div><span>mean active rate</span><b>{summary['mean_active_rate']:.1%}</b></div>
      <div><span>hidden units</span><b>{summary['args']['hidden']}</b></div>
      <div><span>synthetic participants</span><b>{summary['args']['n_samples']}</b></div>
    </section>
    <section class="plots">
      <img src="../outputs/loss.png" alt="Training loss plot">
      <img src="../outputs/latent_heatmap.png" alt="Sparse unit by latent factor heatmap">
      <img src="../outputs/field_heatmap.png" alt="Sparse unit by measured field heatmap">
    </section>
    <section class="grid">
      {cards}
    </section>
  </main>
</body>
</html>
"""
    css = """
:root {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f6f3ec;
  color: #1d2326;
}
body { margin: 0; }
header {
  padding: 52px clamp(20px, 5vw, 72px) 30px;
  background: #e3ecef;
  border-bottom: 1px solid #cbd8dc;
}
.eyebrow {
  margin: 0 0 10px;
  text-transform: uppercase;
  font-size: 12px;
  letter-spacing: .08em;
  color: #385c66;
  font-weight: 700;
}
h1 {
  max-width: 1080px;
  margin: 0;
  font-size: clamp(34px, 5vw, 70px);
  line-height: 1.03;
  letter-spacing: 0;
}
.lede {
  max-width: 820px;
  font-size: 19px;
  line-height: 1.5;
  color: #425057;
}
main { padding: 24px clamp(16px, 4vw, 56px) 48px; }
.metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(150px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.metrics div, .feature, .plots img {
  background: #fff;
  border: 1px solid #ded8ce;
  border-radius: 8px;
}
.metrics div { padding: 16px; }
.metrics span {
  display: block;
  color: #687173;
  font-size: 13px;
}
.metrics b {
  display: block;
  margin-top: 6px;
  font-size: 26px;
}
.plots {
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 16px;
  margin-bottom: 24px;
}
.plots img {
  width: 100%;
}
.plots img:last-child {
  grid-column: 1 / -1;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 14px;
}
.feature { padding: 16px; }
.feature-head {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  align-items: baseline;
}
.feature h3 {
  margin: 0;
  font-size: 18px;
}
.feature p {
  margin: 0;
  color: #19535f;
  font-weight: 700;
  text-align: right;
}
.stats {
  display: grid;
  grid-template-columns: 1fr auto 1fr auto;
  gap: 6px 10px;
  margin: 12px 0;
  font-size: 14px;
}
dt { color: #687173; }
dd { margin: 0; font-weight: 700; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
th, td {
  padding: 6px 0;
  border-top: 1px solid #ece6dc;
}
th {
  text-align: left;
  color: #687173;
  font-weight: 600;
}
td:last-child, th:last-child { text-align: right; font-variant-numeric: tabular-nums; }
@media (max-width: 760px) {
  .metrics, .plots { grid-template-columns: 1fr; }
  .plots img:last-child { grid-column: auto; }
  .feature-head { display: block; }
  .feature p { text-align: left; margin-top: 4px; }
}
"""
    (SITE / "index.html").write_text(html)
    (SITE / "style.css").write_text(css)
    print(SITE / "index.html")

def build_nhanes(summary):
    cards = "\n".join(
        f"""
        <article class="feature">
          <div class="feature-head">
            <h3>Unit {card['unit']:02d}</h3>
            <p>{card['anchor'].replace('_', ' ')}</p>
          </div>
          <dl class="stats">
            <dt>anchor r</dt><dd>{card['anchor_correlation']:+.3f}</dd>
            <dt>active</dt><dd>{card['active_rate']:.1%}</dd>
          </dl>
          <table>
            <thead><tr><th>Top NHANES field</th><th>r</th></tr></thead>
            <tbody>
              {''.join(f"<tr><td>{f['field'].replace('__', ' / ').replace('_', ' ')}</td><td>{f['correlation']:+.2f}</td></tr>" for f in card['top_fields'])}
            </tbody>
          </table>
        </article>
        """
        for card in summary["cards"]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NHANES Phenome SAE</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <p class="eyebrow">Public-use real phenome data</p>
    <h1>Sparse autoencoder features from NHANES labs, traits, and questionnaires</h1>
    <p class="lede">A real-data pilot using CDC public-use NHANES files. Features are interpreted by correlation with measured fields and coarse anchor groups, not as clinical diagnoses.</p>
  </header>
  <main>
    <section class="metrics">
      <div><span>final loss</span><b>{summary['final_loss']:.4f}</b></div>
      <div><span>mean active rate</span><b>{summary['mean_active_rate']:.1%}</b></div>
      <div><span>participants</span><b>{summary['n_participants']}</b></div>
      <div><span>model features</span><b>{summary['n_features']}</b></div>
    </section>
    <section class="plots">
      <img src="../{summary['plots']['loss']}" alt="Training loss plot">
      <img src="../{summary['plots']['anchor_heatmap']}" alt="Sparse unit by anchor group heatmap">
      <img src="../{summary['plots']['field_heatmap']}" alt="Sparse unit by NHANES field heatmap">
    </section>
    <section class="grid">
      {cards}
    </section>
  </main>
</body>
</html>
"""
    (SITE / "index.html").write_text(html)
    if not (SITE / "style.css").exists():
        (SITE / "style.css").write_text("")
    print(SITE / "index.html")


if __name__ == "__main__":
    main()
