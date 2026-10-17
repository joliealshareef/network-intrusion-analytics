# Network Intrusion Analytics

Machine learning analysis of the NSL-KDD intrusion detection dataset. It compares supervised detection (Random Forest, Logistic Regression), unsupervised anomaly detection (Isolation Forest), and a layered approach that combines them. It pays particular attention to how each one handles **attack types that never appeared in training**.

**Live site:** _add your link here_
**Author:** Jolie Alshareef · B.S. Cybersecurity Analytics & Operations, Penn State

## Key results (KDDTest+, 22,544 connections)

| Model | Detection rate | False alarm rate | Precision | ROC AUC |
|---|---|---|---|---|
| Logistic Regression | 62.4% | 7.4% | 91.7% | 0.780 |
| Random Forest | 62.2% | 2.7% | 96.8% | 0.962 |
| Isolation Forest (unsupervised) | 71.9% | 7.7% | 92.5% | 0.941 |
| **Layered (RF + IF)** | **77.1%** | 8.2% | 92.6% | n/a |

- On the 17 attack types missing from training, the Random Forest caught **26.7%**. The Isolation Forest, which never saw a labeled attack, caught **71.0%**.
- Layering the two models raised overall detection from 62% to 77%, at the cost of 5.5 points of extra false alarms.
- Password guessing (`guess_passwd`) went almost entirely undetected. At the connection level it looks like a normal login, so catching it needs host logs (for example Windows event 4625).

## Run it

```bash
mkdir -p data
curl -L -o data/KDDTrain+.txt https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
curl -L -o data/KDDTest+.txt  https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt
pip install pandas scikit-learn
python analysis.py     # writes results.json and mosaic.txt (~20 s)
python build_site.py   # builds index.html from site_template.html
```

## Files

| File | What it is |
|---|---|
| `analysis.py` | Data loading, feature encoding, model training and evaluation |
| `results.json` | Every number shown on the site |
| `mosaic.txt` | One character per test connection (family + model verdicts) for the hero graphic |
| `site_template.html` | The website, with placeholders for the data |
| `build_site.py` | Injects the results into the template, producing `index.html` |
| `index.html` | The finished, self-contained website |

To add your GitHub and LinkedIn buttons, edit the `PROFILE` block near the top of the `<script>` in `site_template.html`, then run `build_site.py` again.

## Limitations

NSL-KDD comes from 1998 DARPA traffic. It's the standard teaching benchmark, but modern traffic is mostly encrypted and the attacks have changed. A natural next step is to run the same pipeline on CIC-IDS2017 or UNSW-NB15.

## Tools

Python · pandas · NumPy · scikit-learn · HTML / CSS / JavaScript (no frameworks)
