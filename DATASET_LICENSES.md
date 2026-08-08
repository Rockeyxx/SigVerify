# Dataset licences

The MIT licence in [LICENSE](LICENSE) covers **the code in this repository and
nothing else**. Every signature corpus used here belongs to a third party and
carries its own terms. None of them are redistributed: `data/` is gitignored, and
nothing derived from a licensed corpus is committed.

## Openly available corpora

These can be downloaded without a signed agreement. Terms are set by each
publisher, and all six are research-use datasets.

| Corpus | Source | Access |
| --- | --- | --- |
| CEDAR | [Kaggle](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset) | Open download |
| BHSig260 | [Google Drive](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E/view?resourcekey=0-MUNnTzBi4h_VE0J84NDF3Q) | Open download |
| UTSig | [Kaggle](https://www.kaggle.com/datasets/sinjinir1999/utsignature-verification?resource=download) | Open download |
| SVC2004 | [HKUST](https://cse.hkust.edu.hk/svc2004/download.html) | Open download |
| SigComp2011 | [ICDAR / IAPR TC11](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)) | Open download. The archive password is the disclaimer sentence printed on the dataset page, so unzipping it is an acceptance of that disclaimer |
| MOBISIG | [Sapientia University](https://www.ms.sapientia.ro/~manyi/mobisig.html) | Open download |

Cite the corpus authors in any published work. The relevant references are in the
README bibliography.

## DeepSignDB (signed agreement, research use only)

DeepSignDB is distributed by the **Biometrics and Data Pattern Analytics (BiDA)
Lab, Universidad Autónoma de Madrid**. It is not a public download. You request
it, sign and return a licence agreement, and BiDA Lab sends the archive back.
Turnaround runs from days to weeks.

This project holds a copy under an agreement effective **25 July 2026**, valid two
years. **The data itself has been deleted from this machine**; the parser and its
tests remain so the work is reproducible by anyone who holds their own licence.

The agreement is specific, and these are the clauses that constrain this
repository:

| Clause | What it means here |
| --- | --- |
| 4.1 | Only the licensee and direct colleagues at the same institution may use it |
| 4.2 | No copying or redistribution beyond your own backup. `data/` is gitignored, and no signature file, feature cache or embedding dump derived from it may ever be committed |
| 4.4 | Research only. The prohibition explicitly covers "files containing extracted features", so features and weights derived from it are non-commercial too |
| 4.5 | Any public work based on it must cite the four references marked with a scales symbol in the README bibliography |
| 4.6 | No signature images in publications. BiDA Lab whitelists only users `u1010`, `u1013`, `u1028` (Development) and `u0103`, `u0145` (Evaluation). Every plot in this repository is a score distribution or a DET/ROC curve, so nothing renders a DeepSignDB stroke. Keep it that way |
| 4.7 | Expires 25 July 2028 unless renewed, after which all copies are destroyed |
| 4.8 | BiDA Lab may instruct that part of the database be destroyed. Comply immediately |

Training on it and using the resulting weights for this project, its report and
its demo is the licensed use. Selling it, or shipping it inside a commercial
product, is not.

**No released weights are trained on DeepSignDB.** Weights derived from it stay
local and carry `DONT_SHIP` in the filename. The published weights come from the
openly available corpora only.

## Adding another licensed corpus

The operational rules live in [AGENTS.md](AGENTS.md#licensed-datasets): never
commit the data or anything derived from it, never render its signatures into a
figure unless the agreement whitelists that user, carry its required citations,
treat derived weights as research-only, and record the expiry date. Add the new
corpus to this file at the same time.
