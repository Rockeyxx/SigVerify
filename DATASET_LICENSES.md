# Dataset licences

The scope of the MIT licence in [LICENSE](LICENSE) applies exclusively to **the code hosted within this repository**. Third parties retain ownership of every signature corpus referenced herein, each subject to distinct terms. We redistribute none of these corpora: `data/` remains gitignored, and zero assets derived from a licensed dataset are committed.

## Openly available corpora

Downloading these datasets requires no signed agreement. Individual publishers define the applicable terms, with all six datasets designated for academic research.

| Corpus      | Source                                                                                                                          | Access                                                                                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| CEDAR       | [Kaggle](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)                                                           | Open download                                                                                                                                   |
| BHSig260    | [Google Drive](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E/view?resourcekey=0-MUNnTzBi4h_VE0J84NDF3Q)          | Open download                                                                                                                                   |
| UTSig       | [Kaggle](https://www.kaggle.com/datasets/sinjinir1999/utsignature-verification?resource=download)                               | Open download                                                                                                                                   |
| SVC2004     | [HKUST](https://cse.hkust.edu.hk/svc2004/download.html)                                                                         | Open download                                                                                                                                   |
| SigComp2011 | [ICDAR / IAPR TC11](<http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)>) | Open download. The archive password is the disclaimer sentence printed on the dataset page, so unzipping it is an acceptance of that disclaimer |
| MOBISIG     | [Sapientia University](https://www.ms.sapientia.ro/~manyi/mobisig.html)                                                         | Open download                                                                                                                                   |

Credit the corpus creators across any published work. Corresponding citations are listed in the README bibliography.

## DeepSignDB (signed agreement, research use only)

Distribution of DeepSignDB is managed by the **Biometrics and Data Pattern Analytics (BiDA) Lab, Universidad Autónoma de Madrid**. Access is not public; obtaining the archive involves submitting a signed license request to BiDA Lab, which usually processes requests within a few days to weeks.

A valid agreement for this project took effect on **25 July 2026** for a period of two years. **The raw archive is held locally under that agreement**. It is gitignored and has never been committed; no derived asset, figure or distributed weight incorporates it, and models trained on it carry the `DONT_SHIP` filename.

---

The binding terms of this agreement constrain repository operations through several explicit provisions:

| Clause | What it means here                                                                                                                                                                                                                                                                                                                   |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 4.1    | Usage is restricted strictly to the primary licensee and immediate institutional collaborators                                                                                                                                                                                                                                       |
| 4.2    | Redistribution or duplication beyond local backup is prohibited. Because `data/` is gitignored, no raw signatures, extracted feature caches, or vector embeddings may enter source control                                                                                                                                           |
| 4.3    | Usage is restricted to academic research. Because explicit restrictions cover "files containing extracted features", models and trained weights are bound to non-commercial terms                                                                                                                                                    |
| 4.5    | Any published findings incorporating this dataset must cite the four references marked with a scales symbol in the README bibliography                                                                                                                                                                                               |
| 4.6    | Publication of raw signature visual renderings is restricted. BiDA Lab limits permitted renderings exclusively to signers `u1010`, `u1013`, `u1028` (Development) and `u0103`, `u0145` (Evaluation). Figures across this repository present only score distributions and DET/ROC evaluation curves, avoiding direct stroke rendering |
| 4.7    | Unless formally extended, authorization terminates on 25 July 2028, requiring complete erasure of all copies                                                                                                                                                                                                                         |
| 4.8    | Immediate compliance is mandatory should BiDA Lab direct the deletion of designated database subsets                                                                                                                                                                                                                                 |

Executing model training and applying resulting weights for internal benchmarks, reports, and system demonstrations falls under authorized academic usage. Commercial distribution or integration into proprietary software remains strictly forbidden.

## Adding another licensed corpus

Operational procedures are specified in [AGENTS.md](AGENTS.md#licensed-datasets): never check in raw data or derived assets, limit visual signature renders strictly to whitelisted participants, include mandatory citations, restrict output model weights to non-commercial scope, and log agreement expiration dates. Update this document concurrently when adding any new corpus.
